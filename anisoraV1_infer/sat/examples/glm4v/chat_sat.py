import torch
from sat.model import GLM4VModel
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy
from blip2_img_processor import blip2_image_processor_sat_1120
from PIL import Image
import argparse

from functools import partial
def get_masks_and_position_ids(seq, image_logits_mask):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    position_ids = []
    pid = -1
    for i in range(len(image_logits_mask[0])):
        if image_logits_mask[0][i] == 0 or (i > 0 and image_logits_mask[0][i] != image_logits_mask[0][i - 1]):
            pid += 1
        position_ids.append(pid)
    for i in range(tokens.shape[1]-image_logits_mask.shape[1]):
        pid += 1
        position_ids.append(pid)
    position_ids = torch.tensor(position_ids, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

def chat(model, tokenizer, 
        max_length: int = 2650, top_p=0.4, top_k=1, temperature=0.8):
    assert model.image_length == 1600
    device = next(model.parameters()).device
    query = '描述这张图片'
    image = Image.open("image path").convert('RGB')
    img_inputs = blip2_image_processor_sat_1120(image)
    for k in img_inputs:
        if k == 'image':
            img_inputs[k] = img_inputs[k].bfloat16()
        img_inputs[k] = img_inputs[k].to(device)
    prompt = f"<|user|>{query}<|assistant|>"
    inputs = tokenizer([prompt])['input_ids'][0]
    inputs = torch.tensor(inputs[:3] + [0] * (model.image_length+2) + inputs[3:]).to(device)

    image_embed_mask = [0] * len(inputs)
    image_embed_mask[3:3+model.image_length+2] = [1] * (model.image_length+2)
    image_embed_mask = torch.tensor(image_embed_mask).unsqueeze(0).to(device)
    rope_mask = [0] * len(inputs)
    rope_mask[4:4+model.image_length] = [1] * model.image_length
    rope_mask = torch.tensor(rope_mask).unsqueeze(0).to(device)
    pos_func = partial(get_masks_and_position_ids, image_logits_mask=rope_mask)
    additional_inputs = {"image_embed_mask": image_embed_mask, **{"vision_"+k:v for k,v in img_inputs.items()}}

    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=1, end_tokens=[tokenizer.eos_token_id])
    output = filling_sequence(
        model, seq,
        batch_size=1,
        get_masks_and_position_ids=pos_func,
        strategy=strategy,
        **additional_inputs
    )[0] # drop memory
    output_list = output.tolist()[0][len(inputs):-1]

    response = tokenizer.decode(output_list)
    print(response)

def main():
    # load model
    model, model_args = GLM4VModel.from_pretrained('glm4v-9b-chat', args=argparse.Namespace(
        fp16=False,
        bf16=True,
        skip_init=True,
        use_gpu_initialization=True,
        mode='inference',
        world_size=1,
        model_parallel_size=1,
        rank=0,
        local_rank=0,
        deepspeed=None,
        seed=1234,
        device='cuda'
    ))
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    torch.cuda.empty_cache()
    with torch.no_grad():
        chat(model, tokenizer)

            
if __name__ == "__main__":
    main()
