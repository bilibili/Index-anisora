import torch
from sat import AutoModel
from transformers import LlamaTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.mpu.initialize import get_model_parallel_rank
import os

def chat(model, tokenizer, 
        max_length: int = 256, num_beams=1, top_p=0.7, top_k=0, temperature=0.95):
    """from https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI/blob/main/app.py
    system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in chatbot:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s> [INST] "

    input_prompt = input_prompt + str(message) + " [/INST] "
    """
    prompt = "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n What is the capital of China? [/INST] "
    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(model.parameters().__next__().device)['input_ids'][0]
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id])
    strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id], num_beams=num_beams, consider_end=True)
    
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy
    )[0] # drop memory
    
    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq
    output_list = list(output)

    response = tokenizer.decode(output_list[0])
    if get_model_parallel_rank() == 0:
        with open(f"{torch.distributed.get_rank()}.txt", 'w') as f:
            f.write(response)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    prefix = '/mnt/shared/official_pretrains/hf_home/'
    model_type = 'Llama-2-70b-chat-hf'

    # load model
    model, model_args = AutoModel.from_pretrained(model_type[:-3].lower(), argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True
    ), url='local', overwrite_args={'model_parallel_size': 2})
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    tokenizer = LlamaTokenizer.from_pretrained(prefix+model_type)
    # tokenizer.eos_token_id = 1 # llama 2 eos_token_id is not 1
    with torch.no_grad():
        chat(model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)