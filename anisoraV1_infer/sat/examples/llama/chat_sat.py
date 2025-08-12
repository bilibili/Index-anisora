import torch
from sat import AutoModel
from transformers import LlamaTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy

def chat(model, tokenizer, 
        max_length: int = 256, num_beams=1, top_p=0.7, top_k=0, temperature=0.95):
    prompt = "The capital of China is"
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
    print(response)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained('llama-7b', args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
    ))
    model = model.eval()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    if not hasattr(tokenizer, 'eos_token_id'):
        tokenizer.eos_token_id = 1
    with torch.no_grad():
        chat(model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)

            
