import torch
from sat import AutoModel
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import stream_filling_sequence
from sat.generation.sampling_strategies import BaseStrategy

def chat(model, tokenizer, 
        max_length: int = 256, top_p=0.7, top_k=0, temperature=0.95):
    prompt = "[Round 0]\n\n问：你好\n\n答："
    inputs = tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id])

    filling_stream = stream_filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy
    )
    offset = 0
    for tokens, mems in filling_stream:
        tmp_response = tokenizer.decode(tokens[0])
        if tmp_response[-1] != "�":
            print(tmp_response[offset:], end='')
            offset = len(tmp_response)
    print()
    output = strategy.finalize(tokens, mems)[0]

    response = tokenizer.decode(output[0])
    print("final:", response)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained('chatglm2-6b', args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
    ))
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    chat(model, tokenizer, max_length=args.max_length, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)

            
