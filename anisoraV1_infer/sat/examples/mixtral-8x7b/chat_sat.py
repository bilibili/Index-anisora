import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
import os
from sat.model import AutoModel

from sat.generation.autoregressive_sampling import filling_sequence, stream_filling_sequence, BaseStrategy, get_masks_and_position_ids_default
from sat.mpu import get_model_parallel_rank

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help='repetition penalty, 1.0 means no penalty.')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="visualglm-6b", help='pretrained ckpt')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    from transformers import AutoTokenizer

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda',
        **vars(args)
    ), url='local', overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()
    torch.cuda.empty_cache()
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())


    def chat_history_to_prompt(history, query):
        prompt = " [INST] "
        for i, (old_query, response) in enumerate(history):
            prompt += old_query + " [/INST] " + response + "</s> [INST] "
        prompt += query + " [/INST] "
        return prompt
    

    history = []
    with torch.no_grad():
        while True:
            if rank == 0:
                query = [input("User: ")]
            else:
                query = [None]
            if world_size > 1:
                torch.distributed.broadcast_object_list(query, 0)
            while query[0] == "clear" or query[0] == "stop":
                if query[0] == "stop":
                    exit()
                history = []
                if rank == 0:
                    query = [input("User: ")]
                else:
                    query = [None]
                if world_size > 1:
                    torch.distributed.broadcast_object_list(query, 0)
            prompt = chat_history_to_prompt(history, query[0])
            inputs = tokenizer(prompt, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].cuda()
            origin = inputs["input_ids"][0]
            seq = torch.cat(
                [origin, torch.tensor([-1]*(args.max_length-len(origin)), device=origin.device)], dim=0
            )
            strategy = BaseStrategy(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, end_tokens=[tokenizer.eos_token_id])
            if args.stream_chat:
                filling_stream = stream_filling_sequence(
                    model, seq,
                    batch_size=1,
                    strategy=strategy,
                )
                if get_model_parallel_rank() == 0:
                    print("Model: ", end='')
                offset = len(tokenizer.decode(origin))
                for tokens, mems in filling_stream:
                    torch.cuda.empty_cache()
                    tmp_response = tokenizer.decode(tokens[0])
                    if tmp_response[-1] != "�":
                        if get_model_parallel_rank() == 0:
                            print(tmp_response[offset:], end='')
                        offset = len(tmp_response)
                if get_model_parallel_rank() == 0:
                    print()
                output = strategy.finalize(tokens, mems)[0]

                response = tokenizer.decode(output[0])

            else:
                output = filling_sequence(
                    model, seq,
                    batch_size=1,
                    strategy=strategy,
                )[0]
                if type(output) is not list:
                    output_list = output.tolist()
                else:
                    output_list = output

                response = tokenizer.decode(output_list[0])
            history.append((query[0], response[len(tokenizer.decode(origin)):]))
            print(history)