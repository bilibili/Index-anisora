import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.model import GLM4VModel
import os
import json
from chat_utils import chat

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--max_length", type=int, default=2650, help='max length of the total sequence')
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help='repetition penalty, 1.0 means no penalty.')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--from_pretrained", type=str, default="visualglm-6b", help='pretrained ckpt')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = GLM4VModel.add_model_specific_args(parser)
    args = parser.parse_args()
    overwrite_args = {}
    from sat.resources.download import auto_create
    if os.path.exists(args.from_pretrained) and os.path.isdir(args.from_pretrained):
        model_path = args.from_pretrained
    else:
        model_path = auto_create(args.from_pretrained, url='local')
    with open(os.path.join(model_path, "model_config.json")) as fp:
        default_mp_size = json.load(fp)["model_parallel_size"]
    if world_size != default_mp_size:
        overwrite_args.update({'model_parallel_size': world_size})
    # load model
    model, model_args = GLM4VModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True,
        device='cuda',
        seed=1234,
        **vars(args)
    ), url='local', overwrite_args=overwrite_args)
    model = model.eval()

    torch.cuda.empty_cache()

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    from chatglm4_chat import chatglm4_text_processor_inference, glm4_tokenizer
    from blip2_img_processor import blip2_image_processor_sat_1120
    tokenizer = glm4_tokenizer("/workspace/data2/glm-4v-9b")

    text_processor_infer = chatglm4_text_processor_inference(tokenizer, args.max_length, model.image_length, model, args.english)

    if rank == 0:
        if not args.english:
            print('欢迎使用 GLM4V-CLI ，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序')
        else:
            print('Welcome to GLM4V-CLI. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')
    with torch.no_grad():
        while True:
            history = None
            cache_image = None
            if not args.english:
                if rank == 0:
                    image_path = [input("请输入图像路径或URL（回车进入纯文本对话）： ")]
                else:
                    image_path = [None]
            else:
                if rank == 0:
                    image_path = [input("Please enter the image path or URL (press Enter for plain text conversation): ")]
                else:
                    image_path = [None]
            if world_size > 1:
                torch.distributed.broadcast_object_list(image_path, 0)
            image_path = image_path[0]
            assert image_path is not None

            if image_path == 'stop':
                break
            if not args.english:
                if rank == 0:
                    query = [input("用户：")]
                else:
                    query = [None]
            else:
                if rank == 0:
                    query = [input("User: ")]
                else:
                    query = [None]
            if world_size > 1:
                torch.distributed.broadcast_object_list(query, 0)
                torch.distributed.broadcast_object_list(answer_prefix, 0)
            query = query[0]
            query = query.replace("\\n", "\n").replace("\\t", "\t") # so that we can input \t and \n from command line
            
            assert query is not None
            while True:
                if query == "clear":
                    break
                if query == "stop":
                    sys.exit(0)
                try:
                    response, history, cache_image = chat(
                        image_path, 
                        model, 
                        text_processor_infer,
                        blip2_image_processor_sat_1120,
                        query, 
                        history=history, 
                        image=cache_image, 
                        max_length=args.max_length, 
                        top_p=args.top_p, 
                        temperature=args.temperature,
                        top_k=args.top_k,
                        invalid_slices=text_processor_infer.invalid_slices,
                        repetition_penalty=args.repetition_penalty,
                        english=args.english,
                        args=args
                        )
                except Exception as e:
                    import traceback
                    print(e)
                    print(traceback.format_exc())
                    break
                if rank == 0 and not args.stream_chat:
                    if not args.english:
                        print("模型："+response)
                    else:
                        print("Model: "+response)
                image_path = None
                if not args.english:
                    if rank == 0:
                        query = [input("用户：")]
                    else:
                        query = [None]
                else:
                    if rank == 0:
                        query = [input("User: ")]
                    else:
                        query = [None]
                if world_size > 1:
                    torch.distributed.broadcast_object_list(query, 0)
                query = query[0]
                assert query is not None


if __name__ == "__main__":
    main()
