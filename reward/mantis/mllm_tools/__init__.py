MLLM_LIST = ["blip2", "instructblip", "llava", "openflamingo", "fuyu", "kosmos2", "qwenVL", "cogvlm", "mfuyu", "mllava", "idefics2", "idefics1", "emu2", "otterimage", "ottervideo", "vila", "gpt4v", "mantis", "videollava", "minicpmv"]
from functools import partial

def get_mfuyu(model_name: str):
    from .mfuyu_eval import MFuyu
    if model_name == "mantis-8b-fuyu":
        return MFuyu(model_id="TIGER-Lab/Mantis-8B-Fuyu")
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
def get_mllava(model_name: str):
    from .mllava_eval import MLlava
    if model_name == "mantis-7b-llava":
        return MLlava(model_path="TIGER-Lab/Mantis-llava-7b")
    elif model_name == "mantis-7b-bakllava":
        return MLlava(model_path="TIGER-Lab/Mantis-bakllava-7b")
    elif model_name == "mantis-8b-clip-llama3":
        return MLlava(model_path="TIGER-Lab/Mantis-8B-clip-llama3")
    elif model_name == "mantis-8b-siglip-llama3":
        return MLlava(model_path="TIGER-Lab/Mantis-8B-siglip-llama3")
    elif model_name == "mantis-8b-siglip-llama3-ablation1":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation1_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-ablation2":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation2_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-ablation3":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation3_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-ablation4":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation4_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-obelics-min":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_obelics_min_8192_lora")
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
def get_mantis_idefics2(model_name:str):
    from .idefics2_eval import Idefics2
    if model_name == "mantis-8b-idefics2_8192":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2_8192")
    elif model_name == "mantis-8b-idefics2_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-1_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2-data-ablation-1_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-2_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2-data-ablation-2_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-3_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2-data-ablation-3_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-4_8192_qlora":
        # same as mantis-8b-idefics2_8192_qlora
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2_8192_qlora")
    else:
        raise ValueError(f'Invalid model name: {model_name}')

def get_mantis_idefics3(model_name:str):
    from .idefics3_eval import MantisIdefics3
    if model_name == "mantis-8b-idefics3_16384":
        return MantisIdefics3(model_path="Mantis-VL/mantis-8b-idefics3_16384")
    elif model_name == "mantis-8b-idefics3_pure_16384":
        return MantisIdefics3(model_path="Mantis-VL/mantis-8b-idefics3-pure_16384")
    else:
        raise ValueError(f'Invalid model name: {model_name}')
 
def MLLM_Models(model_name:str):
    if model_name == "blip2":
        from .blip_flant5_eval import BLIP_FLANT5
        return BLIP_FLANT5
    elif model_name == "instructblip":
        from .instructblip_eval import INSTRUCTBLIP_FLANT5
        return INSTRUCTBLIP_FLANT5
    elif model_name == "llava":
        from .llava_eval import Llava
        return Llava
    elif model_name == "llavanext":
        from .llava_next_eval import LlavaNext
        return LlavaNext
    elif "openflamingo" in model_name.lower():
        if model_name == "openflamingo-9b":
            model_id = "openflamingo/OpenFlamingo-9B-vitl-mpt7b"
            input_type = "pretrained"
        elif model_name == "openflamingo-3b":
            model_id = "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct"
            input_type = "pretrained"
        elif model_name == "llava-9b-openflamingo":
            model_id = "Mantis-VL/llava-9b-openflamingo_2048"
            input_type = "chat"
        elif model_name == "mantis-9b-openflamingo":
            model_id = "Mantis-VL/mantis-9b-openflamingo_2048"
            input_type = "chat"
        else:
            raise ValueError(f'Invalid model name: {model_name}')
        from .openflamingo_eval import OpenFlamingo
        return partial(OpenFlamingo, model_id=model_id, input_type=input_type)
    elif model_name == "fuyu":
        from .fuyu_eval import Fuyu
        return Fuyu
    elif model_name == "kosmos2":
        from .kosmos2_eval import Kosmos2
        return Kosmos2
    elif model_name == "qwenVL":
        from .qwenVL_eval import QwenVL
        return QwenVL
    elif model_name == "cogvlm":
        from .cogvlm_eval import CogVLM
        return CogVLM
    elif model_name == "idefics2":
        from .idefics2_eval import Idefics2
        return Idefics2
    elif model_name == "idefics1":
        from .idefics1_eval import Idefics1
        return Idefics1
    elif model_name == "idefics3":
        from .idefics3_eval import Idefics3
        return Idefics3
    elif model_name == "emu2":
        from .emu2_eval import Emu2
        return Emu2
    elif model_name == "otterimage":
        from .otterimage_eval import OtterImage
        return OtterImage
    elif model_name == "ottervideo":
        from .ottervideo_eval import OtterVideo
        return OtterVideo
    elif model_name == "vila":
        from .vila_eval import VILA
        return VILA
    elif model_name == "minicpm-V-2.5":
        from .minicpmv_eval import MiniCPMV
        return MiniCPMV
    elif "videollava" in model_name.lower():
        if model_name == "videollava":
            model_id = 'LanguageBind/Video-LLaVA-7B-hf'
            input_type = "image"
        elif model_name == "videollava-image":
            model_id = 'LanguageBind/Video-LLaVA-7B-hf'
            input_type = "image"
        elif model_name == "videollava-video":
            model_id = 'LanguageBind/Video-LLaVA-7B-hf'
            input_type = "video"
        else:
            raise ValueError(f'Invalid model name: {model_name}')
        from .videollava_eval import VideoLlava
        return partial(VideoLlava, model_path=model_id, input_type=input_type)
    elif model_name.lower().startswith("gpt4v"):
        from .gpt4v_eval import GPT4V
        return GPT4V
    elif model_name.lower().startswith("mantis"):
        if "fuyu" in model_name.lower():
            return partial(get_mfuyu, model_name=model_name)
        elif "idefics2" in model_name.lower():
            return partial(get_mantis_idefics2, model_name=model_name)
        elif "idefics3" in model_name.lower():
            return partial(get_mantis_idefics3, model_name=model_name)
        elif "openflamingo" in model_name.lower():
            raise NotImplementedError
        else:
            return partial(get_mllava, model_name=model_name)
    else:
        raise ValueError(f'Invalid model name: {model_name}, must be one of {MLLM_LIST}')
    
