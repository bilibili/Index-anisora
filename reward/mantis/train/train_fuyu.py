from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import requests
import torch
import os
import wandb
import regex as re
from train_utils import load_json_data, load_image, load_images, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from conversation import conv_mfuyu_v1
from mantis.models.mfuyu.processor import MFuyuProcessor
from mantis.models.mfuyu.modeling_mfuyu import MFuyuForCausalLM
from mantis.train.data import load_data_from_config, set_default_image_token, set_default_image_token_id, set_ignore_index
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union, List
from accelerate import Accelerator
from pathlib import Path

os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
IGNORE_INDEX = -100
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

@dataclass
class DataArguments:
    max_seq_len: Optional[int] = field(
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                          "than this will be truncated.", "default": 1024, "required": False},
        default=1024,
    )
    data_config_file: Optional[str] = field(
        metadata={"help": "Pretrained config name or path if not the same as model_name", "default": None, "required": False},
        default=None,
    )
    dataset_balancing: Optional[bool] = field(
        metadata={"help": "Whether to balance the dataset", "default": True, "required": False},
        default=True,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models", "default": "adept/fuyu-8b", "required": False},
        default="adept/fuyu-8b",
    )
    lora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use LoRA", "default": False, "required": False},
        default=False,
    )
    qlora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use QLoRA", "default": False, "required": False},
        default=False,
    )
    dora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use Dora", "default": False, "required": False},
        default=True,
    )
    lora_r: Optional[int] = field(
        metadata={"help": "LoRA r", "default": 128, "required": False},
        default=128,
    )
    lora_alpha: Optional[float] = field(
        metadata={"help": "LoRA alpha", "default": 256, "required": False},
        default=256,
    )
    lora_dropout: Optional[float] = field(
        metadata={"help": "LoRA dropout", "default": 0.05, "required": False},
        default=0.05,
    )
    lora_bias: Optional[str] = field(
        metadata={"help": "LoRA bias", "default": 'none', "required": False},
        default='none',
    )
    attn_implementation: Optional[str] = field(
        metadata={"help": "The attention implementation to use", "default": "flash_attention_2", "required": False},
        default="flash_attention_2",
    )
    max_image_size: Optional[str] = field(
        metadata={"help": "The maximum image size", "default": "(1080,1920)", "required": False},
        default="(1080,1920)",
    )

def set_max_image_size(processor, max_image_size:str):
    max_image_size = model_args.max_image_size.strip("()").split(",")
    max_image_size = {"height": int(max_image_size[0]), "width": int(max_image_size[1])}
    print("Max image size:", max_image_size)
    patch_size = processor.image_processor.patch_size
    
    if not max_image_size['height'] % patch_size["height"] == 0:
        max_image_size['height'] = max_image_size['height'] + patch_size["height"] - max_image_size['height'] % patch_size["height"]
        print("Changed max image height to be divisible by patch height, now", max_image_size['height'])
    if not max_image_size['width'] % patch_size["width"] == 0:
        max_image_size['width'] = max_image_size['width'] + patch_size["width"] - max_image_size['width'] % patch_size["width"]
        print("Changed max image width to be divisible by patch width, now", max_image_size['width'])
    processor.image_processor.size = max_image_size

def load_model(model_args, training_args):
    print("Loading model...")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    processor = MFuyuProcessor.from_pretrained(model_args.model_name_or_path)
    set_max_image_size(processor, model_args.max_image_size)
    
    if model_args.qlora_enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    model = MFuyuForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch_dtype, 
        attn_implementation = model_args.attn_implementation,
        quantization_config=bnb_config if model_args.qlora_enabled else None,
        )
    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.language_model.resize_token_embeddings(len(processor.tokenizer))
    model.config.text_config.vocab_size = len(processor.tokenizer)
    model.config.vocab_size = len(processor.tokenizer)
    if model_args.lora_enabled:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h", "lm_head"],
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters...")
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        print("Successfully added LoRA adapters")
        
    print("Successfully loaded model from:", model_args.model_name_or_path)
    
    image_placeholder_id = processor.tokenizer("|SPEAKER|", add_special_tokens=False)["input_ids"][1]
    image_newline_id = processor.tokenizer("|NEWLINE|", add_special_tokens=False)["input_ids"][1]
    set_default_image_token_id(image_placeholder_id)
    set_default_image_token("<image>")
    set_ignore_index(IGNORE_INDEX)
    return model, processor

def main(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    training_args.output_dir = Path(training_args.output_dir) / model_args.model_name_or_path.split("/")[-1] / training_args.run_name
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(training_args.output_dir)
    training_args.remove_unused_columns = False
    data_args.is_master_worker = training_args.local_rank in [-1, 0]
    
    if not training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = True
    if training_args.resume_from_checkpoint == True:
        # search for the latest checkpoint
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [x for x in all_checkpoints if (x / "trainer_state.json").exists() and not x.name.endswith("final")]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
    
    model, processor = load_model(model_args, training_args)
    data_args.conv_format = conv_mfuyu_v1
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        raise ValueError("Data config file is required")    
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor
    )
    if trainer.is_world_process_zero():
        print("Training arguments:")
        print(training_args)
        print("Data arguments:")
        print(data_args)
        print("Model arguments:")
        print(model_args)
    if training_args.do_train:
        print("Training model...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # save
        final_checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-final')
        if model_args.lora_enabled:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), model_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(final_checkpoint_dir)
                model.save_pretrained(final_checkpoint_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(final_checkpoint_dir, 'non_lora_trainables.bin'))
        else:
            trainer.save_model(output_dir=final_checkpoint_dir)
        processor.save_pretrained(final_checkpoint_dir)
    if training_args.do_predict:
        print("Predicting...")
        trainer.predict(test_dataset)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    main(training_args, data_args, model_args)