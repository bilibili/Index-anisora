
import dataclasses
from dataclasses import dataclass, field
import torch
import os
import wandb
import regex as re
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from transformers.hf_argparser import HfArgumentParser
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from conversation import conv_templates
from mantis.train.data import (
    load_data_from_config, 
    set_ignore_index, set_default_image_token, 
    set_default_image_token_id,
    set_default_video_token,
    set_default_video_token_id,
    ClassificationDataset,
)
from pathlib import Path
from typing import Optional
from pathlib import Path

os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
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
        default=False,
    )
    use_video_encoder: Optional[bool] = field(
        metadata={"help": "Whether to use video encoder", "default": True, "required": False},
        default=True,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models", "default": "Qwen/Qwen2-VL-7B-Instruct", "required": False},
        default="Qwen/Qwen2-VL-7B-Instruct",
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
        metadata={"help": "LoRA r", "default": 8, "required": False},
        default=8,
    )
    lora_alpha: Optional[float] = field(
        metadata={"help": "LoRA alpha", "default": 8, "required": False},
        default=8,
    )
    lora_dropout: Optional[float] = field(
        metadata={"help": "LoRA dropout", "default": 0.1, "required": False},
        default=0.1,
    )
    lora_bias: Optional[str] = field(
        metadata={"help": "LoRA bias", "default": 'none', "required": False},
        default='none',
    )
    attn_implementation: Optional[str] = field(
        metadata={"help": "The attention implementation to use", "default": "flash_attention_2", "required": False},
        default="flash_attention_2",
    )
    conv_template: Optional[str] = field(
        metadata={"help": "The conversation template to use", "default": None, "required": False},
        default=None,
    )
    num_labels: Optional[int] = field(
        metadata={"help": "The number of labels", "default": None, "required": False},
        default=None,
    )
    problem_type: Optional[str] = field(
        metadata={"help": "The problem type", "default": "generation", "required": False, "choices": ["regression", "single_label_classification", "multi_label_classification", "generation"]},
        default="generation",
    )
    min_pixels: Optional[int] = field(
        metadata={"help": "The minimum number of pixels", "default": 16, "required": False},
        default=256,
    )
    max_pixels: Optional[int] = field(
        metadata={"help": "The maximum number of pixels", "default": 256, "required": False},
        default=1280,
    )
    score_type: Optional[str] = field(
        metadata={"help": "The label prefix", "default": "end_token_mlp", "required": False},
        default="end_token_mlp",
    )
    

def load_model(model_args, training_args):
    print("Loading model...")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    from transformers import Qwen2VLProcessor, AutoConfig
    from mantis.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLForSequenceClassification
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, do_image_splitting=False,
        min_pixels=model_args.min_pixels * 28 * 28, max_pixels=model_args.max_pixels * 28 * 28
    ) # seems high vmem usage when image splitting is enabled
    
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
    
    assert model_args.problem_type in ["regression", "single_label_classification", "multi_label_classification", "generation"]
    if model_args.problem_type == "generation": 
        print("Using generation model")
        MODEL_CLASS = Qwen2VLForConditionalGeneration
        model_init_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": model_args.attn_implementation,
            "quantization_config": bnb_config,
        }
    else:
        print("Using classification model")
        MODEL_CLASS =  Qwen2VLForSequenceClassification
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, problem_type=model_args.problem_type, num_labels=model_args.num_labels)
        model_init_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": model_args.attn_implementation,
            "quantization_config": bnb_config,
            # "problem_type": model_args.problem_type,
            # "num_labels": model_args.num_labels,
        }
        if model_args.score_type:
            if model_args.score_type == "end_token_mlp":
                # default
                pass 
            elif model_args.score_type == "special_token":
                print("Using special token score type")
                print("Previous tokenizer size:", len(processor.tokenizer))
                label_special_tokens = [f"<|LABEL_{i}|>" for i in range(1, model_args.num_labels + 1)]
                processor.tokenizer.add_tokens(label_special_tokens, special_tokens=True)
                label_special_token_ids = processor.tokenizer.convert_tokens_to_ids(label_special_tokens)
                # model_init_kwargs["score_type"] = "special_token" 
                # model_init_kwargs["label_special_tokens"] = label_special_tokens
                # model_init_kwargs["label_special_token_ids"] = label_special_token_ids
                config.score_type = "special_token"
                config.label_special_tokens = label_special_tokens
                config.label_special_token_ids = label_special_token_ids
                print(config)
                model_init_kwargs["config"] = config
                print("Added special tokens for labels:", label_special_tokens)
                print("Special tokens ids:", label_special_token_ids)
                print("New tokenizer size:", len(processor.tokenizer))
                processor.tokenizer.label_special_tokens = label_special_tokens
                processor.tokenizer.score_type = "special_token"
            else:
                raise ValueError("Invalid score type")
        if model_args.lora_enabled or model_args.qlora_enabled:
            raise ValueError("LoRA and QLoRA are not supported for Qwen2VLForSequenceClassification for now")
    
    
    model = MODEL_CLASS.from_pretrained(
        model_args.model_name_or_path,
        **model_init_kwargs
    )
    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print("Successfully loaded model from:", model_args.model_name_or_path)
    
    if model_args.lora_enabled or model_args.qlora_enabled:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules='.*(visual|model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            lora_dropout=model_args.lora_dropout,
            use_dora=model_args.dora_enabled,
            init_lora_weights="gaussian"
        )
        model = get_peft_model(model, lora_config)
    
    # keep the vision backbone frozen all the time
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
        
    set_ignore_index(-100)
    set_default_image_token_id(model.config.image_token_id)
    set_default_video_token_id(model.config.video_token_id)
    set_default_image_token("<image>") # this will be transformed to <|vision_start|><|image_pad|><|vision_end|> in the conversation template of qwen2_vl
    set_default_video_token("<video>") # this will be transformed to <|vision_start|><|video_pad|><|vision_end|> in the conversation template of qwen2_vl
    
    # resize token embeddings
    if len(processor.tokenizer) > model.config.vocab_size:
        print("Tokenizer size:", len(processor.tokenizer))
        print("Model vocab size:", model.config.vocab_size)
        print("Resizing token embeddings to match tokenizer size")
        model.resize_token_embeddings(len(processor.tokenizer))
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
    
    if model_args.conv_template:
        data_args.conv_format = conv_templates[model_args.conv_template] 
    else:
        data_args.conv_format = conv_templates["qwen2_vl"]
    print("Using conversation template:", data_args.conv_format)
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        raise ValueError("Data config file is required")
    
    if model_args.problem_type != "generation":
        assert all([isinstance(x, ClassificationDataset) for x in train_dataset.datasets])
    
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