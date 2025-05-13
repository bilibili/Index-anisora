
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
from conversation import conv_openflamingo as default_conv, conv_templates
from mantis.train.data import load_data_from_config, set_ignore_index, set_default_image_token, set_default_image_token_id
from huggingface_hub import hf_hub_download
from mantis.models.openflamingo.factory import create_model_and_transforms
from mantis.models.openflamingo.processor import OpenFlamingoProcessor
from transformers.trainer import (
    is_torch_xla_available, is_sagemaker_mp_enabled, IS_SAGEMAKER_MP_POST_1_10, 
    version, accelerate_version, logger, remove_dummy_checkpoint, 
    WEIGHTS_NAME, SAFE_WEIGHTS_NAME
)
from mantis.models.openflamingo.utils import filter_state_dict_to_trainable
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

EOC = "<|endofchunk|>"
IMG = "<image>"
IGNORE_INDEX = -100

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
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models", "default": "openflamingo/OpenFlamingo-9B-vitl-mpt7b", "required": False},
        default="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    )
    tokenizer_path: Optional[str] = field(
        metadata={"help": "Path to pretrained tokenizer", "default": "anas-awadalla/mpt-7b", "required": False},
        default="mosaicml/mpt-7b",
    )
    lang_encoder_path: Optional[str] = field(
        metadata={"help": "Path to pretrained language model", "default": "anas-awadalla/mpt-7b", "required": False},
        default="mosaicml/mpt-7b",
    )
    clip_vision_encoder_path: Optional[str] = field(
        metadata={"help": "Path to pretrained vision encoder", "default": "ViT-L-14", "required": False},
        default="ViT-L-14",
    )
    clip_vision_encoder_pretrained: Optional[str] = field(
        metadata={"help": "Path to pretrained vision encoder", "default": "openai", "required": False},
        default="openai",
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
        metadata={"help": "LoRA r", "default": 64, "required": False},
        default=64,
    )
    lora_alpha: Optional[float] = field(
        metadata={"help": "LoRA alpha", "default": 128, "required": False},
        default=128,
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
    tune_xatten_layer_only: Optional[bool] = field(
        metadata={"help": "Whether to tune only the x-attention layer", "default": False, "required": False},
        default=False,
    )
    do_pretrain: Optional[bool] = field(
        metadata={"help": "Whether to pretrain the projector", "default": False, "required": False},
        default=False,
    )
    llm_backbone: Optional[str] = field(
        metadata={"help": "The LLM backbone to use", "default": "meta-llama/Meta-Llama-3-8B", "required": False},
        default="meta-llama/Meta-Llama-3-8B",
    )
    vision_backbone: Optional[str] = field(
        metadata={"help": "The vision backbone to use", "default": "openai/clip-vit-large-patch14-336", "required": False},
        default="openai/clip-vit-large-patch14-336",
    )
    conv_template: Optional[str] = field(
        metadata={"help": "The conversation template to use", "default": None, "required": False},
        default=None,
    )
    projector : Optional[str] = field(
        metadata={"help": "The projector from vision to LLM", "default": "MLP", "required": False},
        default="MLP",
    )

def load_model(model_args, training_args):
    print("Loading model...")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=model_args.clip_vision_encoder_path,
        clip_vision_encoder_pretrained=model_args.clip_vision_encoder_pretrained,
        lang_encoder_path=model_args.lang_encoder_path,
        tokenizer_path=model_args.tokenizer_path,
        cross_attn_every_n_layers=4
    )
    processor = OpenFlamingoProcessor(image_processor, tokenizer)
    
    # load pre-trained weights
    checkpoint_path = hf_hub_download(model_args.model_name_or_path, "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print("Successfully loaded model from:", model_args.model_name_or_path)
    
    
    if model_args.qlora_enabled:
        print("Qlora not supported yet")
    
    if model_args.lora_enabled or model_args.qlora_enabled:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules='.*(lang_encoder|vision_encoder|perceiver).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj|Wqkv).*$',
            lora_dropout=model_args.lora_dropout,
            use_dora=model_args.dora_enabled,
            init_lora_weights="gaussian"
        )
        model = get_peft_model(model, lora_config)
        print("Successfully added LoRA adapter to model")
        
    set_ignore_index(IGNORE_INDEX)
    set_default_image_token(IMG)
    set_default_image_token_id(model.media_token_id)
        
    return model, processor

class OpenFlamingoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_xla_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if not state_dict:
                return
            state_dict = filter_state_dict_to_trainable(self.model_wrapped, state_dict)
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.is_fsdp_enabled:
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                state_dict = self.accelerator.get_state_dict(self.model)
                if not state_dict:
                    return
                state_dict = filter_state_dict_to_trainable(self.model, state_dict)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if not state_dict:
                    return
                state_dict = filter_state_dict_to_trainable(self.model, state_dict)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
            
def main(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    if model_args.do_pretrain:
        training_args.output_dir = Path(training_args.output_dir) / model_args.llm_backbone.split("/")[-1] / training_args.run_name
    else:
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
        data_args.conv_format = conv_templates['openflamingo']
    print("Using conversation template:", data_args.conv_format)
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        raise ValueError("Data config file is required")
    
    training_args.save_safetensors = False
    trainer = OpenFlamingoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
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
    if training_args.do_predict:
        print("Predicting...")
        trainer.predict(test_dataset)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    main(training_args, data_args, model_args)