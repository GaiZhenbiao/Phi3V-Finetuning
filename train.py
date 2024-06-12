import argparse
import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
import ujson as json
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments

from model.modeling_phi3_v import Phi3VForCausalLM, Phi3VConfig
from model.processing_phi3_v import Phi3VProcessor
from phi3v_trainer import Phi3VTrainer

import warnings

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
LLaVA_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN = "<|image_1|>"
local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="microsoft/Phi-3-vision-128k-instruct")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)

    freeze_vision_tower: bool = field(default=False)
    tune_img_projector: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=131072, # This is the default max_length for phi3-vision-128k-instruct
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    non_lora_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lora_namespan_exclude: List[str] = field(default_factory=list, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = 1



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)

def llava_to_openai(data):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for entry in data:
        transformed_entry = {
            "role": role_mapping.get(entry["from"], entry["from"]),
            "content": entry["value"].replace(LLaVA_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN),
        }
        transformed_data.append(transformed_entry)

    return transformed_data

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        processor = self.processor
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder

            if not os.path.exists(image_file):
                image_file = os.path.join(image_folder, image_file)
            image = [Image.open(image_file).convert("RGB")]
        else:
            image = None
        sources = copy.deepcopy([e["conversations"] for e in sources])
        for i in range(len(sources)):
            sources[i] = llava_to_openai(sources[i])

        prompt = processor.tokenizer.apply_chat_template(sources[0], tokenize=False)

        prompt += processor.tokenizer.eos_token

        data_dict = processor(prompt, image, return_tensors="pt")

        if self.padding:
            # data_dict = processor.tokenizer.pad(
            #     data_dict,
            #     padding="max_length",
            #     max_length=training_length,
            #     return_tensors="pt",
            # )
            if 'pixel_values' not in data_dict:
                data_dict['pixel_values'] = torch.zeros([1, 17, 3, 336, 336], dtype=torch.bfloat16)
                data_dict['image_sizes'] = torch.zeros([1, 2], dtype=torch.int64)
            
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                attention_mask=data_dict["attention_mask"][0],
                pixel_values=data_dict["pixel_values"][0],
                image_sizes=data_dict["image_sizes"][0],
                labels=data_dict["labels"][0],
                # labels=processor.tokenizer.pad(
                #     {"input_ids": data_dict["labels"][0]},
                #     padding="max_length",
                #     max_length=training_length,
                #     return_tensors="pt",
                # ).input_ids,
            )
        else:
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                attention_mask=data_dict["attention_mask"][0],
                pixel_values=data_dict["pixel_values"][0],
                image_sizes=data_dict["image_sizes"][0],
                labels=data_dict["labels"][0],
            )
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        pixel_values = [instance["pixel_values"] for instance in instances]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in instances]
        image_sizes = torch.stack(image_sizes, dim=0)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch

## Training
def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["self_attn", "lm_head"], verbose=True):
    linear_cls = torch.nn.Linear
    # lora_module_names = set()
    lora_module_names = []
    lora_namespan_exclude += ["vision_model", "img_projection"]
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():
    global local_rank
    
    ACCELERATE_USE_FSDP = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if isinstance(training_args.lora_namespan_exclude, str):
        training_args.lora_namespan_exclude = json.loads(training_args.lora_namespan_exclude)

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # rank0_print('compute_dtype', compute_dtype)
    # print(compute_dtype)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            load_in_4bit=training_args.bits==4,
            load_in_8bit=training_args.bits==8,
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["img_projection"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    config = Phi3VConfig.from_pretrained(model_args.model_id)

    if training_args.disable_flash_attn2:
        config._attn_implementation = "eager"

    model = Phi3VForCausalLM.from_pretrained(
        model_args.model_id,
        config=config,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir, 
        **bnb_model_from_pretrained_args
    )

    # rank0_print(model)

    model.config.use_cache = False

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})
    if training_args.gradient_checkpointing:
        if ACCELERATE_USE_FSDP:
            warnings.warn("``gradient_checkpointing`` may not work well with ``fsdp``. We will enable it for you. Please be sure you know what you are doing and aware of the potential errors.")
        model.enable_input_require_grads()
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        rank0_print("Gradient checkpointing:", model.model.gradient_checkpointing)

    if training_args.lora_enable:
        # lora_namespan_exclude = eval(training_args.lora_namespan_exclude)
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)
        # print(model)


    processor = Phi3VProcessor.from_pretrained(model_args.model_id, 
                                               cache_dir=training_args.cache_dir, 
                                               padding_side='right', 
                                               model_max_length=training_args.max_seq_length)

    # use unk rather than eos token to prevent endless generation
    processor.tokenizer.pad_token = processor.tokenizer.unk_token
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    processor.tokenizer.padding_side = 'right'

    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    
    # When using LoRA, the model is rapped once more.
    if training_args.lora_enable:
        vision_tower = model.model.model.vision_embed_tokens.img_processor.vision_model
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        data_args.is_multimodal = True

        if not training_args.tune_img_projector:
            for p in model.model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = False
        else:
            for p in model.model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = True

        if training_args.freeze_vision_tower:
            for p in model.model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = False
        else:
            for p in model.model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = True


        if training_args.bits in [4, 8]:
            model.model.model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=training_args.device)

    else:
        vision_tower = model.model.vision_embed_tokens.img_processor.vision_model
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        data_args.is_multimodal = True

        if not training_args.tune_img_projector:
            for p in model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = False
        else:
            for p in model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = True

        if training_args.freeze_vision_tower:
            for p in model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = False
        else:
            for p in model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = True


        if training_args.bits in [4, 8]:
            model.model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=training_args.device)

    model.config.non_lora_lr = training_args.non_lora_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_module():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module.weight = module.weight.to(torch.bfloat16)

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    trainer = Phi3VTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()