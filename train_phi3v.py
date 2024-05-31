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
from peft import LoraConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

# Argument parser
parser = argparse.ArgumentParser(description="Script for training Phi3-V with PEFT")
parser.add_argument("--data_path", type=str, required=True, help="Path to the llava formatted training data (a json file).")
parser.add_argument("--image_folder", type=str, required=True, help="Path to the images folder (as referenced in the llava formatted training data).")
parser.add_argument("--model_id", type=str, required=True, help="Path to Phi3-V model.")
parser.add_argument("--proxy", type=str, default=None, help="Proxy settings (default: None).")
parser.add_argument("--output_dir", type=str, default="output/test_train", required=False, help="Output directory for model checkpoints.")
parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
parser.add_argument("--deepspeed_config", type=str, default="scripts/zero2.json", help="Path to DeepSpeed config file.")
parser.add_argument("--target_modules", type=int, default=-1, help="Number of target modules to train (-1 means all layers).")
parser.add_argument("--max_seq_length", type=int, default=3072, help="Maximum sequence length.")
parser.add_argument("--quantization", action='store_true', help="Enable quantization.")
parser.add_argument("--report_to", type=str, choices=['tensorboard', 'wandb'], default='tensorboard', help="Reporting tool (tensorboard or wandb).")
parser.add_argument("--logging_dir", type=str, default="./tf-logs", help="Logging directory.")
parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank.")
parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha.")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps.")
parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of data loader workers.")
args = parser.parse_args()

if args.proxy:
    os.environ["HTTP_PROXY"] = args.proxy
    os.environ["HTTPS_PROXY"] = args.proxy

logger = logging.getLogger(__name__)

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float16
print(compute_dtype)

quantization_config = None
if args.quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["img_projection"],
    )

processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

# Prepare Dataset
# Load LLaVA formatted dataset, this step may take a while if the dataset is large

with open(args.data_path, "r") as f:
    train_data = json.load(f)

IMAGE_TOKEN_INDEX = -200

local_rank = None


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


LLaVA_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN = "<|image_1|>"
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
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
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
        processor = self.data_args.image_processor
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

        prompt = processor.tokenizer.apply_chat_template(
            sources, tokenize=False, add_generation_prompt=True
        )
        data_dict = processor(prompt[0], image, return_tensors="pt")

        if self.padding:
            training_length = args.max_seq_length
            data_dict = processor.tokenizer.pad(
                data_dict,
                padding="max_length",
                max_length=training_length,
                return_tensors="pt",
            )
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                attention_mask=data_dict["attention_mask"][0],
                pixel_values=data_dict["pixel_values"][0],
                image_sizes=data_dict["image_sizes"][0],
                labels=processor.tokenizer.pad(
                    {"input_ids": data_dict["labels"][0]},
                    padding="max_length",
                    max_length=training_length,
                    return_tensors="pt",
                ).input_ids,
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


data_args = DataArguments(
    data_path=args.data_path,
    is_multimodal=True,
    image_folder=args.image_folder,
)
data_args.image_processor = processor
aitw_train = LazySupervisedDataset(
    data_path=train_data, tokenizer=processor.tokenizer, data_args=data_args
)


## Training
def find_all_linear_names(model):
    cls = torch.nn.Linear
    # lora_module_names = set()
    lora_module_names = []
    multimodal_keywords = ["vision_model", "img_projection"]
    for name, module in model.named_modules():
        # print(name, end="\n\n")
        # print(module, end="\n==========\n")
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls) and "qkv_proj" in name:
            names = name.split(".")
            name_to_add = names[0] if len(names) == 1 else names[-1]
            # print(f"Adding {name_to_add} from {name}")
            # lora_module_names.add(name)
            lora_module_names.append(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    # return list(lora_module_names)
    return lora_module_names


target_modules = [
    "model.layers.0.self_attn.qkv_proj",
    "model.layers.1.self_attn.qkv_proj",
    "model.layers.2.self_attn.qkv_proj",
    "model.layers.3.self_attn.qkv_proj",
    "model.layers.4.self_attn.qkv_proj",
    "model.layers.5.self_attn.qkv_proj",
    "model.layers.6.self_attn.qkv_proj",
    "model.layers.7.self_attn.qkv_proj",
    "model.layers.8.self_attn.qkv_proj",
    "model.layers.9.self_attn.qkv_proj",
    "model.layers.10.self_attn.qkv_proj",
    "model.layers.11.self_attn.qkv_proj",
    "model.layers.12.self_attn.qkv_proj",
    "model.layers.13.self_attn.qkv_proj",
    "model.layers.14.self_attn.qkv_proj",
    "model.layers.15.self_attn.qkv_proj",
    "model.layers.16.self_attn.qkv_proj",
    "model.layers.17.self_attn.qkv_proj",
    "model.layers.18.self_attn.qkv_proj",
    "model.layers.19.self_attn.qkv_proj",
    "model.layers.20.self_attn.qkv_proj",
    "model.layers.21.self_attn.qkv_proj",
    "model.layers.22.self_attn.qkv_proj",
    "model.layers.23.self_attn.qkv_proj",
    "model.layers.24.self_attn.qkv_proj",
    "model.layers.25.self_attn.qkv_proj",
    "model.layers.26.self_attn.qkv_proj",
    "model.layers.27.self_attn.qkv_proj",
    "model.layers.28.self_attn.qkv_proj",
    "model.layers.29.self_attn.qkv_proj",
    "model.layers.30.self_attn.qkv_proj",
    "model.layers.31.self_attn.qkv_proj",
]
if args.target_modules != -1:
    target_modules = target_modules[-args.target_modules:]
print("Target module count: ", len(target_modules))
peft_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
# model.add_adapter(peft_config)

# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )
# print_trainable_parameters(model)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    bf16=True,
    deepspeed=args.deepspeed_config,
    dataloader_num_workers=args.dataloader_num_workers,
    report_to=args.report_to,
    logging_dir=args.logging_dir,
    logging_steps=args.logging_steps,
)

trainer = SFTTrainer(
    model=args.model_id,
    tokenizer=None,
    args=training_args,
    train_dataset=aitw_train,
    eval_dataset=None,
    peft_config=peft_config,
    max_seq_length=args.max_seq_length,
    packing=True,
    model_init_kwargs=dict(
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        _attn_implementation="eager",
        quantization_config=quantization_config,
    ),
)

trainer.train()

trainer.save_model(args.output_dir)
