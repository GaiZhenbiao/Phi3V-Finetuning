<div align="right">
  English | <a title="简体中文" href="./readme/README_zhcn.md">简体中文</a></a>
</div>

# Training Phi3-V with PEFT

This repository contains a script for training the [Phi3-V model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) with Parameter-Efficient Fine-Tuning (PEFT) techniques using various configurations and options.

## Table of Contents

- [Training Phi3-V with PEFT](#training-phi3-v-with-peft)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Using `requirements.txt`](#using-requirementstxt)
    - [Using `environment.yml`](#using-environmentyml)
  - [Model Download](#model-download)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Dataset Preparation](#dataset-preparation)
  - [TODO](#todo)
  - [License](#license)
  - [Citation](#citation)

## Installation

Install the required packages using either `requirements.txt` or `environment.yml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt
```

### Using `environment.yml`

```bash
conda env create -f environment.yml
conda activate phi3v
```

## Model Download

Before training, download the Phi3-V model from HuggingFace. It is recommended to use the `huggingface-cli` to do this.

1. Install the HuggingFace CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

2. Download the model:

```bash
huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir Phi-3-vision-128k-instruct --resume-download
```

3. Replace the modeling files:

Replace the modeling files under `Phi-3-vision-128k-instruct` with the ones under `overwrites/Phi-3-vision-128k-instruct`.

## Usage

To run the training script, use the following command:

```bash
bash scripts/train.sh
```

**Note:** Remember to replace the paths in `train.sh` with your specific paths.

## Arguments

- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Phi3-V model. **(Required)**
- `--proxy` (str): Proxy settings (default: None).
- `--output_dir` (str): Output directory for model checkpoints (default: "output/test_train").
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--deepspeed_config` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 3072).
- `--quantization` (flag): Enable quantization.
- `--disable_flash_attn2` (flag): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.

<details>
<summary>Example Dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```
</details>

## TODO
- [ ] Add support for DeepSpeed ZeRO-3.
- [ ] Add support for FSDP
- [ ] Add support for full finetuning
- [ ] Add support for grounded finetuning
- [ ] Add support for multi-image finetuning
- [ ] Intergration with [Chuanhu Chat](https://github.com/GaiZhenbiao/ChuanhuChatGPT)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

This project borrowed code from [LLaVA](https://github.com/haotian-liu/LLaVA) and [Microsoft Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct). Thanks to both projects for their contributions.

## Citation

If you use this codebase in your work, please cite this project:

```bibtex
@misc{phi3vfinetuning2023,
  author = {Gai Zhenbiao & Shao Zhenwei},
  title = {Phi3V-Finetuning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/GaiZhenbiao/Phi3V-Finetuning},
  note = {GitHub repository},
}
```
