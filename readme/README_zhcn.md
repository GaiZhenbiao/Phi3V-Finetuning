<div align="right">
  简体中文 | <a title="English" href="../README.md">English</a></a>
</div>

# 使用 PEFT 训练 Phi3-V

本仓库包含一个使用参数高效微调（PEFT）技术训练 [Phi3-V 模型](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)的脚本，支持各种配置和选项。

## 目录

- [使用 PEFT 训练 Phi3-V](#使用-peft-训练-phi3-v)
  - [目录](#目录)
  - [安装](#安装)
    - [使用 `requirements.txt`](#使用-requirementstxt)
    - [使用 `environment.yml`](#使用-environmentyml)
  - [模型下载](#模型下载)
  - [使用方法](#使用方法)
  - [参数](#参数)
  - [数据集准备](#数据集准备)
  - [TODO](#todo)
  - [许可证](#许可证)
  - [引用](#引用)

## 安装

使用 `requirements.txt` 或 `environment.yml` 安装所需的软件包。

### 使用 `requirements.txt`

```bash
pip install -r requirements.txt
```

### 使用 `environment.yml`

```bash
conda env create -f environment.yml
conda activate phi3v
```

## 模型下载

在训练之前，从 HuggingFace 下载 Phi3-V 模型。建议使用 `huggingface-cli` 进行下载。

1. 安装 HuggingFace CLI：

```bash
pip install -U "huggingface_hub[cli]"
```

2. 下载模型：

```bash
huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir Phi-3-vision-128k-instruct --resume-download
```

3. 替换模型文件：

将 `Phi-3-vision-128k-instruct` 目录下的模型文件替换为 `overwrites/Phi-3-vision-128k-instruct` 目录下的文件。

## 使用方法

运行训练脚本，请使用以下命令：

```bash
bash scripts/train.sh
```

**注意：** 请记得将 `train.sh` 中的路径替换为你的具体路径。

## 参数

- `--data_path` (str): LLaVA 格式的训练数据路径（JSON 文件）。**（必需）**
- `--image_folder` (str): LLaVA 格式的训练数据中引用的图像文件夹路径。**（必需）**
- `--model_id` (str): Phi3-V 模型的路径。**（必需）**
- `--proxy` (str): 代理设置（默认：无）。
- `--output_dir` (str): 模型checkpoint的输出目录（默认："output/test_train"）。
- `--num_train_epochs` (int): 训练epoch数（默认：1）。
- `--batch_size` (int): 训练批量大小（默认：1）。
- `--gradient_accumulation_steps` (int): 梯度累积步骤（默认：4）。
- `--deepspeed_config` (str): DeepSpeed 配置文件的路径（默认："scripts/zero2.json"）。
- `--target_modules` (int): 要训练的目标模块数（-1 表示所有层）（默认：-1）。
- `--max_seq_length` (int): 最大序列长度（默认：3072）。
- `--quantization` (flag): 启用量化。
- `--report_to` (str): 报告工具（选项：'tensorboard', 'wandb'）（默认：'tensorboard'）。
- `--logging_dir` (str): 日志目录（默认："./tf-logs"）。
- `--lora_rank` (int): LoRA rank（默认：128）。
- `--lora_alpha` (int): LoRA alpha（默认：256）。
- `--lora_dropout` (float): LoRA dropout（默认：0.05）。
- `--logging_steps` (int): 每隔几步打印日志（默认：1）。
- `--dataloader_num_workers` (int): 数据加载器工作线程数（默认：4）。

## 数据集准备

该脚本需要按照 LLaVA 规范格式化的数据集。数据集应为 JSON 文件，每个条目包含对话和图像信息。确保数据集中图像路径与提供的 `--image_folder` 相匹配。

<details>
<summary>数据集示例</summary>

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
- [ ] 添加对 DeepSpeed ZeRO-3 的支持。
- [ ] 添加对 FSDP 的支持。
- [ ] 添加对全量微调的支持。

## 许可证

本项目使用 Apache-2.0 许可证。详见 [LICENSE](LICENSE) 文件。

本项目借用了 [LLaVA](https://github.com/haotian-liu/LLaVA) 和 [Microsoft Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) 的代码。感谢这两个项目的贡献。

## 引用

如果你在工作中使用了这个代码库，请引用本项目：

```bibtex
@misc{phi3vfinetuning2023,
  author = {Gai Zhenbiao},
  title = {Phi3V-Finetuning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/GaiZhenbiao/Phi3V-Finetuning},
  note = {GitHub repository},
}
```