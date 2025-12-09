# LLaMA-3.2-3B Instruction LoRA Fine-Tuning Notebook

This repository contains workflow notebooks for fine-tuning Meta LLaMA 3.2 (3B) using **LoRA + QLoRA** on an instruction dataset derived from the **Databricks Dolly** corpus. The notebooks demonstrates **parameter-efficient fine-tuning**, including dataset preprocessing, training, and evaluation.

You can run the full fine-tuning workflow **on a free-tier Google Colab T4 GPU**.

---

## Notebook

- **Training Notebook:** [Colab link](https://colab.research.google.com/drive/1awDTodQo_5eYVC0OeU8IhQNYSsR7SBO2?usp=sharing)  

## Hugging Face Model

The LoRA adapter, tokenizer, and dataset are available on Hugging Face:

- **Repo:** [MagicaNeko/llama-3b-lora-dolly](https://huggingface.co/MagicaNeko/llama-3b-lora-dolly)

## Model Details
- **Base Model:** [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- **Model Type:** LoRA adapter for causal language modeling
- **Finetuning Method:** LoRA + QLoRA (4-bit quantization)
- **Trainable Parameters:** ~0.33% via LoRA (rank=8)
- **Quantization:** 4-bit NF4 
- **Language:** English

<img src="assets/finetuning workflow.png" width="600">


## Training Data
- **Dataset:** [databricks-dolly-1k](https://huggingface.co/datasets/MagicaNeko/databricks-dolly-1k) subset of [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- **Samples:** 982 training, 110 validation
- **Split:** 90% train/ 10% validation
- **Text Field:** "text" (instruction + response pairs)

## Training Configuration

### LoRA Hyperparameters
```json
{
  "r": 8,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

### Training Hyperparameters
```json
{
  "num_train_epochs": 3,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 2e-4,
  "weight_decay": 0.001,
  "warmup_ratio": 0.03,
  "lr_scheduler_type": "constant",
  "max_seq_length": 256,
  "optim": "paged_adamw_32bit",
  "gradient_checkpointing": true,
  "eval_strategy": "steps",
  "eval_steps": 100,
  "save_steps": 100,
  "logging_steps": 100,
  "packing": true
}
```

### QLoRA Hyperparameters
```json
{
  "load_in_4bit": true,
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_use_double_quant": false
}
```

### Training Setup
```json
{
  "framework": "PyTorch with Hugging Face Transformers",
  "fine_tuning_method": "LoRA (Low-Rank Adaptation)",
  "quantization": "4-bit NF4 with QLoRA",
  "compute": "Google Colab T4",
  "gpu_memory": "~16GB VRAM",
}
```

## Usage

### Installation
```bash
pip install torch transformers accelerate peft bitsandbytes
```

### Basic Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
model_name = "meta-llama/Llama-3.2-3B"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    return_dict=True,
    device_map="auto",
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "MagicaNeko/llama-3b-lora-dolly",
    subfolder="model-ft-tokenizer"
)

# Load LoRA Adapter
model = PeftModel.from_pretrained(
    base_model,
    "MagicaNeko/llama-3b-lora-dolly",
    subfolder="model-ft-lora-adapter"
)

# Merge adapters
model = model.merge_and_unload()

# Inference
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))```
```

## License
This LoRA adapter is released as open-source under the Apache 2.0 License.
It contains only the adapter weights and does not include any Meta LLaMA 3B base model weights.

You must still comply with the [Meta Llama 3 license](https://ai.meta.com/llama/license/) if using the base model together with this adapter.
