# Fine-Tuning on CNN-DailyMail — Step-by-Step Guide

> **Model:** `Qwen/Qwen2.5-7B-Instruct`
> **Method:** QLoRA (4-bit quantization + LoRA adapters)
> **Trainer:** HuggingFace `SFTTrainer` (TRL)
> **Dataset:** [CNN / DailyMail 3.0.0](https://huggingface.co/datasets/cnn_dailymail) — ~287k articles + highlights

---

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Configuration](#3-configuration)
4. [Quantization (QLoRA)](#4-quantization-qlora)
5. [LoRA Adapter Setup](#5-lora-adapter-setup)
6. [Training Arguments](#6-training-arguments)
7. [Running Training](#7-running-training)
8. [Monitoring](#8-monitoring)
9. [Inference / Evaluation](#9-inference--evaluation)
10. [Common Errors & Fixes](#10-common-errors--fixes)

---

## 1. Environment Setup

### 1.1 Install dependencies

```bash
pip install transformers>=4.45.0 \
            trl>=0.12.0 \
            peft>=0.13.0 \
            bitsandbytes>=0.43.0 \
            datasets \
            accelerate \
            flash-attn \
            rouge-score \
            pyyaml
```

> **Kaggle / Colab tip:** Enable GPU (T4 / A100) before running.
> Flash-Attention 2 requires an Ampere GPU (A100, A10G, RTX 30xx+).

### 1.2 Set HuggingFace token (avoids rate limits)

```bash
export HF_TOKEN="hf_your_token_here"
# or inside a notebook:
import os; os.environ["HF_TOKEN"] = "hf_your_token_here"
```

---

## 2. Dataset Preparation

CNN-DailyMail articles are converted into **chat-style JSONL** so `SFTTrainer` can apply the model's chat template automatically.

### 2.1 Format of each record

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Summarize the following news article in 2-3 sentences.\n\n<article text>"
    },
    {
      "role": "assistant",
      "content": "<highlight sentences>"
    }
  ]
}
```

### 2.2 Run the preparation script

```bash
python -m src.data.prepare_cnndm_dataset \
    --config configs/cnndm_training_config.yaml
```

This script will:
- Download `cnn_dailymail` v3.0.0 from HuggingFace Hub
- Sample `max_train_samples` (default 50 000) and `max_val_samples` (default 2 000)
- Write `data/cnndm/train.jsonl` and `data/cnndm/val.jsonl`

### 2.3 Dataset statistics

| Split | Full size | Used (default) |
|-------|-----------|----------------|
| Train | 287,113 | 50,000 |
| Validation | 13,368 | 2,000 |
| Test | 11,490 | — |

- Average article length: **~781 tokens**
- Average highlight length: **~56 tokens**
- `max_seq_length` is set to **1024** to comfortably fit both

---

## 3. Configuration

All hyperparameters live in `configs/cnndm_training_config.yaml`:

```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
max_seq_length: 1024

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  output_dir: "./outputs/cnndm"
  num_train_epochs: 1
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2   # effective batch = 16
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  logging_steps: 50
  save_strategy: "steps"
  save_steps: 500
  eval_strategy: "steps"
  eval_steps: 500
  bf16: true
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
  max_grad_norm: 0.3

data:
  train_file: "./data/cnndm/train.jsonl"
  val_file:   "./data/cnndm/val.jsonl"
  max_train_samples: 50000
  max_val_samples: 2000
```

---

## 4. Quantization (QLoRA)

4-bit NF4 quantization reduces the 7B model from ~14 GB (bf16) to **~4 GB VRAM**, making it trainable on a single T4/A100.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # saves ~0.4 bits/param extra
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,               # NOTE: use `dtype`, not `torch_dtype`
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False          # required for gradient checkpointing
```

> ⚠️ **`torch_dtype` is deprecated** in recent `transformers`. Always use `dtype=`.

---

## 5. LoRA Adapter Setup

LoRA injects small trainable rank-decomposition matrices into the attention and MLP layers.
Only **~1-2% of parameters** are trained — the base model weights stay frozen.

```python
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    r=16,                      # rank — higher = more capacity, more VRAM
    lora_alpha=32,             # scaling factor (usually 2×r)
    lora_dropout=0.05,
    target_modules=[           # all linear projections in Qwen2.5
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `r` | 16 | Rank; try 8 to save VRAM, 32 for more quality |
| `lora_alpha` | 32 | Keep at `2 × r` |
| `lora_dropout` | 0.05 | Light regularisation |

---

## 6. Training Arguments

```python
from trl import SFTConfig   # ← use SFTConfig, NOT TrainingArguments (TRL ≥ 0.12)

# warmup_ratio is deprecated → convert to warmup_steps at runtime
effective_batch = per_device_train_batch_size * gradient_accumulation_steps  # 16
total_steps     = (len(train_dataset) // effective_batch) * num_epochs       # ~3125
warmup_steps    = max(1, int(total_steps * 0.03))                            # ~94

training_args = SFTConfig(
    output_dir="./outputs/cnndm",
    # ── max_seq_length lives here now (removed from SFTTrainer.__init__) ──
    max_seq_length=1024,
    # ── schedule ──────────────────────────────────────────────────────────
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=warmup_steps,          # replaces deprecated warmup_ratio
    lr_scheduler_type="cosine",
    # ── logging / saving / eval ───────────────────────────────────────────
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    # ── precision / memory ────────────────────────────────────────────────
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    # NOTE: `group_by_length` was REMOVED in transformers ≥ 4.45 — do not use
)
```

> **Key changes vs older TRL:**
> - `SFTConfig` replaces `TrainingArguments` — it inherits everything + adds `max_seq_length`
> - `max_seq_length` is set in `SFTConfig`, **not** in `SFTTrainer()`
> - `warmup_ratio` → compute `warmup_steps` manually to silence the deprecation warning

---

## 7. Running Training

### 7.1 Single-GPU (Kaggle / Colab)

```bash
python -m src.train --config configs/cnndm_training_config.yaml
```

### 7.2 Resume from a checkpoint

```bash
python -m src.train \
    --config configs/cnndm_training_config.yaml \
    --resume-from-checkpoint ./outputs/cnndm/checkpoint-1000
```

### 7.3 Expected timeline (Kaggle T4, 50k samples)

| Batch size (eff.) | Steps / epoch | Time / epoch |
|:-----------------:|:-------------:|:------------:|
| 16 | ~3,125 | ~5–6 h |

---

## 8. Monitoring

Training logs are printed every `logging_steps=50` steps.
A typical log line looks like:

```
{'loss': 1.423, 'learning_rate': 1.87e-04, 'epoch': 0.16, 'step': 500}
```

### 8.1 Enable Weights & Biases (optional)

```bash
pip install wandb
wandb login
```

Change `report_to="none"` → `report_to="wandb"` in `TrainingArguments`.

---

## 9. Inference / Evaluation

### 9.1 Load the saved adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./outputs/cnndm/final_adapter")
tokenizer = AutoTokenizer.from_pretrained("./outputs/cnndm/final_adapter")
```

### 9.2 Generate a summary

```python
article = """(CNN) -- An American woman died aboard a cruise ship ..."""

messages = [
    {"role": "user", "content": f"Summarize the following news article in 2-3 sentences.\n\n{article}"}
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

output = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
)
print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

### 9.3 Evaluate with ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print(scores)
# e.g. {'rouge1': Score(precision=..., recall=..., fmeasure=0.44), ...}
```

**Baseline scores on CNN/DM (extractive oracle):**

| Metric | Score |
|--------|-------|
| ROUGE-1 | ~44.4 |
| ROUGE-2 | ~21.2 |
| ROUGE-L | ~40.9 |

---

## 10. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: unexpected keyword argument 'group_by_length'` | Removed in `transformers ≥ 4.45` | Delete `group_by_length` from `TrainingArguments` and YAML |
| `TypeError: unexpected keyword argument 'max_seq_length'` (in `SFTTrainer`) | Moved to `SFTConfig` in TRL 0.12, then fully removed in TRL ≥ 0.15 | Remove from both `SFTTrainer` and `SFTConfig`; set `tokenizer.model_max_length = 1024` instead |
| `warmup_ratio is deprecated` | Removed in TRL v5.2 | Compute `warmup_steps = total_steps × ratio` and use `warmup_steps=` instead |
| `UserWarning: torch_dtype is deprecated` | Renamed parameter | Use `dtype=torch.bfloat16` instead of `torch_dtype=` |
| `CUDA out of memory` | Batch size too large | Reduce `per_device_train_batch_size` to 4 or 2; increase `gradient_accumulation_steps` |
| `Flash Attention not available` | GPU is not Ampere | Remove `attn_implementation="flash_attention_2"` (falls back to SDPA) |
| `HF_TOKEN unauthenticated` | No token set | `export HF_TOKEN=hf_...` before running |
| Loss is NaN | LR too high or bad data | Lower `learning_rate` to `1e-4`; check JSONL format |

---

## Project File Structure

```
MeetingSummarization/
├── configs/
│   └── cnndm_training_config.yaml   # All hyperparameters
├── data/
│   └── cnndm/
│       ├── train.jsonl              # Generated by prepare script
│       └── val.jsonl
├── src/
│   ├── data/
│   │   └── prepare_cnndm_dataset.py
│   └── train.py                     # Main training entry point
└── outputs/
    └── cnndm/
        ├── checkpoint-500/
        ├── checkpoint-1000/
        └── final_adapter/           # Saved LoRA weights + tokenizer
```
