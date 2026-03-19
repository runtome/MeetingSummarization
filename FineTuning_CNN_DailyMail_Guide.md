# Fine-Tuning on CNN-DailyMail: Step-by-Step Guide

> **Task:** News article → concise summary (abstractive summarization)
> **Model:** Qwen2.5-7B-Instruct (QLoRA, 4-bit)
> **Dataset:** CNN/DailyMail 3.0.0 — 287k train / 13k val / 11k test
> **Method:** QLoRA + SFTTrainer (same pipeline as the meeting summarization project)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Dataset Overview](#3-dataset-overview)
4. [Step 1 — Environment Setup](#step-1--environment-setup)
5. [Step 2 — Understand the Config](#step-2--understand-the-config)
6. [Step 3 — Prepare the Dataset](#step-3--prepare-the-dataset)
7. [Step 4 — Inspect the Data](#step-4--inspect-the-data)
8. [Step 5 — Fine-Tune the Model](#step-5--fine-tune-the-model)
9. [Step 6 — Monitor Training](#step-6--monitor-training)
10. [Step 7 — Evaluate with ROUGE](#step-7--evaluate-with-rouge)
11. [Step 8 — Run Inference](#step-8--run-inference)
12. [Step 9 — Merge & Export (Optional)](#step-9--merge--export-optional)
13. [GPU Memory Reference](#gpu-memory-reference)
14. [Hyperparameter Tuning](#hyperparameter-tuning)
15. [Troubleshooting](#troubleshooting)
16. [Expected Results](#expected-results)

---

## 1. Prerequisites

### Hardware
| Requirement | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB (A4000 / T4) | 24 GB (A10G / RTX 3090) |
| System RAM | 32 GB | 64 GB |
| Disk space | 30 GB free | 60 GB free |

> **No GPU?** Use Google Colab Pro (A100 40GB) or Kaggle (P100 16GB). The default config with `max_train_samples: 50000` fits in 16GB.

### Software
- Python 3.10+
- CUDA 11.8 or 12.1
- Git

---

## 2. Project Structure

```
MeetingSummarization/
├── configs/
│   ├── training_config.yaml          # Meeting summarization config (original)
│   └── cnndm_training_config.yaml    # CNN-DailyMail config (new)
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py        # Meeting dataset pipeline (original)
│   │   └── prepare_cnndm_dataset.py  # CNN-DailyMail dataset pipeline (new)
│   ├── train.py                      # Training script (shared, no changes needed)
│   ├── evaluate.py                   # ROUGE evaluation
│   └── inference.py                  # Inference / generation
├── data/
│   └── cnndm/                        # Created by prepare script
│       ├── train.jsonl
│       └── val.jsonl
├── outputs/
│   └── cnndm/                        # Checkpoints saved here
│       └── final_adapter/            # Final LoRA adapter
└── requirements.txt
```

---

## 3. Dataset Overview

| Field | Description | Avg Tokens |
|---|---|---|
| `article` | Full news article body | ~781 |
| `highlights` | Author-written bullet summary | ~56 |
| `id` | SHA1 hash of the source URL | — |

### Split sizes (version 3.0.0)
| Split | Samples |
|---|---|
| Train | 287,113 |
| Validation | 13,368 |
| Test | 11,490 |

### Data format after preparation (ChatML JSONL)
Each line in `train.jsonl` / `val.jsonl` looks like:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a news summarization assistant. Given a news article, write a concise and accurate summary capturing the most important facts."
    },
    {
      "role": "user",
      "content": "Summarize the following news article:\n\n(CNN) -- An American woman died aboard a cruise ship..."
    },
    {
      "role": "assistant",
      "content": "The elderly woman suffered from diabetes and hypertension, ship's doctors say.\nPreviously, 86 passengers had fallen ill on the ship."
    }
  ]
}
```

---

## Step 1 — Environment Setup

### 1.1 Clone and enter the project
```bash
git clone <your-repo-url>
cd MeetingSummarization
```

### 1.2 Create a virtual environment
```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 1.3 Install dependencies
```bash
# Install PyTorch with CUDA 12.1 (adjust cu version to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip install -r requirements.txt
```

### 1.4 Install flash-attention (strongly recommended for speed)
```bash
pip install flash-attn --no-build-isolation
```

> If flash-attn fails to compile, remove `attn_implementation="flash_attention_2"` from `src/train.py:61` and re-run without it. Training will be slower but will work.

### 1.5 Verify GPU access
```python
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected output (example):
```
CUDA available : True
GPU            : NVIDIA A10G
VRAM           : 22.5 GB
```

### 1.6 Log in to Hugging Face (required to download Qwen2.5)
```bash
huggingface-cli login
```
Paste your HuggingFace access token when prompted. You can get one at `huggingface.co/settings/tokens`.

---

## Step 2 — Understand the Config

Open `configs/cnndm_training_config.yaml` and review each section:

```yaml
# ── Model ──────────────────────────────────────────────────────────────────
model_name: "Qwen/Qwen2.5-7B-Instruct"
max_seq_length: 1024   # articles ~781 tokens + summary ~56 tokens → 1024 is safe

# ── QLoRA Quantization ────────────────────────────────────────────────────
quantization:
  load_in_4bit: true               # Load weights in 4-bit NF4 format
  bnb_4bit_quant_type: "nf4"       # NormalFloat4 — best quality for LLMs
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true  # Nested quantization saves ~0.4 bits/param

# ── LoRA Adapters ─────────────────────────────────────────────────────────
lora:
  r: 16              # Rank — controls capacity. 16 is sufficient for news summarization.
  lora_alpha: 32     # Scaling = alpha/r = 2. Keep at 2× rank.
  lora_dropout: 0.05
  target_modules:    # All projection layers in Qwen2.5 attention + FFN
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# ── Training Hyperparameters ──────────────────────────────────────────────
training:
  output_dir: "./outputs/cnndm"
  num_train_epochs: 1          # 50k samples × 1 epoch is usually enough
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2   # Effective batch = 8 × 2 = 16
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
  gradient_checkpointing: true     # Trades speed for memory
  optim: "paged_adamw_8bit"        # Memory-efficient optimizer from bitsandbytes
  max_grad_norm: 0.3
  group_by_length: true            # Packs similar-length sequences → faster

# ── Data ──────────────────────────────────────────────────────────────────
data:
  train_file: "./data/cnndm/train.jsonl"
  val_file:   "./data/cnndm/val.jsonl"
  max_train_samples: 50000  # Use null to train on all 287k (slower)
  max_val_samples:   2000   # Use null for full 13k validation
```

### Key decisions explained

| Setting | Value | Why |
|---|---|---|
| `max_seq_length` | 1024 | Articles average 781 tokens; 1024 covers 95%+ without waste |
| `lora.r` | 16 | News summarization is uniform; lower rank than meeting (32) is enough |
| `num_train_epochs` | 1 | 50k × 1 epoch = 50k steps; more can overfit |
| `max_train_samples` | 50000 | Good accuracy fast; set `null` for full training |

---

## Step 3 — Prepare the Dataset

This script downloads CNN-DailyMail from HuggingFace, converts it to ChatML format, and saves JSONL files.

```bash
python -m src.data.prepare_cnndm_dataset --config configs/cnndm_training_config.yaml
```

### What happens internally

```
HuggingFace Hub
  └─ cnn_dailymail 3.0.0 (train split, first 50k)
       │
       ▼
  for each row:
    article   → user message
    highlights → assistant message
       │
       ▼
  data/cnndm/train.jsonl   (50,000 rows)
  data/cnndm/val.jsonl     (2,000 rows)
```

### Expected output
```
Loading CNN-DailyMail 'train' split...
Loaded 50000 examples from 'train' split
Loading CNN-DailyMail 'validation' split...
Loaded 2000 examples from 'validation' split
Formatting: 100%|████████| 50000/50000 [00:12<00:00]
Formatting: 100%|████████|  2000/2000 [00:00<00:00]
Saved 50000 examples to ./data/cnndm/train.jsonl
Saved 2000 examples to ./data/cnndm/val.jsonl

Dataset ready: 50000 train, 2000 val
```

### To use the full dataset (optional)

Edit `configs/cnndm_training_config.yaml`:
```yaml
data:
  max_train_samples: null   # Downloads all 287,113 samples
  max_val_samples:   null   # Uses all 13,368 validation samples
```
> Full download takes ~10–15 minutes depending on connection speed.

---

## Step 4 — Inspect the Data

Before training, verify the JSONL files look correct:

```bash
# Show the first record (pretty-printed)
python -c "
import json
with open('data/cnndm/train.jsonl') as f:
    row = json.loads(f.readline())
for msg in row['messages']:
    print(f\"=== {msg['role'].upper()} ===\")
    print(msg['content'][:300])
    print()
"
```

Expected output:
```
=== SYSTEM ===
You are a news summarization assistant. Given a news article, write a concise and accurate summary capturing the most important facts.

=== USER ===
Summarize the following news article:

(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill...

=== ASSISTANT ===
The elderly woman suffered from diabetes and hypertension, ship's doctors say .
Previously, 86 passengers had fallen ill on the ship, Agencia Brasil says .
```

```bash
# Count total records
python -c "
train = sum(1 for _ in open('data/cnndm/train.jsonl'))
val   = sum(1 for _ in open('data/cnndm/val.jsonl'))
print(f'Train: {train:,}  |  Val: {val:,}')
"
```

---

## Step 5 — Fine-Tune the Model

Run the training script pointing to the CNN-DailyMail config:

```bash
python -m src.train --config configs/cnndm_training_config.yaml
```

### To resume from a checkpoint

```bash
python -m src.train \
  --config configs/cnndm_training_config.yaml \
  --resume-from-checkpoint ./outputs/cnndm/checkpoint-1000
```

### Training flow (inside `src/train.py`)

```
1. Load config  (cnndm_training_config.yaml)
2. Build BitsAndBytesConfig  (4-bit NF4)
3. Load Qwen2.5-7B-Instruct in 4-bit
4. Load tokenizer  (pad_token = eos_token)
5. Build LoraConfig  (r=16, all proj layers)
6. Load train.jsonl + val.jsonl → HuggingFace Dataset
7. Build TrainingArguments
8. SFTTrainer.train()
9. Save final adapter  → outputs/cnndm/final_adapter/
```

### Approximate training time

| GPU | Samples | Batch Size | Time / Epoch |
|---|---|---|---|
| RTX 3090 (24 GB) | 50k | 8×2 = 16 | ~2.5 hours |
| A10G (24 GB) | 50k | 8×2 = 16 | ~2 hours |
| A100 (40 GB) | 50k | 16×2 = 32 | ~1 hour |
| T4 (16 GB) | 50k | 4×4 = 16 | ~5 hours |

---

## Step 6 — Monitor Training

### Watch training logs in real time
```bash
# The trainer prints loss every `logging_steps: 50` steps
# Sample output:
{'loss': 1.823, 'learning_rate': 1.98e-04, 'epoch': 0.03}
{'loss': 1.612, 'learning_rate': 1.91e-04, 'epoch': 0.06}
{'loss': 1.401, 'learning_rate': 1.80e-04, 'epoch': 0.12}
...
{'eval_loss': 1.287, 'epoch': 0.50}
...
{'loss': 1.182, 'learning_rate': 5.20e-05, 'epoch': 0.90}
```

### Healthy training signals
- **Loss** should drop steadily from ~1.8 to ~1.1–1.3 over 1 epoch
- **Eval loss** should track train loss (gap < 0.2 is good)
- If eval loss rises while train loss falls → overfitting (reduce epochs or increase dropout)

### Enable Weights & Biases (optional)
```bash
pip install wandb
wandb login
```
Then change `report_to` in `src/train.py:106`:
```python
report_to="wandb",
```

### Check saved checkpoints
```bash
ls outputs/cnndm/
# checkpoint-500/
# checkpoint-1000/
# checkpoint-1500/
# final_adapter/
```

---

## Step 7 — Evaluate with ROUGE

After training, evaluate the adapter on the CNN-DailyMail test set:

```python
# evaluate_cnndm.py — run this after training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import evaluate
from tqdm import tqdm

ADAPTER_PATH = "./outputs/cnndm/final_adapter"
BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
NUM_TEST     = 500   # increase for more reliable scores

# ── Load model ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

# ── Load test data ───────────────────────────────────────────────────────────
ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
ds = ds.select(range(NUM_TEST))

SYSTEM = (
    "You are a news summarization assistant. Given a news article, "
    "write a concise and accurate summary capturing the most important facts."
)

def generate_summary(article: str) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": f"Summarize the following news article:\n\n{article}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ── Compute ROUGE ────────────────────────────────────────────────────────────
rouge = evaluate.load("rouge")
preds, refs = [], []

for row in tqdm(ds, desc="Evaluating"):
    preds.append(generate_summary(row["article"]))
    refs.append(row["highlights"])

scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
print("\n=== ROUGE Scores ===")
for k, v in scores.items():
    print(f"  {k}: {v * 100:.2f}")
```

Run it:
```bash
python evaluate_cnndm.py
```

---

## Step 8 — Run Inference

### Quick single-article test
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER_PATH = "./outputs/cnndm/final_adapter"
BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

article = """
LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported
£20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money
won't change his lifestyle. The young actor said he had no plans to abuse his riches.
"I don't plan to be one of those people who, as soon as they turn 18, suddenly buy a
massive sports car collection or something similar," he told an Australian interviewer.
"""

SYSTEM = (
    "You are a news summarization assistant. Given a news article, "
    "write a concise and accurate summary capturing the most important facts."
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"Summarize the following news article:\n\n{article.strip()}"},
]

text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = out[0][inputs["input_ids"].shape[1]:]
summary   = tokenizer.decode(generated, skip_special_tokens=True)
print("Summary:", summary)
```

### Expected output
```
Summary: Daniel Radcliffe turns 18 and gains access to his £20 million fortune,
but the Harry Potter star says the money won't change his lifestyle or spending habits.
```

---

## Step 9 — Merge & Export (Optional)

By default, the fine-tuned weights are stored as a lightweight LoRA adapter (~100MB). To merge the adapter back into the base model for deployment:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./outputs/cnndm/final_adapter"
MERGED_PATH  = "./outputs/cnndm/merged_model"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)

print("Merging adapter into base model...")
merged = model.merge_and_unload()
merged.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print(f"Merged model saved to {MERGED_PATH}")
```

> The merged model is ~15GB (bf16). Use this for production inference or HuggingFace Hub upload.

### Push to HuggingFace Hub
```bash
huggingface-cli upload your-username/qwen2.5-7b-cnndm ./outputs/cnndm/merged_model
```

---

## GPU Memory Reference

Memory usage with the default config (`per_device_train_batch_size: 8`):

| Component | VRAM |
|---|---|
| Model weights (4-bit) | ~4 GB |
| Gradients + optimizer (paged) | ~3 GB |
| Activations (batch=8, seq=1024) | ~6 GB |
| LoRA adapters | ~0.3 GB |
| **Total** | **~13–14 GB** |

### Reduce memory if you run out

Edit `configs/cnndm_training_config.yaml`:

```yaml
training:
  per_device_train_batch_size: 4    # was 8  → saves ~3 GB
  gradient_accumulation_steps: 4   # was 2  → keeps effective batch = 16
```

Or reduce LoRA rank:
```yaml
lora:
  r: 8          # was 16 → saves ~0.2 GB but slightly lower quality
  lora_alpha: 16
```

---

## Hyperparameter Tuning

### Learning rate
| LR | Effect |
|---|---|
| `5e-5` | Underfits — too conservative |
| `2e-4` | Default — good balance |
| `5e-4` | May overfit or diverge |

Start with `2e-4`. If eval loss spikes early, lower to `1e-4`.

### LoRA rank `r`
| r | Trainable params | Use when |
|---|---|---|
| 8 | ~10M | Low memory, simple task |
| 16 | ~20M | Default — news summarization |
| 32 | ~40M | Complex / multilingual tasks |
| 64 | ~80M | Maximum quality (needs 24GB+) |

### Number of training samples
| Samples | Quality | Time (A10G) |
|---|---|---|
| 10,000 | Baseline | ~25 min |
| 50,000 | Good (default) | ~2 hours |
| 100,000 | Better | ~4 hours |
| 287,113 (full) | Best | ~11 hours |

---

## Troubleshooting

### `CUDA out of memory`
```bash
# Reduce batch size and increase gradient accumulation steps to compensate
# In cnndm_training_config.yaml:
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # effective batch still = 16
```

### `ValueError: Flash Attention is not installed`
```bash
pip install flash-attn --no-build-isolation
# OR remove flash_attention_2 from src/train.py line 61
```

### `OSError: model not found` for Qwen2.5
```bash
huggingface-cli login   # ensure you are logged in
# Verify model exists:
python -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen2.5-7B-Instruct'))"
```

### Loss is NaN from step 1
- Usually a bad learning rate. Lower to `5e-5`.
- Check `bf16: true` — your GPU must support bfloat16 (Ampere+). For older GPUs, switch to `fp16: true` and remove `bf16`.

### Training loss decreases but ROUGE does not improve
- The model may be memorizing format rather than content.
- Try lowering `temperature` during inference to `0.1`.
- Increase `max_new_tokens` to `256` to allow longer outputs.

### Slow data loading
```bash
# Pre-download the dataset to disk
python -c "from datasets import load_dataset; load_dataset('cnn_dailymail', '3.0.0')"
```

---

## Expected Results

### ROUGE scores (test set, 500 samples)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| `facebook/bart-large-cnn` (baseline) | 44.16 | 21.28 | 40.90 |
| Qwen2.5-7B zero-shot | ~38–42 | ~17–20 | ~35–39 |
| **Qwen2.5-7B QLoRA (50k, 1 epoch)** | **~42–45** | ~20–22 | ~39–42 |
| Qwen2.5-7B QLoRA (full 287k, 1 epoch) | ~44–47 | ~21–23 | ~41–44 |

> ROUGE scores are approximate. Results vary by generation hyperparameters (`temperature`, `top_p`, `max_new_tokens`).

### What good outputs look like

**Article excerpt:**
> Scientists have confirmed that daily coffee consumption is linked to a reduced risk of type 2 diabetes, according to a study of 500,000 participants published in the New England Journal of Medicine...

**Reference highlight:**
> A study of 500,000 people links daily coffee consumption to lower type 2 diabetes risk.

**Model output (good):**
> Research published in the New England Journal of Medicine finds that drinking coffee daily is associated with reduced type 2 diabetes risk, based on data from half a million participants.

**Model output (bad — sign of overfitting or too high temperature):**
> Coffee is good for health and people should drink it every day to stay healthy and reduce risk.

---

*For questions or issues, open a GitHub issue or check the [HuggingFace TRL docs](https://huggingface.co/docs/trl).*
