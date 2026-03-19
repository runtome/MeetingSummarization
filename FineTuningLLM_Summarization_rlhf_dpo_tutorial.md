# Complete Hands-On Tutorial: Fine-Tuning LLMs for Summarization with RLHF & DPO

> **Author:** Claude AI Tutorial Series
> **Level:** Intermediate to Advanced
> **Prerequisites:** Python, PyTorch basics, familiarity with Transformers

---

## Table of Contents

1. [Part 1: Fine-Tuning a Summarization Model](#part-1)
2. [Part 2: RLHF — Reinforcement Learning from Human Feedback](#part-2)
3. [Part 3: DPO — Direct Preference Optimization](#part-3)
4. [Appendix: Tips, Troubleshooting & Resources](#appendix)

---

## Environment Setup (All Parts)

```bash
# Create a virtual environment
python -m venv llm-finetune
source llm-finetune/bin/activate  # Linux/Mac
# llm-finetune\Scripts\activate   # Windows

# Install all required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate accelerate
pip install peft bitsandbytes   # For LoRA & Quantization
pip install trl                 # For RLHF & DPO
pip install rouge-score nltk sentencepiece
pip install wandb               # Optional: experiment tracking
```

### Verify GPU Access
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")
```

---

# Part 1: Fine-Tuning a Summarization Model

## 1.1 Understanding the Task

Abstractive summarization generates new text that captures the key information from the source. We'll fine-tune **BART** and **Flan-T5** on the CNN/DailyMail dataset.

### Model Selection Guide

| Model | Size | Best For | GPU Requirement |
|-------|------|----------|-----------------|
| `t5-small` | 60M | Learning & experiments | 4GB+ |
| `google/flan-t5-base` | 250M | Good balance | 8GB+ |
| `facebook/bart-large-cnn` | 400M | News summarization | 12GB+ |
| `google/flan-t5-large` | 780M | High quality | 16GB+ |
| `google/pegasus-xsum` | 568M | Extreme summarization | 16GB+ |

## 1.2 Loading and Exploring the Dataset

```python
from datasets import load_dataset

# Load CNN/DailyMail
dataset = load_dataset("cnn_dailymail", "3.0.0")

print(f"Train: {len(dataset['train'])} examples")
print(f"Validation: {len(dataset['validation'])} examples")
print(f"Test: {len(dataset['test'])} examples")

# Explore a sample
sample = dataset["train"][0]
print(f"\n--- Article (first 500 chars) ---")
print(sample["article"][:500])
print(f"\n--- Summary ---")
print(sample["highlights"])
print(f"\nArticle length: {len(sample['article'].split())} words")
print(f"Summary length: {len(sample['highlights'].split())} words")
```

### Using a Subset for Faster Experimentation
```python
# Use a small subset first to debug your pipeline
small_train = dataset["train"].select(range(5000))
small_val = dataset["validation"].select(range(500))
small_test = dataset["test"].select(range(500))
```

## 1.3 Tokenization

```python
from transformers import AutoTokenizer

# === Option A: BART ===
model_name = "facebook/bart-large-cnn"

# === Option B: Flan-T5 (recommended for beginners) ===
# model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define max lengths
MAX_INPUT_LENGTH = 1024   # Source article
MAX_TARGET_LENGTH = 128   # Summary

def preprocess_function(examples):
    """Tokenize articles and summaries."""
    # For T5 models, add a task prefix
    prefix = "summarize: " if "t5" in model_name.lower() else ""

    inputs = [prefix + doc for doc in examples["article"]]
    targets = examples["highlights"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token id with -100 so it's ignored in loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization
tokenized_train = small_train.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized_val = small_val.map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)
tokenized_test = small_test.map(preprocess_function, batched=True, remove_columns=dataset["test"].column_names)

print(f"Tokenized train sample keys: {tokenized_train[0].keys()}")
```

## 1.4 Model Loading

### Standard Loading
```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### LoRA Loading (Recommended for Limited GPU)
```python
from peft import LoraConfig, get_peft_model, TaskType

# Load base model in 8-bit for memory efficiency
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    load_in_8bit=True,          # 8-bit quantization
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,                        # Rank of update matrices
    lora_alpha=32,               # Scaling factor
    lora_dropout=0.1,            # Dropout for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~1.2M || all params: ~400M || trainable%: 0.30%
```

## 1.5 Training Configuration

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

# Load ROUGE metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """Compute ROUGE scores for evaluation."""
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (ignored tokens) with pad token
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Extract mid-F1 scores and scale to percentage
    result = {key: value * 100 for key, value in result.items()}

    # Add average generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results-summarization",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=4,        # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,         # Effective batch size = 4 * 4 = 16
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    predict_with_generate=True,            # Required for seq2seq evaluation
    generation_max_length=MAX_TARGET_LENGTH,
    fp16=True,                             # Mixed precision training
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",
    greater_is_better=True,
    save_total_limit=3,
    report_to="wandb",                     # Or "none" to disable
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)
```

## 1.6 Training

```python
# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train!
print("Starting training...")
train_result = trainer.train()

# Print results
print(f"\nTraining completed!")
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Training time: {train_result.metrics['train_runtime']:.0f}s")

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_test)
print(f"Test ROUGE-1: {test_results['eval_rouge1']:.2f}")
print(f"Test ROUGE-2: {test_results['eval_rouge2']:.2f}")
print(f"Test ROUGE-L: {test_results['eval_rougeL']:.2f}")
```

## 1.7 Inference — Generate Summaries

```python
from transformers import pipeline

# Load fine-tuned model
summarizer = pipeline(
    "summarization",
    model="./results-summarization/checkpoint-best",
    tokenizer=tokenizer,
    device=0,
)

# Test on a new article
article = """
Scientists have discovered a new species of deep-sea fish that can produce its own light
through a unique bioluminescent mechanism. The fish, found at depths of over 3,000 meters
in the Pacific Ocean, uses a combination of specialized cells and symbiotic bacteria to
create a blue-green glow. Researchers believe this adaptation helps the fish attract prey
and communicate with potential mates in the pitch-dark environment. The discovery was
published in the journal Nature Marine Biology and has implications for understanding
how organisms adapt to extreme environments.
"""

summary = summarizer(
    article,
    max_length=60,
    min_length=20,
    do_sample=False,       # Greedy decoding
)
print("Generated Summary:", summary[0]["summary_text"])

# Try different decoding strategies
summary_beam = summarizer(article, max_length=60, num_beams=4, no_repeat_ngram_size=3)
summary_sample = summarizer(article, max_length=60, do_sample=True, temperature=0.7, top_p=0.9)

print("\nBeam Search:", summary_beam[0]["summary_text"])
print("Sampling:", summary_sample[0]["summary_text"])
```

## 1.8 Save and Share Your Model

```python
# Save locally
model.save_pretrained("./my-summarization-model")
tokenizer.save_pretrained("./my-summarization-model")

# Push to Hugging Face Hub
# First: huggingface-cli login
model.push_to_hub("your-username/my-summarization-model")
tokenizer.push_to_hub("your-username/my-summarization-model")
```

---

# Part 2: RLHF — Reinforcement Learning from Human Feedback

## 2.1 What is RLHF?

RLHF improves model outputs by training with human preferences. The pipeline has 3 stages:

```
Stage 1: Supervised Fine-Tuning (SFT)
    └── Train on high-quality demonstrations (Part 1 above)

Stage 2: Reward Model Training
    └── Train a model to predict human preferences between pairs of outputs

Stage 3: RL Optimization (PPO)
    └── Use the reward model to guide the policy model via Proximal Policy Optimization
```

## 2.2 Stage 1: SFT Model (Already Done in Part 1)

Your fine-tuned summarization model from Part 1 serves as the SFT model. This is the starting point for RLHF.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sft_model_name = "./my-summarization-model"  # From Part 1
tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(sft_model_name)
```

## 2.3 Stage 2: Building a Reward Model

The reward model learns to score outputs based on human preferences.

### Creating a Preference Dataset

In practice, you'd collect human annotations. Here we'll use an existing preference dataset or create a synthetic one:

```python
from datasets import Dataset
import random

def create_preference_dataset(base_dataset, model, tokenizer, num_samples=1000):
    """
    Generate pairs of summaries and simulate preferences.
    In production, you'd use REAL human annotations.
    """
    records = []

    for i in range(min(num_samples, len(base_dataset))):
        article = base_dataset[i]["article"]
        reference = base_dataset[i]["highlights"]

        # Generate two different summaries with different strategies
        inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True).to(model.device)

        # Summary A: Beam search (typically more conservative)
        output_a = model.generate(**inputs, max_length=128, num_beams=4, no_repeat_ngram_size=3)
        summary_a = tokenizer.decode(output_a[0], skip_special_tokens=True)

        # Summary B: Sampling (typically more diverse but potentially less accurate)
        output_b = model.generate(**inputs, max_length=128, do_sample=True, temperature=1.2, top_p=0.9)
        summary_b = tokenizer.decode(output_b[0], skip_special_tokens=True)

        records.append({
            "prompt": article[:512],      # Truncated article as prompt
            "chosen": summary_a,          # Preferred summary
            "rejected": summary_b,        # Less preferred summary
        })

    return Dataset.from_list(records)

# For real projects, use existing preference datasets:
# dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
```

### Training the Reward Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer

# Load a pre-trained model as the reward model base
reward_model_name = "distilbert-base-uncased"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name,
    num_labels=1,  # Single scalar reward
)

# Tokenize the preference dataset
def preprocess_reward_data(examples):
    """Format: combine prompt with each response for comparison."""
    chosen = [p + " [SEP] " + c for p, c in zip(examples["prompt"], examples["chosen"])]
    rejected = [p + " [SEP] " + r for p, r in zip(examples["prompt"], examples["rejected"])]

    chosen_tokens = reward_tokenizer(chosen, truncation=True, max_length=512, padding="max_length")
    rejected_tokens = reward_tokenizer(rejected, truncation=True, max_length=512, padding="max_length")

    return {
        "input_ids_chosen": chosen_tokens["input_ids"],
        "attention_mask_chosen": chosen_tokens["attention_mask"],
        "input_ids_rejected": rejected_tokens["input_ids"],
        "attention_mask_rejected": rejected_tokens["attention_mask"],
    }

# Training with TRL's RewardTrainer
reward_training_args = TrainingArguments(
    output_dir="./reward-model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    fp16=True,
    remove_unused_columns=False,
)

reward_trainer = RewardTrainer(
    model=reward_model,
    args=reward_training_args,
    train_dataset=preference_train,
    eval_dataset=preference_val,
    tokenizer=reward_tokenizer,
)

reward_trainer.train()
reward_model.save_pretrained("./reward-model-final")
```

## 2.4 Stage 3: PPO Training with TRL

```python
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer
import torch

# Load the SFT model with a value head for PPO
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(sft_model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(sft_model_name)  # Frozen reference
tokenizer = AutoTokenizer.from_pretrained(sft_model_name)

# PPO Configuration
ppo_config = PPOConfig(
    model_name="summarization-rlhf",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,                  # PPO epochs per batch
    kl_penalty="kl",              # KL divergence penalty type
    init_kl_coef=0.2,             # Initial KL penalty coefficient
    target_kl=6.0,                # Target KL divergence
    max_grad_norm=0.5,
    log_with="wandb",             # Or None
)

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Load reward model for scoring
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward-model-final")
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

def get_reward(prompt_text, summary_text):
    """Score a summary using the reward model."""
    combined = prompt_text + " [SEP] " + summary_text
    inputs = reward_tokenizer(combined, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        reward = reward_model(**inputs).logits.squeeze()
    return reward

# PPO Training Loop
from tqdm import tqdm

num_training_steps = 1000

for step in tqdm(range(num_training_steps)):
    # 1. Sample a batch of prompts
    batch = next(iter(dataloader))  # Your article dataloader
    prompt_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze() for p in batch["article"]]

    # 2. Generate responses from the current policy
    response_tensors = []
    for prompt in prompt_tensors:
        response = ppo_trainer.generate(
            prompt.unsqueeze(0),
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response_tensors.append(response.squeeze())

    # 3. Compute rewards
    rewards = []
    for prompt, response in zip(batch["article"], response_tensors):
        summary_text = tokenizer.decode(response, skip_special_tokens=True)
        reward = get_reward(prompt[:512], summary_text)
        rewards.append(reward)

    # 4. PPO update step
    stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)

    # 5. Log metrics
    if step % 10 == 0:
        print(f"Step {step}: mean_reward={torch.stack(rewards).mean():.3f}, "
              f"kl={stats['ppo/mean_kl']:.3f}")

# Save the RLHF-tuned model
model.save_pretrained("./summarization-rlhf-final")
tokenizer.save_pretrained("./summarization-rlhf-final")
```

---

# Part 3: DPO — Direct Preference Optimization

## 3.1 Why DPO Over RLHF?

DPO is a simpler and more stable alternative to RLHF-PPO:

| Aspect | RLHF (PPO) | DPO |
|--------|------------|-----|
| Reward Model | Required (separate model) | Not needed |
| Training Stability | Can be unstable | More stable |
| Compute Cost | High (4 models in memory) | Lower (2 models) |
| Hyperparameters | Many to tune | Fewer (mainly β) |
| Performance | Excellent | Comparable or better |
| Implementation | Complex | Simple |

DPO directly optimizes the policy using preference pairs — no reward model needed!

## 3.2 Preparing the Preference Dataset for DPO

```python
from datasets import load_dataset, Dataset

# Option 1: Use an existing preference dataset
# dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

# Option 2: Create your own preference dataset
# Format: {"prompt": str, "chosen": str, "rejected": str}

def create_dpo_dataset():
    """
    Create a DPO-compatible dataset.
    Each example has a prompt, a preferred (chosen) response,
    and a less preferred (rejected) response.
    """
    examples = [
        {
            "prompt": "Summarize: Scientists discovered that regular exercise can reduce "
                      "the risk of heart disease by up to 50%. The study, conducted over "
                      "10 years with 50,000 participants, found that just 30 minutes of "
                      "moderate activity daily significantly improved cardiovascular health.",
            "chosen": "A decade-long study of 50,000 people found that 30 minutes of "
                      "daily moderate exercise can cut heart disease risk by up to 50%.",
            "rejected": "Scientists did a study about exercise and found it is good for "
                        "your heart. You should exercise more.",
        },
        # ... more examples (aim for 1000-10000+ pairs)
    ]
    return Dataset.from_list(examples)

# For real projects, use larger preference datasets:
# - "Anthropic/hh-rlhf"
# - "openai/summarize_from_feedback"
# - Your own human-annotated data
```

## 3.3 DPO Training with TRL

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, TaskType

# 1. Load the SFT model (from Part 1)
sft_model_name = "./my-summarization-model"
tokenizer = AutoTokenizer.from_pretrained(sft_model_name)

# 2. Load model (optionally with LoRA for memory efficiency)
model = AutoModelForSeq2SeqLM.from_pretrained(sft_model_name)

# Optional: Apply LoRA for parameter-efficient DPO
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# 3. Load the reference model (frozen copy of SFT model)
ref_model = AutoModelForSeq2SeqLM.from_pretrained(sft_model_name)

# 4. Load preference dataset
preference_dataset = load_dataset("your-preference-dataset")
# Or use the synthetic one created above

# 5. DPO Configuration
dpo_config = DPOConfig(
    output_dir="./dpo-summarization",

    # Core DPO hyperparameters
    beta=0.1,                          # Temperature parameter (lower = stronger preference)
    loss_type="sigmoid",               # "sigmoid" (standard) or "hinge" or "ipo"

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,                # DPO typically uses lower LR than SFT
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,

    # Tokenization
    max_length=1024,                   # Max total length (prompt + response)
    max_prompt_length=512,             # Max prompt length
    max_target_length=128,             # Max response length

    # Evaluation & saving
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=25,
    save_total_limit=3,
    load_best_model_at_end=True,

    # Logging
    report_to="wandb",                 # Or "none"
    run_name="dpo-summarization",
)

# 6. Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset["train"],
    eval_dataset=preference_dataset["validation"],
    tokenizer=tokenizer,
    peft_config=peft_config,           # Optional: pass LoRA config
)

# 7. Train!
print("Starting DPO training...")
dpo_trainer.train()

# 8. Save the model
dpo_trainer.save_model("./dpo-summarization-final")
tokenizer.save_pretrained("./dpo-summarization-final")
```

## 3.4 Understanding the β (Beta) Hyperparameter

The `beta` parameter controls how much the model can deviate from the reference:

```python
# beta = 0.1 (default) — Moderate constraint
# beta = 0.01 — Loose constraint: model deviates more from reference
# beta = 0.5 — Tight constraint: model stays close to reference

# Experiment with different beta values
for beta in [0.05, 0.1, 0.2, 0.5]:
    config = DPOConfig(beta=beta, output_dir=f"./dpo-beta-{beta}", ...)
    trainer = DPOTrainer(model=model, ref_model=ref_model, args=config, ...)
    trainer.train()
    # Compare results across beta values
```

## 3.5 Advanced: DPO Loss Variants

TRL supports multiple DPO loss types:

```python
# Standard DPO (default)
DPOConfig(loss_type="sigmoid", beta=0.1)

# IPO (Identity Preference Optimization) — more robust to noisy preferences
DPOConfig(loss_type="ipo", beta=0.1)

# Hinge loss variant
DPOConfig(loss_type="hinge", beta=0.1)

# KTO (Kahneman-Tversky Optimization) — works with binary feedback (good/bad)
# Instead of pairs, just needs "is this output good or bad?"
from trl import KTOConfig, KTOTrainer
kto_config = KTOConfig(
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)
```

## 3.6 DPO Inference and Evaluation

```python
from transformers import pipeline
import evaluate

# Load the DPO-trained model
dpo_summarizer = pipeline(
    "summarization",
    model="./dpo-summarization-final",
    tokenizer=tokenizer,
    device=0,
)

# Compare: SFT vs DPO
sft_summarizer = pipeline(
    "summarization",
    model="./my-summarization-model",
    tokenizer=tokenizer,
    device=0,
)

test_article = """
The global semiconductor shortage that began in 2020 continues to impact multiple
industries. Automakers have been forced to cut production by millions of vehicles,
while consumer electronics companies face delays in launching new products.
Industry analysts predict the shortage could persist until late 2025, as new
fabrication plants take years to build and become operational. Governments worldwide
are investing billions in domestic chip production to reduce reliance on Asian
manufacturers.
"""

sft_summary = sft_summarizer(test_article, max_length=80)[0]["summary_text"]
dpo_summary = dpo_summarizer(test_article, max_length=80)[0]["summary_text"]

print("=== SFT Summary ===")
print(sft_summary)
print("\n=== DPO Summary ===")
print(dpo_summary)

# Quantitative evaluation with ROUGE
rouge = evaluate.load("rouge")

def evaluate_model(summarizer, test_data, num_samples=100):
    predictions = []
    references = []
    for i in range(num_samples):
        pred = summarizer(test_data[i]["article"], max_length=128)[0]["summary_text"]
        predictions.append(pred)
        references.append(test_data[i]["highlights"])

    results = rouge.compute(predictions=predictions, references=references)
    return {k: round(v * 100, 2) for k, v in results.items()}

sft_scores = evaluate_model(sft_summarizer, dataset["test"])
dpo_scores = evaluate_model(dpo_summarizer, dataset["test"])

print("\n=== ROUGE Comparison ===")
for metric in ["rouge1", "rouge2", "rougeL"]:
    print(f"{metric}: SFT={sft_scores[metric]:.2f} | DPO={dpo_scores[metric]:.2f}")
```

---

# Part 4: Complete Pipeline — Putting It All Together

## End-to-End Workflow

```python
"""
Complete pipeline: SFT → DPO for Summarization
This is the recommended modern approach (simpler than RLHF-PPO).
"""

# ============================================
# STEP 1: Supervised Fine-Tuning (SFT)
# ============================================
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
base_model = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, r=16,
    lora_alpha=32, lora_dropout=0.1,
    target_modules=["q", "v"],
)
model = get_peft_model(model, lora_config)

# Load and tokenize dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
# ... (tokenize as shown in Part 1)

# Train SFT
sft_args = Seq2SeqTrainingArguments(
    output_dir="./pipeline-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    predict_with_generate=True,
    fp16=True,
)
sft_trainer = Seq2SeqTrainer(model=model, args=sft_args, ...)
sft_trainer.train()
model.save_pretrained("./pipeline-sft-final")

# ============================================
# STEP 2: Collect/Load Preferences
# ============================================
preference_data = load_dataset("your-preference-dataset")
# Or generate synthetic preferences from SFT model outputs

# ============================================
# STEP 3: DPO Training
# ============================================
from trl import DPOConfig, DPOTrainer

dpo_model = AutoModelForSeq2SeqLM.from_pretrained("./pipeline-sft-final")
ref_model = AutoModelForSeq2SeqLM.from_pretrained("./pipeline-sft-final")

dpo_config = DPOConfig(
    output_dir="./pipeline-dpo",
    beta=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    fp16=True,
)

dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_data["train"],
    eval_dataset=preference_data["validation"],
    tokenizer=tokenizer,
)
dpo_trainer.train()
dpo_trainer.save_model("./pipeline-final")

print("Pipeline complete! Final model saved to ./pipeline-final")
```

---

# Appendix: Tips, Troubleshooting & Resources

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce batch size, use gradient accumulation, enable LoRA/QLoRA, use 8-bit/4-bit quantization |
| Poor ROUGE scores | Check tokenization (ensure labels have -100 padding), increase training data, try different learning rates |
| Model generates repetitive text | Add `no_repeat_ngram_size=3`, use `repetition_penalty=1.2` |
| DPO loss not decreasing | Lower beta, check preference dataset quality, ensure chosen/rejected are meaningfully different |
| PPO training unstable | Lower learning rate, increase KL penalty, ensure reward model is well-calibrated |

## Hyperparameter Cheat Sheet

### SFT (Supervised Fine-Tuning)
- **Learning Rate:** 1e-5 to 5e-5
- **Batch Size:** 16-32 (effective, with gradient accumulation)
- **Epochs:** 3-5
- **Warmup:** 500-1000 steps

### DPO
- **Beta:** 0.05 to 0.5 (start with 0.1)
- **Learning Rate:** 1e-7 to 5e-6 (much lower than SFT)
- **Epochs:** 1-3
- **Max Length:** Match your SFT training lengths

### RLHF (PPO)
- **KL Coefficient:** 0.1 to 0.5
- **Learning Rate:** 1e-6 to 5e-5
- **PPO Epochs:** 2-4 per batch
- **Mini-batch Size:** 4-8

## Recommended Reading & Resources

### Documentation
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)

### Key Papers
- **BART:** Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training" (2019)
- **T5:** Raffel et al., "Exploring the Limits of Transfer Learning" (2019)
- **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **RLHF:** Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
- **DPO:** Rafailov et al., "Direct Preference Optimization" (2023)

### Datasets for Summarization
- `cnn_dailymail` — News articles (long document → multi-sentence summary)
- `xsum` — BBC articles (long document → 1 sentence summary)
- `samsum` — Dialogue summarization
- `multi_news` — Multi-document summarization
- `big_patent` — Patent document summarization
- `openai/summarize_from_feedback` — Summarization with human preference labels

### Free GPU Resources
- **Google Colab** — Free T4 GPU (15GB)
- **Kaggle Notebooks** — Free T4/P100 GPU (16GB)
- **Lightning AI** — Free GPU credits
- **Hugging Face Spaces** — Free inference hosting

---

*Happy fine-tuning! Start small, experiment often, and scale up when your pipeline works.*