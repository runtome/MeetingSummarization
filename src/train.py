"""QLoRA fine-tuning script for meeting summarization using TRL's SFTTrainer."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_jsonl_dataset(path: str) -> Dataset:
    """Load a JSONL file with messages format into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model for meeting summarization")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    max_seq_length = config["max_seq_length"]
    q_config = config["quantization"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # 1. Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_config["load_in_4bit"],
        bnb_4bit_quant_type=q_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q_config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q_config["bnb_4bit_use_double_quant"],
    )

    # 2. Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # 3. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA config
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )

    # 5. Load datasets
    print("Loading datasets...")
    train_dataset = load_jsonl_dataset(data_cfg["train_file"])
    val_dataset = load_jsonl_dataset(data_cfg["val_file"])
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 6. Training arguments (SFTConfig = TrainingArguments + max_seq_length, TRL ≥ 0.12)
    output_dir = train_cfg["output_dir"]

    # Convert warmup_ratio → warmup_steps to avoid the deprecation warning.
    effective_batch = train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]
    total_steps = (len(train_dataset) // effective_batch) * train_cfg["num_train_epochs"]
    warmup_steps = max(1, int(total_steps * train_cfg["warmup_ratio"]))

    training_args = SFTConfig(
        # ── output ──────────────────────────────────────────────
        output_dir=output_dir,
        # ── sequence length (replaces SFTTrainer kwarg) ─────────
        max_seq_length=max_seq_length,
        # ── schedule ────────────────────────────────────────────
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=warmup_steps,              # replaces deprecated warmup_ratio
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        # ── logging / saving / eval ──────────────────────────────
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=3,
        # ── precision / memory ───────────────────────────────────
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        max_grad_norm=train_cfg["max_grad_norm"],
        report_to="none",
    )

    # 7. Setup trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,          # max_seq_length now lives inside SFTConfig
    )

    # 8. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 9. Save adapter
    adapter_path = Path(output_dir) / "final_adapter"
    print(f"Saving adapter to {adapter_path}")
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    print("Training complete!")


if __name__ == "__main__":
    main()
