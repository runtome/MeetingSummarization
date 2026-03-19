"""Dataset preparation for CNN-DailyMail news summarization fine-tuning."""

import argparse
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a news summarization assistant. Given a news article, "
    "write a concise and accurate summary capturing the most important facts."
)

USER_TEMPLATE = "Summarize the following news article:\n\n{article}"


def load_cnndm(split: str, max_samples: int | None = None) -> list[dict]:
    """Load CNN-DailyMail dataset split and return list of {article, highlights}."""
    print(f"Loading CNN-DailyMail '{split}' split...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    examples = [{"article": row["article"], "highlights": row["highlights"]} for row in ds]
    print(f"Loaded {len(examples)} examples from '{split}' split")
    return examples


def format_for_training(examples: list[dict]) -> list[dict]:
    """Convert CNN-DailyMail examples to ChatML messages format for SFTTrainer."""
    formatted = []
    for ex in tqdm(examples, desc="Formatting"):
        article = ex["article"].strip()
        summary = ex["highlights"].strip()
        if not article or not summary:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(article=article)},
            {"role": "assistant", "content": summary},
        ]
        formatted.append({"messages": messages})
    return formatted


def save_jsonl(data: list[dict], path: str):
    """Save list of dicts as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {path}")


def build_dataset(config: dict):
    """Main dataset building pipeline for CNN-DailyMail."""
    data_cfg = config["data"]
    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    max_test = data_cfg.get("max_test_samples")

    train_examples = load_cnndm("train", max_samples=max_train)
    val_examples = load_cnndm("validation", max_samples=max_val)
    test_examples = load_cnndm("test", max_samples=max_test)

    train_formatted = format_for_training(train_examples)
    val_formatted = format_for_training(val_examples)
    test_formatted = format_for_training(test_examples)

    random.shuffle(train_formatted)

    save_jsonl(train_formatted, data_cfg["train_file"])
    save_jsonl(val_formatted, data_cfg["val_file"])
    save_jsonl(test_formatted, data_cfg["test_file"])

    print(f"\nDataset ready: {len(train_formatted)} train, {len(val_formatted)} val, {len(test_formatted)} test")


def main():
    parser = argparse.ArgumentParser(description="Prepare CNN-DailyMail dataset for fine-tuning")
    parser.add_argument("--config", type=str, default="configs/cnndm_training_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    build_dataset(config)


if __name__ == "__main__":
    main()
