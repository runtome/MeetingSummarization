"""
CNN-DailyMail evaluation script for the QLoRA fine-tuned summarization model.

Usage
-----
python -m src.evaluate_cnndm \
    --adapter-path ./outputs/cnndm/final_adapter \
    --test-csv     /kaggle/working/MeetingSummarization/datasets/test.csv \
    --config       configs/cnndm_training_config.yaml \
    --output       ./outputs/cnndm/eval_results.json \
    --max-samples  500        # omit to evaluate the full test set
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Prompt template (must match prepare_cnndm_dataset.py) ────────────────────
SYSTEM_PROMPT = (
    "You are a news summarization assistant. Given a news article, "
    "write a concise and accurate summary capturing the most important facts."
)
USER_TEMPLATE = "Summarize the following news article:\n\n{article}"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def detect_gpu():
    """Return (dtype, attn_impl) appropriate for the current GPU."""
    if not torch.cuda.is_available():
        return torch.float32, "eager"
    major = torch.cuda.get_device_capability()[0]
    is_ampere_plus = major >= 8
    dtype     = torch.bfloat16 if is_ampere_plus else torch.float16
    attn_impl = "flash_attention_2" if is_ampere_plus else "sdpa"
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}  sm_{major}0 → dtype={dtype}, attn={attn_impl}")
    return dtype, attn_impl


def load_model(adapter_path: str, base_model_name: str):
    """Load the base model with 4-bit QLoRA and apply the fine-tuned adapter."""
    dtype, attn_impl = detect_gpu()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test_csv(path: str, max_samples: int | None = None) -> pd.DataFrame:
    """Load the CNN-DailyMail test CSV and return a DataFrame."""
    df = pd.read_csv(path)

    # Normalise column names (handle lowercase / title-case variants)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"article", "highlights"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Test CSV is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows with missing values in either column
    before = len(df)
    df = df.dropna(subset=["article", "highlights"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing article/highlights.")

    if max_samples:
        df = df.head(max_samples)

    print(f"Loaded {len(df)} test examples from {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(
    article: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Generate a summary for one article using the chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(article=article.strip())},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE scoring
# ─────────────────────────────────────────────────────────────────────────────

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Return mean / min / max precision, recall, F1 for ROUGE-1/2/L."""
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)

    per_metric: dict[str, dict[str, list[float]]] = {
        k: {"precision": [], "recall": [], "fmeasure": []}
        for k in ROUGE_KEYS
    }

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in ROUGE_KEYS:
            per_metric[key]["precision"].append(scores[key].precision)
            per_metric[key]["recall"].append(scores[key].recall)
            per_metric[key]["fmeasure"].append(scores[key].fmeasure)

    summary: dict = {}
    for key, vals in per_metric.items():
        summary[key] = {
            stat: {
                "mean": sum(v) / len(v),
                "min":  min(v),
                "max":  max(v),
            }
            for stat, v in vals.items()
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    adapter_path: str,
    test_csv: str,
    config: dict,
    max_samples: int | None = None,
    save_predictions: bool = True,
) -> dict:
    """Full evaluation pipeline: load → generate → score → report."""

    inf_cfg = config.get("inference", {})
    max_new_tokens = inf_cfg.get("max_new_tokens", 256)
    temperature    = inf_cfg.get("temperature", 0.3)
    top_p          = inf_cfg.get("top_p", 0.9)

    # 1. Load data
    df = load_test_csv(test_csv, max_samples=max_samples)

    # 2. Load model + adapter
    model, tokenizer = load_model(adapter_path, config["model_name"])

    # 3. Generate summaries
    predictions: list[str] = []
    references:  list[str] = []
    per_sample:  list[dict] = []

    print(f"\nGenerating summaries for {len(df)} articles…")
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        article    = str(row["article"])
        reference  = str(row["highlights"])

        prediction = generate_summary(
            article, model, tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        predictions.append(prediction)
        references.append(reference)

        if save_predictions:
            per_sample.append({
                "article":    article[:500] + "…" if len(article) > 500 else article,
                "reference":  reference,
                "prediction": prediction,
            })

    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.1f}s  "
          f"({elapsed / len(df):.2f}s / sample)")

    # 4. Compute ROUGE
    print("\nComputing ROUGE scores…")
    rouge_results = compute_rouge(predictions, references)

    # 5. Pretty-print
    print("\n" + "=" * 60)
    print("  CNN-DAILYMAIL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated : {len(predictions)}")
    print(f"  Adapter           : {adapter_path}")
    print("-" * 60)
    for metric in ROUGE_KEYS:
        f1   = rouge_results[metric]["fmeasure"]["mean"] * 100
        rec  = rouge_results[metric]["recall"]["mean"]   * 100
        prec = rouge_results[metric]["precision"]["mean"]* 100
        print(f"  {metric.upper():<8}  F1={f1:5.2f}   P={prec:5.2f}   R={rec:5.2f}")
    print("=" * 60)

    # 6. Sample predictions (first 3)
    print("\n── Sample predictions ──────────────────────────────────────")
    for i, s in enumerate(per_sample[:3]):
        print(f"\n[{i+1}] REFERENCE : {s['reference'][:200]}")
        print(f"    PREDICTION: {s['prediction'][:200]}")

    # 7. Package results
    results = {
        "adapter_path":   adapter_path,
        "test_csv":       test_csv,
        "num_samples":    len(predictions),
        "elapsed_seconds": round(elapsed, 2),
        "rouge":          rouge_results,
    }
    if save_predictions:
        results["per_sample"] = per_sample

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QLoRA fine-tuned model on CNN-DailyMail test set"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./outputs/cnndm/final_adapter",
        help="Path to the saved LoRA adapter (output of training)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="/kaggle/working/MeetingSummarization/datasets/test.csv",
        help="Path to the CNN-DailyMail test CSV file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnndm_training_config.yaml",
        help="Training config YAML (for model_name and inference settings)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/cnndm/eval_results.json",
        help="Where to save the JSON evaluation report",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N rows (omit for the full test set)",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not include per-sample predictions in the JSON output",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results = evaluate(
        adapter_path=args.adapter_path,
        test_csv=args.test_csv,
        config=config,
        max_samples=args.max_samples,
        save_predictions=not args.no_save_predictions,
    )

    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
