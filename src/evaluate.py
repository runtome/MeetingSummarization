"""Evaluation: ROUGE scores and structured output quality checks."""

import argparse
import json

import yaml
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.inference import load_model, parse_sections, summarize_meeting

REQUIRED_SECTIONS = ["Summary", "Key Points", "Decisions", "Action Items"]


def _find_message_content(messages: list[dict], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


def preprocess_for_rouge(text: str) -> str:
    """Pre-tokenize Thai text for ROUGE scoring by inserting spaces."""
    try:
        from pythainlp.tokenize import word_tokenize

        # Check if text contains Thai characters
        has_thai = any("\u0e00" <= c <= "\u0e7f" for c in text)
        if has_thai:
            tokens = word_tokenize(text, engine="newmm")
            return " ".join(tokens)
    except ImportError:
        pass
    return text


def compute_rouge_scores(
    predictions: list[str], references: list[str]
) -> dict[str, dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        pred = preprocess_for_rouge(pred)
        ref = preprocess_for_rouge(ref)
        scores = scorer.score(ref, pred)

        for key in all_scores:
            all_scores[key].append(scores[key].fmeasure)

    # Compute means
    results = {}
    for key, values in all_scores.items():
        results[key] = {
            "mean": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
        }

    return results


def check_structure_quality(output: str) -> dict:
    """Check if output contains all required sections with content."""
    sections = parse_sections(output)

    section_results = {}
    for section in REQUIRED_SECTIONS:
        present = section in sections
        content = sections.get(section, "")
        has_content = bool(content.strip()) and content.strip().lower() not in [
            "none",
            "n/a",
            "-",
        ]
        bullet_count = len(
            [line for line in content.splitlines() if line.strip().startswith("-")]
        )
        section_results[section] = {
            "present": present,
            "has_content": has_content,
            "bullet_count": bullet_count,
        }

    # Overall score: fraction of sections present and with content
    score = sum(
        1 for s in section_results.values() if s["present"] and s["has_content"]
    ) / len(REQUIRED_SECTIONS)

    return {
        "sections": section_results,
        "score": score,
        "has_all_sections": all(s["present"] for s in section_results.values()),
    }


def evaluate_model(
    adapter_path: str,
    test_data_path: str,
    config: dict,
    max_samples: int = None,
) -> dict:
    """Run full evaluation: ROUGE + structure quality."""
    preferred_dtype = config.get("quantization", {}).get("bnb_4bit_compute_dtype")
    model, tokenizer = load_model(
        adapter_path,
        config["model_name"],
        preferred_dtype=preferred_dtype,
    )

    # Load test data
    test_examples = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))

    if max_samples:
        test_examples = test_examples[:max_samples]

    predictions = []
    references = []
    structure_scores = []

    print(f"Evaluating on {len(test_examples)} examples...")

    for example in tqdm(test_examples):
        messages = example.get("messages", [])
        if not messages:
            print("Skipping example without messages")
            continue
        user_content = _find_message_content(messages, "user")
        assistant_content = _find_message_content(messages, "assistant")
        if not user_content or not assistant_content:
            print("Skipping example missing user or assistant messages")
            continue
        transcript = user_content.removeprefix(
            "Summarize the following meeting transcript into structured meeting minutes:\n\n"
        ).strip()
        if not transcript:
            print("Skipping example with empty transcript after prefix removal")
            continue
        reference = assistant_content

        # Generate prediction
        prediction = summarize_meeting(transcript, model, tokenizer, config)

        predictions.append(prediction)
        references.append(reference)

        # Check structure
        quality = check_structure_quality(prediction)
        structure_scores.append(quality["score"])

    # Compute ROUGE
    rouge_results = compute_rouge_scores(predictions, references)

    # Aggregate structure scores
    avg_structure = sum(structure_scores) / len(structure_scores) if structure_scores else 0

    results = {
        "rouge": rouge_results,
        "structure_quality": {
            "mean": avg_structure,
            "min": min(structure_scores) if structure_scores else 0,
            "max": max(structure_scores) if structure_scores else 0,
        },
        "num_samples": len(predictions),
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, values in rouge_results.items():
        print(f"{metric}: {values['mean']:.4f} (min={values['min']:.4f}, max={values['max']:.4f})")
    print(f"Structure Quality: {avg_structure:.4f}")
    print(f"Samples evaluated: {len(predictions)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate meeting summarization model")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results = evaluate_model(
        adapter_path=args.adapter_path,
        test_data_path=args.test_data,
        config=config,
        max_samples=args.max_samples,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
