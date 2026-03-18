"""Dataset preparation: AMI corpus loading, synthetic data generation, and formatting."""

import argparse
import json
import os
import random
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.data.thai_utils import normalize_meeting_text

SYSTEM_PROMPT = (
    "You are a meeting summarization assistant. Given a meeting transcript, "
    "provide a structured summary with the following sections:\n\n"
    "## Summary\n[2-3 sentence overview of the meeting]\n\n"
    "## Key Points\n- [key discussion points as bullet list]\n\n"
    "## Decisions\n- [decisions made during the meeting]\n\n"
    "## Action Items\n- [action items with assignee and deadline if mentioned]"
)

USER_TEMPLATE = "Summarize the following meeting transcript into structured meeting minutes:\n\n{transcript}"

# Topics for synthetic data generation
MEETING_TOPICS = [
    "quarterly budget review",
    "product launch planning",
    "engineering sprint retrospective",
    "marketing campaign strategy",
    "hiring pipeline discussion",
    "customer feedback review",
    "system architecture redesign",
    "Q3 sales performance",
    "office relocation planning",
    "annual company retreat",
    "API integration with partner",
    "data privacy compliance",
    "mobile app feature prioritization",
    "vendor contract negotiation",
    "team restructuring proposal",
]

SYNTHETIC_PROMPT = """Generate a realistic meeting transcript and its structured summary.

Meeting topic: {topic}
Language: {language}
Number of participants: {num_participants}
Meeting length: {length}

Generate the output in this exact JSON format:
{{
  "transcript": "Speaker 1: ... \\nSpeaker 2: ...",
  "summary": "## Summary\\n...\\n\\n## Key Points\\n- ...\\n\\n## Decisions\\n- ...\\n\\n## Action Items\\n- ..."
}}

Requirements for the transcript:
- Use realistic dialogue with natural speech patterns
- Include back-and-forth discussion, not just monologues
- {lang_instruction}
- Include specific names, dates, and numbers where relevant

Requirements for the summary:
- Summary section: 2-3 sentences
- Key Points: 3-5 bullet points
- Decisions: 1-3 bullet points (use "- None" if no clear decisions)
- Action Items: 2-4 items with format "- [Person] → [Task] by [Deadline]"

Return ONLY the JSON, no other text."""


def load_ami_corpus(dataset_name: str) -> list[dict]:
    """Load AMI Meeting Corpus and format as training examples."""
    print("Loading AMI corpus...")
    ds = load_dataset(dataset_name)
    examples = []

    for split in ds:
        for item in ds[split]:
            dialogue = item.get("dialogue", "")
            summary = item.get("summary", "")
            if not dialogue or not summary:
                continue

            # Wrap flat AMI summary into structured format
            structured = (
                f"## Summary\n{summary}\n\n"
                f"## Key Points\n- Key discussion points covered in the meeting\n\n"
                f"## Decisions\n- None explicitly recorded\n\n"
                f"## Action Items\n- None explicitly recorded"
            )

            examples.append({
                "transcript": dialogue,
                "structured_output": structured,
            })

    print(f"Loaded {len(examples)} examples from AMI corpus")
    return examples


def generate_synthetic_data(
    num_samples: int,
    api_key: str,
    output_path: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Generate synthetic meeting transcript/summary pairs using an LLM API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    languages = [
        ("Thai + English mixed", "Mix Thai and English naturally, as in a real Thai workplace meeting"),
        ("Thai only", "Write entirely in Thai"),
        ("English only", "Write entirely in English"),
    ]
    lang_weights = [0.5, 0.25, 0.25]

    print(f"Generating {num_samples} synthetic samples...")

    for i in tqdm(range(num_samples)):
        topic = random.choice(MEETING_TOPICS)
        lang_name, lang_instruction = random.choices(languages, weights=lang_weights, k=1)[0]
        num_participants = random.randint(2, 6)
        length = random.choice(["short (5 min)", "medium (15 min)", "long (30 min)"])

        prompt = SYNTHETIC_PROMPT.format(
            topic=topic,
            language=lang_name,
            num_participants=num_participants,
            length=length,
            lang_instruction=lang_instruction,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response (handle markdown code blocks)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            data = json.loads(content)
            if "transcript" in data and "summary" in data:
                examples.append({
                    "transcript": data["transcript"],
                    "structured_output": data["summary"],
                })
                # Save incrementally
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Sample {i} failed: {e}")
            continue

    print(f"Generated {len(examples)} synthetic samples")
    return examples


def format_for_training(examples: list[dict]) -> list[dict]:
    """Convert examples to ChatML messages format for SFTTrainer."""
    formatted = []
    for ex in examples:
        transcript = normalize_meeting_text(ex["transcript"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)},
            {"role": "assistant", "content": ex["structured_output"]},
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


def build_dataset(config: dict, generate_synthetic: bool = False,
                  api_key: str = None, num_synthetic: int = 200,
                  api_base: str = "https://api.openai.com/v1",
                  model: str = "gpt-4o-mini"):
    """Main dataset building pipeline."""
    data_config = config["data"]
    all_examples = []

    # 1. Load AMI corpus
    ami_examples = load_ami_corpus(data_config["ami_dataset"])
    all_examples.extend(ami_examples)

    # 2. Generate or load synthetic data
    synthetic_path = data_config["synthetic_output"]
    if generate_synthetic and api_key:
        synthetic_examples = generate_synthetic_data(
            num_samples=num_synthetic,
            api_key=api_key,
            output_path=synthetic_path,
            api_base=api_base,
            model=model,
        )
        all_examples.extend(synthetic_examples)
    elif os.path.exists(synthetic_path):
        print(f"Loading existing synthetic data from {synthetic_path}")
        with open(synthetic_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_examples.append({
                    "transcript": data["transcript"],
                    "structured_output": data["summary"],
                })
        print(f"Loaded {len(all_examples) - len(ami_examples)} synthetic examples")

    # 3. Format for training
    formatted = format_for_training(all_examples)

    # 4. Shuffle and split
    random.shuffle(formatted)
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    # 5. Save
    save_jsonl(train_data, data_config["train_file"])
    save_jsonl(val_data, data_config["val_file"])

    print(f"\nDataset ready: {len(train_data)} train, {len(val_data)} val")


def main():
    parser = argparse.ArgumentParser(description="Prepare meeting summarization dataset")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Generate synthetic training data using LLM API")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for LLM (or set OPENAI_API_KEY env var)")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1",
                        help="API base URL (for compatible APIs)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use for synthetic generation")
    parser.add_argument("--num-synthetic", type=int, default=200,
                        help="Number of synthetic samples to generate")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    build_dataset(
        config=config,
        generate_synthetic=args.generate_synthetic,
        api_key=api_key,
        num_synthetic=args.num_synthetic,
        api_base=args.api_base,
        model=args.model,
    )


if __name__ == "__main__":
    main()
