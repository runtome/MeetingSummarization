"""Inference pipeline: chunk transcripts, summarize, and merge into structured output."""

import argparse
import re

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.thai_utils import normalize_meeting_text

SYSTEM_PROMPT = (
    "You are a meeting summarization assistant. Given a meeting transcript, "
    "provide a structured summary with the following sections:\n\n"
    "## Summary\n[2-3 sentence overview of the meeting]\n\n"
    "## Key Points\n- [key discussion points as bullet list]\n\n"
    "## Decisions\n- [decisions made during the meeting]\n\n"
    "## Action Items\n- [action items with assignee and deadline if mentioned]"
)

MERGE_PROMPT = (
    "You are a meeting summarization assistant. Below are partial summaries from "
    "different sections of a meeting. Merge them into a single coherent structured "
    "summary with sections: Summary, Key Points, Decisions, Action Items.\n\n"
    "Deduplicate any repeated points. The final Summary should be 2-3 sentences.\n\n"
    "Partial summaries:\n{partials}"
)

SECTIONS = ["Summary", "Key Points", "Decisions", "Action Items"]


def load_model(
    adapter_path: str,
    base_model: str,
    device_map: str = "auto",
) -> tuple:
    """Load base model with QLoRA quantization and apply LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def chunk_transcript(
    text: str, chunk_size: int = 3000, overlap: int = 300
) -> list[str]:
    """Split transcript into overlapping chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at sentence boundary
            for sep in ["\n\n", "\n", ". ", "। ", "。 "]:
                boundary = text.rfind(sep, start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def summarize_chunk(
    chunk: str,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Summarize a single chunk using the fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Summarize the following meeting transcript into structured meeting minutes:\n\n{chunk}",
        },
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_sections(text: str) -> dict[str, str]:
    """Parse structured summary into sections."""
    sections = {}
    current_section = None
    current_content = []

    for line in text.splitlines():
        # Check for section header
        header_match = re.match(r"^##\s+(.+)$", line.strip())
        if header_match:
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = header_match.group(1).strip()
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def merge_summaries(
    chunk_summaries: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Merge multiple chunk summaries into a single coherent summary."""
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Combine all partial summaries
    partials = "\n\n---\n\n".join(
        f"Part {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": MERGE_PROMPT.format(partials=partials)},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def summarize_meeting(
    transcript: str,
    model,
    tokenizer,
    config: dict,
) -> str:
    """Full pipeline: normalize → chunk → summarize → merge."""
    inf_cfg = config.get("inference", {})
    chunk_size = inf_cfg.get("chunk_size", 3000)
    chunk_overlap = inf_cfg.get("chunk_overlap", 300)
    max_new_tokens = inf_cfg.get("max_new_tokens", 1024)
    temperature = inf_cfg.get("temperature", 0.3)
    top_p = inf_cfg.get("top_p", 0.9)

    # Normalize
    transcript = normalize_meeting_text(transcript)

    # Chunk
    chunks = chunk_transcript(transcript, chunk_size, chunk_overlap)
    print(f"Transcript split into {len(chunks)} chunk(s)")

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = summarize_chunk(
            chunk, model, tokenizer, max_new_tokens, temperature, top_p
        )
        chunk_summaries.append(summary)

    # Merge
    if len(chunk_summaries) > 1:
        print("Merging chunk summaries...")
        return merge_summaries(
            chunk_summaries, model, tokenizer, max_new_tokens, temperature, top_p
        )
    return chunk_summaries[0]


def main():
    parser = argparse.ArgumentParser(description="Summarize a meeting transcript")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to transcript .txt file")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Path to save summary")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_model(args.adapter_path, config["model_name"])

    with open(args.input, "r", encoding="utf-8") as f:
        transcript = f.read()

    result = summarize_meeting(transcript, model, tokenizer, config)
    print("\n" + "=" * 60)
    print(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
