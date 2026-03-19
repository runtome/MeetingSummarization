# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meeting summarization system using **Qwen2.5-7B-Instruct** fine-tuned with **QLoRA** (4-bit NF4 quantization + LoRA adapters). Produces structured meeting minutes with four sections: Summary, Key Points, Decisions, Action Items. Supports Thai + English mixed-language meetings.

There is also a secondary CNN-DailyMail news summarization task with its own config and evaluation scripts.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare meeting dataset (AMI corpus + optional synthetic data)
python -m src.data.prepare_dataset --config configs/training_config.yaml
python -m src.data.prepare_dataset --config configs/training_config.yaml --generate-synthetic --api-key $OPENAI_API_KEY --num-synthetic 200

# Prepare CNN-DailyMail dataset
python -m src.data.prepare_cnndm_dataset --config configs/cnndm_training_config.yaml

# Train
python -m src.train --config configs/training_config.yaml
python -m src.train --config configs/training_config.yaml --resume-from-checkpoint

# Evaluate
python -m src.evaluate --adapter-path ./outputs/final_adapter --test-data ./data/processed/val.jsonl --config configs/training_config.yaml --max-samples 100
python -m src.evaluate_cnndm --adapter-path ./outputs/cnndm/final_adapter --test-csv path/to/test.csv --config configs/cnndm_training_config.yaml --max-samples 500

# Inference (CLI)
python -m src.inference --adapter-path ./outputs/final_adapter --input meeting.txt --config configs/training_config.yaml --output summary.md

# Web UI
streamlit run app.py
```

All source modules are run with `python -m src.<module>` (not as scripts directly).

## Architecture

### Pipeline Flow
```
Transcript → Thai+English normalization → Chunking (with overlap) → Per-chunk QLoRA summarization → Merge summaries → Structured output
```

### Key Modules

- **`src/train.py`** — QLoRA fine-tuning via TRL's `SFTTrainer`. Auto-detects GPU capability (Ampere→bfloat16+flash_attention_2, Pascal/Volta→float16+sdpa). Supports both JSONL (ChatML messages) and CSV input formats. Saves final adapter to `output_dir/final_adapter/`.
- **`src/inference.py`** — Full inference pipeline: `load_model()` → `chunk_transcript()` → `summarize_chunk()` → `merge_summaries()` → `parse_sections()`. Entry point: `summarize_meeting()`.
- **`src/evaluate.py`** — ROUGE scoring with Thai pre-tokenization (PyThaiNLP) + structure quality scoring (checks all 4 sections present and populated).
- **`src/evaluate_cnndm.py`** — Enhanced evaluation for news summarization with timing stats and JSON report generation.
- **`src/data/prepare_dataset.py`** — Loads AMI corpus from HuggingFace (`knkarthick/AMI`), optionally generates synthetic Thai/English meeting data via OpenAI API. Outputs ChatML-formatted JSONL.
- **`src/data/thai_utils.py`** — Thai text normalization pipeline: `normalize_thai_text()` → `remove_fillers()` → `normalize_speaker_labels()` → `normalize_whitespace()`. Main entry: `normalize_meeting_text()`.
- **`app.py`** — Streamlit web UI with model caching, text/file input, adjustable parameters, tabbed output, and Markdown download.

### Configuration

YAML configs in `configs/` control model, LoRA, training, inference, and data settings:
- `training_config.yaml` — Meeting summarization (LoRA rank 32, seq len 4096, batch 4×4)
- `cnndm_training_config.yaml` — News summarization (LoRA rank 16, seq len 512, batch 2×8, optimized for P100)

### Data Formats

- Training data: JSONL with ChatML `messages` format (system/user/assistant) or CSV with `article`/`highlights` columns
- Processed data goes in `./data/processed/` (train.jsonl, val.jsonl)
- Large artifacts excluded via .gitignore: `datasets/`, `outputs/`, `models/`, `data/`

## Technical Notes

- GPU memory management: uses `PYTORCH_ALLOC_CONF="expandable_segments:True"`, gradient checkpointing, paged AdamW 8-bit optimizer
- No unit test framework; validation is done through ROUGE evaluation scripts
- Synthetic data generation uses language distribution: 50% Thai+English mixed, 25% Thai only, 25% English only
