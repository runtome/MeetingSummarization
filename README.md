# Meeting Summarization

Fine-tuned meeting summarization system using **Qwen2.5-7B-Instruct** with **QLoRA**. Supports Thai + English mixed-language meetings and produces structured meeting minutes.

## Output Format

The model generates structured summaries with four sections:

```
## Summary
2-3 sentence overview of the meeting.

## Key Points
- Key discussion point 1
- Key discussion point 2

## Decisions
- Decision made during the meeting

## Action Items
- John → Prepare report by Friday
- Mary → Review budget proposal by Monday
```

## Project Structure

```
MeetingSummarization/
├── configs/
│   └── training_config.yaml    # Model, QLoRA, training & inference settings
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py  # AMI corpus loading + synthetic data generation
│   │   └── thai_utils.py       # Thai text normalization (PyThaiNLP)
│   ├── train.py                # QLoRA fine-tuning with TRL SFTTrainer
│   ├── inference.py            # Chunk → summarize → merge pipeline
│   └── evaluate.py             # ROUGE metrics + structure quality checks
├── app.py                      # Streamlit demo UI
├── requirements.txt
└── .gitignore
```

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (A100 recommended, Colab-compatible with smaller models)

### Install

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Load the AMI Meeting Corpus and optionally generate synthetic Thai+English training data:

```bash
# AMI corpus only
python -m src.data.prepare_dataset --config configs/training_config.yaml

# With synthetic data generation (requires OpenAI-compatible API)
python -m src.data.prepare_dataset \
    --config configs/training_config.yaml \
    --generate-synthetic \
    --api-key YOUR_API_KEY \
    --num-synthetic 200
```

Options:
- `--api-base` — Custom API base URL for compatible providers (default: OpenAI)
- `--model` — Model for synthetic generation (default: `gpt-4o-mini`)
- `--num-synthetic` — Number of synthetic samples (default: 200)

Output: `data/processed/train.jsonl` and `data/processed/val.jsonl`

### 2. Train

Fine-tune Qwen2.5-7B with QLoRA:

```bash
python -m src.train --config configs/training_config.yaml
```

Training details:
- 4-bit NF4 quantization with double quantization
- LoRA rank 32, alpha 64, targeting all linear layers
- Cosine LR schedule, paged AdamW 8-bit optimizer
- Gradient checkpointing enabled

Output: LoRA adapter saved to `outputs/final_adapter/`

### 3. Evaluate

```bash
python -m src.evaluate \
    --adapter-path ./outputs/final_adapter \
    --test-data ./data/processed/val.jsonl \
    --config configs/training_config.yaml
```

Metrics:
- ROUGE-1, ROUGE-2, ROUGE-L (with Thai pre-tokenization via PyThaiNLP)
- Structure quality score (checks all 4 sections are present and non-empty)

### 4. Run Inference

Summarize a transcript file:

```bash
python -m src.inference \
    --adapter-path ./outputs/final_adapter \
    --input meeting_transcript.txt \
    --config configs/training_config.yaml \
    --output summary.md
```

The pipeline automatically handles long transcripts by chunking, summarizing each chunk, and merging results.

### 5. Streamlit Demo

```bash
streamlit run app.py
```

Features:
- Paste text or upload `.txt` file
- Adjustable temperature, token limit, and chunk size
- Tabbed output view (Summary / Key Points / Decisions / Action Items)
- Download summary as Markdown

## Architecture

```
Raw Transcript
     ↓
Thai + English Normalization (PyThaiNLP)
     ↓
Chunking (sentence-boundary splitting with overlap)
     ↓
Fine-tuned Qwen2.5-7B QLoRA (per-chunk summarization)
     ↓
Merge Summaries (deduplicate + condense)
     ↓
Structured Meeting Minutes
```

## Dataset Strategy

Three data sources combined:

| Source | Language | Size |
|--------|----------|------|
| AMI Meeting Corpus | English | ~279 examples |
| Synthetic (LLM-generated) | Thai + English mixed | 200-500 examples |
| Synthetic (LLM-generated) | Thai / English only | included in above |

Synthetic data uses 50% mixed Thai+English, 25% Thai-only, 25% English-only to reflect real Thai workplace meetings.

## Configuration

All hyperparameters are centralized in `configs/training_config.yaml`. Key settings:

| Parameter | Value | Note |
|-----------|-------|------|
| Base model | Qwen2.5-7B-Instruct | Strong multilingual support |
| Quantization | 4-bit NF4 | QLoRA memory efficiency |
| LoRA rank | 32 | Good balance for summarization |
| LoRA alpha | 64 | Scaling factor = 2.0 |
| Learning rate | 2e-4 | Standard for QLoRA |
| Epochs | 3 | Sufficient for LoRA |
| Max seq length | 4096 | Covers chunked transcripts |
| Chunk size | 3000 chars | With 300 char overlap |

## License

MIT
