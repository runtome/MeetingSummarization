"""Thai text normalization utilities for mixed Thai/English meeting transcripts."""

import re
import unicodedata

from pythainlp.util import normalize as thai_normalize


# Invisible Unicode characters to remove
INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f\u061c\u115f\u1160"
    "\u17b4\u17b5\u180e\u2000-\u200f\u202a-\u202e\u2066-\u2069\ufff9-\ufffb]"
)

# Common filler words
THAI_FILLERS = re.compile(r"\b(?:อ้า|เอ่อ|อืม|อื้อ|เออ|หืม|อ่า)\b")
EN_FILLERS = re.compile(r"\b(?:um|uh|uhm|hmm|hm|er|erm|like,?\s+you know)\b", re.IGNORECASE)

# Speaker label patterns → normalize to "Speaker N:" format
SPEAKER_PATTERNS = re.compile(
    r"^(ผู้พูด|speaker|spk|ผู้เข้าร่วม|participant)\s*(\d+)\s*[:：]",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_thai_text(text: str) -> str:
    """Apply PyThaiNLP normalization and remove invisible Unicode characters."""
    text = thai_normalize(text)
    text = INVISIBLE_CHARS.sub("", text)
    return text


def remove_fillers(text: str) -> str:
    """Remove common filler words in Thai and English."""
    text = THAI_FILLERS.sub("", text)
    text = EN_FILLERS.sub("", text)
    # Clean up resulting double spaces
    text = re.sub(r"  +", " ", text)
    return text


def normalize_speaker_labels(text: str) -> str:
    """Standardize speaker labels to 'Speaker N:' format."""

    def replace_label(match):
        number = match.group(2)
        return f"Speaker {number}:"

    return SPEAKER_PATTERNS.sub(replace_label, text)


def normalize_whitespace(text: str) -> str:
    """Clean up whitespace: collapse multiple spaces/newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def normalize_meeting_text(text: str) -> str:
    """Full normalization pipeline for meeting transcripts.

    Applies: Thai normalization → filler removal → speaker label
    standardization → whitespace cleanup.
    """
    text = normalize_thai_text(text)
    text = remove_fillers(text)
    text = normalize_speaker_labels(text)
    text = normalize_whitespace(text)
    return text
