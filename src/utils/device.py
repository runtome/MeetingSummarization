"""Shared GPU capability detection for MeetingSummarization scripts."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "f16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def detect_compute_dtype_and_attn(preferred_dtype: str | None = None) -> tuple[
    torch.dtype, str, bool, int
]:
    """Return the compute dtype, attention implementation, and GPU metadata."""
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    is_ampere_plus = major >= 8
    default_dtype = torch.bfloat16 if is_ampere_plus else torch.float16
    attn_impl = "flash_attention_2" if is_ampere_plus else "sdpa"

    dtype = default_dtype
    if preferred_dtype:
        normalized = preferred_dtype.strip().lower()
        mapped = _DTYPE_ALIASES.get(normalized)
        if mapped is None:
            logger.warning(
                "Unknown dtype '%s' requested; falling back to %s.",
                preferred_dtype,
                default_dtype,
            )
        elif mapped is torch.bfloat16 and not is_ampere_plus:
            logger.warning(
                "Requested bf16 compute on sm_%s0 is unsupported; using fp16 instead.",
                major,
            )
            dtype = torch.float16
        else:
            dtype = mapped

    return dtype, attn_impl, is_ampere_plus, major


__all__ = ["detect_compute_dtype_and_attn"]
