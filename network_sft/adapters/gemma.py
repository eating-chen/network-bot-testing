"""Gemma renderer boundary; Gemma-specific adjustments belong only here."""

from __future__ import annotations

from typing import Any

from network_sft.adapters.huggingface import render_for_hf


def render_for_gemma(
    example: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return render_for_hf(example)
