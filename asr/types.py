"""Shared ASR adapter types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ASRConfig:
    backend: str
    model_key: str
    model_name: str
    hf_token: str | None = None
    language: str = "en"
    task: str = "transcribe"
    trust_remote_code: bool = False
    return_timestamps: bool = False
    chunk_length_s: int | None = None
    stride_length_s: int | tuple[int, int] | None = None
    device: str | None = None
    attention_implementation: str | None = None


@dataclass
class SimpleHypothesis:
    text: str
    timestamp: dict[str, list[dict[str, Any]]]
