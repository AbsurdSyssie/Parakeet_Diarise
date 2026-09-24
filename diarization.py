#!/usr/bin/env python3
"""Shared NeMo speaker-diarization runtime."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
import torchaudio

from settings import SETTINGS

DIARIZATION_MODEL = SETTINGS.diarization_model

_MODEL = None
_MODEL_LOCK = threading.RLock()


def is_diarizer_loaded() -> bool:
    with _MODEL_LOCK:
        return _MODEL is not None


def _configure_streaming(model) -> None:
    modules = model.sortformer_modules
    modules.chunk_len = 340
    modules.chunk_right_context = 40
    modules.fifo_len = 40
    modules.spkcache_update_period = 300

    if hasattr(model, "_check_streaming_parameters"):
        model._check_streaming_parameters()


def _get_diarizer_unlocked():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if os.environ.get("DIARIZE_TF32", "0") == "1" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from nemo.collections.asr.models import SortformerEncLabelModel

    print(f"Loading diarization model: {DIARIZATION_MODEL}")
    model = SortformerEncLabelModel.from_pretrained(DIARIZATION_MODEL)
    model.eval()
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    _configure_streaming(model)
    _MODEL = model
    return model


def get_diarizer():
    """Load the configured diarization model once and return it."""
    with _MODEL_LOCK:
        return _get_diarizer_unlocked()


def unload_diarizer() -> bool:
    """Drop the cached diarizer. CUDA cleanup remains the caller's responsibility."""
    global _MODEL

    with _MODEL_LOCK:
        if _MODEL is None:
            return False

        model = _MODEL
        _MODEL = None
        del model
        return True


def normalize_speaker(label: Any) -> str:
    """Normalize model speaker labels to SPEAKER_XX where possible."""
    if isinstance(label, int) and not isinstance(label, bool):
        return f"SPEAKER_{label:02d}"
    if not isinstance(label, str):
        return "UNKNOWN"

    label = label.strip()
    if label.isdigit():
        return f"SPEAKER_{int(label):02d}"
    if label.startswith("SPEAKER_"):
        return label
    if label.startswith("speaker_"):
        try:
            return f"SPEAKER_{int(label.split('_', 1)[1]):02d}"
        except (ValueError, IndexError):
            return label
    return label or "UNKNOWN"


def parse_diarization_output(predicted_segments) -> list[dict]:
    """Convert NeMo diarization output into sorted speaker turns."""
    raw = predicted_segments[0] if predicted_segments else []
    turns: list[dict] = []

    for segment in raw:
        try:
            if isinstance(segment, dict):
                start = float(segment.get("start", segment.get("start_time", 0.0)))
                end = float(segment.get("end", segment.get("end_time", 0.0)))
                speaker = segment.get(
                    "speaker",
                    segment.get("speaker_label", segment.get("label", "UNKNOWN")),
                )
            elif isinstance(segment, str):
                parts = segment.strip().split()
                if len(parts) < 3:
                    continue
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
            elif isinstance(segment, (list, tuple)) and len(segment) >= 3:
                start = float(segment[0])
                end = float(segment[1])
                speaker = segment[2]
            else:
                continue
        except (TypeError, ValueError):
            continue

        turns.append(
            {
                "start": start,
                "end": end,
                "speaker": normalize_speaker(speaker),
            }
        )

    turns.sort(key=lambda turn: (turn["start"], turn["end"]))
    return turns


def diarize_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    tmpdir: Path,
) -> list[dict]:
    """Run diarization on an audio tensor and return normalized speaker turns."""
    wav = waveform.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() != 2:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(wav.shape)}")

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        sample_rate = 16000

    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    mono_path = tmpdir / "diarize_mono16k.wav"
    torchaudio.save(str(mono_path), wav, sample_rate)

    with _MODEL_LOCK:
        model = _get_diarizer_unlocked()
        predicted_segments = model.diarize(audio=[str(mono_path)], batch_size=1)

    return parse_diarization_output(predicted_segments)
