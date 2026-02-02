#!/usr/bin/env python3
"""Diarization API service using NeMo Sortformer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import time

import soundfile as sf
import gc
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from nemo.collections.asr.models import SortformerEncLabelModel

app = FastAPI()

_MODEL = None


def cuda_mem():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline_loaded": _PIPELINE is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_mem": cuda_mem(),
    }


def _get_model() -> SortformerEncLabelModel:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if os.environ.get("DIARIZE_TF32", "0") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1"
    )
    model.eval()
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    model.sortformer_modules.chunk_len = 340
    model.sortformer_modules.chunk_right_context = 40
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.spkcache_update_period = 300

    _MODEL = model
    return model


def _normalize_speaker(label: str) -> str:
    if not isinstance(label, str):
        return "UNKNOWN"
    if label.startswith("SPEAKER_"):
        return label
    if label.startswith("speaker_"):
        try:
            idx = int(label.split("_", 1)[1])
            return f"SPEAKER_{idx:02d}"
        except (ValueError, IndexError):
            return label
    return label


@app.post("/v1/diarize")
async def diarize(file: UploadFile = File(...)):
    model = _get_model()
    empty_cache = os.environ.get("DIARIZE_EMPTY_CACHE", "0") == "1"

    with tempfile.TemporaryDirectory() as tmpdir:
        start_req = time.perf_counter()
        tmp_path = Path(tmpdir) / file.filename
        tmp_path.write_bytes(await file.read())

        waveform, sr = torchaudio.load(tmp_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)

        mono_path = Path(tmpdir) / f"{tmp_path.stem}.mono16k.wav"
        sf.write(str(mono_path), waveform.numpy(), sr)

        print(f"cuda mem before diarize: {cuda_mem()}")
        predicted_segments = model.diarize(audio=[str(mono_path)], batch_size=1)
        diarize_elapsed = time.perf_counter() - start_req
        print(f"diarize in {diarize_elapsed:.2f}s")
        print(f"cuda mem after diarize: {cuda_mem()}")

        turns = []
        raw = predicted_segments[0] if predicted_segments else []
        for seg in raw:
            if isinstance(seg, dict):
                start = float(seg.get("start", seg.get("start_time", 0.0)))
                end = float(seg.get("end", seg.get("end_time", 0.0)))
                speaker = seg.get("speaker", seg.get("speaker_label", seg.get("label", "UNKNOWN")))
            elif isinstance(seg, str):
                parts = seg.strip().split()
                if len(parts) >= 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    speaker = parts[2]
                else:
                    continue
            elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
                start = float(seg[0])
                end = float(seg[1])
                speaker = seg[2]
            else:
                continue
            turns.append({"start": start, "end": end, "speaker": _normalize_speaker(speaker)})

    turns.sort(key=lambda t: (t["start"], t["end"]))
    if empty_cache:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"cuda mem after gc.collect: {cuda_mem()}")
    return turns
