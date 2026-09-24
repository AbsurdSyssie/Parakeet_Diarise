#!/usr/bin/env python3
"""Standalone speaker diarization API using a configurable NeMo Sortformer model."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import time

import gc
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile

from diarization import DIARIZATION_MODEL, diarize_waveform, is_diarizer_loaded

app = FastAPI()


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
        "diarization_loaded": is_diarizer_loaded(),
        "diarization_model": DIARIZATION_MODEL,
        "cuda_available": torch.cuda.is_available(),
        "cuda_mem": cuda_mem(),
    }


@app.post("/v1/diarize")
async def diarize(file: UploadFile = File(...)):
    empty_cache = os.environ.get("DIARIZE_EMPTY_CACHE", "0") == "1"

    with tempfile.TemporaryDirectory() as tmpdir:
        start_req = time.perf_counter()
        tmp_path = Path(tmpdir) / file.filename
        tmp_path.write_bytes(await file.read())

        waveform, sr = torchaudio.load(tmp_path)

        print(f"cuda mem before diarize: {cuda_mem()}")
        turns = diarize_waveform(waveform, sr, Path(tmpdir))
        diarize_elapsed = time.perf_counter() - start_req
        print(f"diarize in {diarize_elapsed:.2f}s")
        print(f"cuda mem after diarize: {cuda_mem()}")

    if empty_cache:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"cuda mem after gc.collect: {cuda_mem()}")
    return turns
