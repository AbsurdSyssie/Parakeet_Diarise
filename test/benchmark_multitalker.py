#!/usr/bin/env python3
"""Benchmark Multitalker Parakeet + Sortformer without reviewing transcripts."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_multi import load_models, preprocess_audio, transcribe_audio


def run_one(path: Path, diar_model, asr_model, device: torch.device) -> dict:
    preprocess_started = time.perf_counter()
    processed_audio, is_temp = preprocess_audio(str(path))
    preprocess_s = time.perf_counter() - preprocess_started
    duration_s = sf.info(processed_audio).duration
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            segments, words = transcribe_audio(
                processed_audio, diar_model, asr_model, device, output_path=None
            )
        torch.cuda.synchronize()
        inference_s = time.perf_counter() - started
        speakers = sorted(
            {
                str(segment.get("speaker", "unknown"))
                for segment in segments
                if isinstance(segment, dict)
            }
        )
        return {
            "audio": str(path),
            "duration_s": round(duration_s, 3),
            "preprocess_s": round(preprocess_s, 3),
            "inference_s": round(inference_s, 3),
            "total_s": round(preprocess_s + inference_s, 3),
            "realtime_factor": round(inference_s / duration_s, 4),
            "audio_seconds_per_inference_second": round(duration_s / inference_s, 2),
            "peak_cuda_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "segments": len(segments),
            "words": len(words),
            "speakers": speakers,
        }
    finally:
        if is_temp:
            os.unlink(processed_audio)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        diar_model, asr_model = load_models(device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started

    if args.warmup:
        run_one(args.warmup, diar_model, asr_model, device)

    results = []
    for path in args.audio:
        try:
            result = run_one(path, diar_model, asr_model, device)
        except Exception as exc:
            result = {"audio": str(path), "error": f"{type(exc).__name__}: {exc}"}
        result["model_load_s"] = round(load_s, 3)
        results.append(result)
        print(json.dumps(result), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
