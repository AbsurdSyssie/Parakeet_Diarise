#!/usr/bin/env python3
"""Benchmark Sortformer backends and post-processing on one local recording."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import tempfile
import time
from pathlib import Path

import torch
import torchaudio
from nemo.collections.asr.models import SortformerEncLabelModel


MODELS = {
    "streaming": "nvidia/diar_streaming_sortformer_4spk-v2.1",
    "offline": "nvidia/diar_sortformer_4spk-v1",
}


def prepare_audio(path: Path, out_path: Path) -> float:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        sample_rate = 16_000
    torchaudio.save(str(out_path), waveform, sample_rate)
    return waveform.shape[-1] / sample_rate


def configure(model, backend: str) -> None:
    model.eval()
    if backend == "streaming":
        model.sortformer_modules.chunk_len = 340
        model.sortformer_modules.chunk_right_context = 40
        model.sortformer_modules.fifo_len = 40
        model.sortformer_modules.spkcache_update_period = 300


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--backend", action="append", choices=MODELS, dest="backends")
    parser.add_argument("--postprocessing", type=Path, default=Path("configs/sortformer_postprocessing.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    backends = args.backends or list(MODELS)
    temp_audio = tempfile.TemporaryDirectory(prefix="sortformer-benchmark-")
    model_audio = Path(temp_audio.name) / "mono16k.wav"
    duration_s = prepare_audio(args.audio, model_audio)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    results = []

    for backend in backends:
        started = time.perf_counter()
        model = SortformerEncLabelModel.from_pretrained(MODELS[backend], map_location="cpu")
        configure(model, backend)
        model.to(torch.device(args.device))
        load_s = time.perf_counter() - started
        for postprocessing in (False, True):
            kwargs = {
                "audio": [str(model_audio)],
                "batch_size": 1,
                "include_tensor_outputs": True,
                "verbose": False,
            }
            if postprocessing:
                kwargs["postprocessing_yaml"] = str(args.postprocessing.resolve())
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            segments, probabilities = model.diarize(**kwargs)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - started
            probs = torch.as_tensor(probabilities[0]).detach().cpu()
            if probs.ndim == 3 and probs.shape[0] == 1:
                probs = probs[0]
            item = {
                "audio": str(args.audio),
                "duration_s": round(duration_s, 3),
                "backend": backend,
                "model": MODELS[backend],
                "postprocessing": postprocessing,
                "device": args.device,
                "load_s": round(load_s, 3),
                "inference_s": round(elapsed_s, 3),
                "realtime_factor": round(elapsed_s / duration_s, 3),
                "segments": len(segments[0]),
                "probability_shape": list(probs.shape),
                "mean_activity": [round(float(value), 4) for value in probs.mean(dim=0)],
                "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "peak_cuda_mb": (
                    round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                    if args.device == "cuda"
                    else None
                ),
            }
            results.append(item)
            print(json.dumps(item), flush=True)
        del model
        gc.collect()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    temp_audio.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
