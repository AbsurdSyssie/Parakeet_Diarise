#!/usr/bin/env python3
"""Run pyannote diarization on a full file and emit speaker turns JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.serialization
import torchaudio
from pyannote.audio import Pipeline

import inspect
import torch.serialization
from pyannote.audio.core import task as py_task

SAFE = [obj for _, obj in inspect.getmembers(py_task, inspect.isclass)]
torch.serialization.add_safe_globals(SAFE)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pyannote diarization and save speaker turns as JSON."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="Examples/MoreOrLessFull.wav",
        help="Path to input WAV (default: Examples/MoreOrLessFull.wav)",
    )
    parser.add_argument(
        "--out-json",
        default="Output/diarization_turns.json",
        help="Output JSON for speaker turns (default: Output/diarization_turns.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.audio_path):
        raise SystemExit(f"Audio file not found: {args.audio_path}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN env var is required for pyannote model access.")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    waveform, sr = torchaudio.load(args.audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    diarization_out = pipeline({"waveform": waveform, "sample_rate": sr})

    def extract_annotation(out):
        candidates = [
            "diarization",
            "annotation",
            "segments",
            "turns",
            "predicted_diarization",
        ]
        for name in candidates:
            if hasattr(out, name):
                val = getattr(out, name)
                if val is not None:
                    return val

        if hasattr(out, "get"):
            for name in candidates:
                val = out.get(name)
                if val is not None:
                    return val

        if hasattr(out, "__dict__"):
            for v in out.__dict__.values():
                if hasattr(v, "itertracks"):
                    return v

        if hasattr(out, "values"):
            try:
                for v in out.values():
                    if hasattr(v, "itertracks"):
                        return v
            except TypeError:
                pass

        return None

    annotation = extract_annotation(diarization_out)
    if annotation is None:
        raise SystemExit(f"Unexpected diarization output format: {type(diarization_out)}")

    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )

    turns.sort(key=lambda t: (t["start"], t["end"]))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(turns, indent=2))

    print(f"Wrote {len(turns)} turns to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
