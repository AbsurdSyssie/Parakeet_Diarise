#!/usr/bin/env python3
"""
Run ASR API calls with different lever settings to debug boundary artifacts.

Examples:
  # Baseline with force_vad on/off (word timestamps only)
  .venv/bin/python test/asr_debug_levers.py --audio Examples/MoreOrLessFull.wav --force-vad both --timestamps word

  # Sweep a single VAD override
  .venv/bin/python test/asr_debug_levers.py --audio Examples/MoreOrLessFull.wav \\
    --force-vad both --timestamps word --sweep-key vad_uniform_chunk_s --sweep-values 45,30,20

  # Sweep two overrides (cartesian product)
  .venv/bin/python test/asr_debug_levers.py --audio Examples/MoreOrLessFull.wav \\
    --force-vad both --timestamps word --sweep-keys vad_uniform_chunk_s,vad_uniform_overlap_s \\
    --sweep-values 45,30,20 --sweep-values-2 0.5,0.2

Output analysis:
  - summary.json shows suspect_count, suspect_timestamps, and nearest chunk boundaries.
  - Use suspect_count as the primary objective.
  - If suspect_count ties, prefer smaller delta_s in suspect_nearest_boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests


DEFAULT_URL = "http://localhost:8000/v1/audio/transcriptions"
DEFAULT_OUT_DIR = "test/tmp/asr_debug_levers"


def parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def call_api(
    url: str,
    audio_path: Path,
    diarization: bool,
    timestamps: str,
    chunk_mode: str,
    chunk_only: bool,
    force_vad: str,
    trace_audio: bool,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "audio/wav")}
        data = {
            "response_format": "verbose_json",
            "diarization": str(diarization).lower(),
            "timestamps": timestamps,
            "chunk_mode": chunk_mode,
            "chunk_only": str(chunk_only).lower(),
            "force_vad": force_vad,
            "trace_audio": str(trace_audio).lower(),
        }
        if overrides:
            data.update(overrides)
        resp = requests.post(url, files=files, data=data, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _nearest_chunk_boundary(ts: float, chunks: List[Dict[str, Any]]):
    if not chunks:
        return None
    best = None
    for ch in chunks:
        for key in ("start_s", "end_s"):
            if key not in ch:
                continue
            b = float(ch[key])
            delta = abs(ts - b)
            if best is None or delta < best["delta_s"]:
                best = {
                    "boundary_s": b,
                    "delta_s": delta,
                    "chunk_index": ch.get("index"),
                }
    return best


def summarize(
    result: Dict[str, Any],
    suspect_words: List[str],
    chunk_info: Dict[str, Any] | None = None,
    suspect_boundary_s: float | None = None,
) -> Dict[str, Any]:
    words = result.get("words", []) or []
    segments = result.get("segments", []) or []
    speakers = result.get("speakers", [])
    unknown_words = sum(1 for w in words if w.get("speaker") == "UNKNOWN")
    last_words = [w.get("word") for w in words[-5:]]
    lowered = {s.lower() for s in suspect_words}
    hit = False
    hit_words = []
    hit_timestamps = []
    for w in words:
        token = str(w.get("word", "")).strip().lower().strip(".,?!")
        if token in lowered:
            hit = True
            hit_words.append(w.get("word"))
            hit_timestamps.append(
                {
                    "word": w.get("word"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                }
            )
    boundary_repeats = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) >= 2 and parts[-1].lower().rstrip(".,?!") == parts[-2].lower().rstrip(".,?!"):
            boundary_repeats += 1
    summary = {
        "words": len(words),
        "segments": len(segments),
        "speakers": speakers,
        "unknown_words": unknown_words,
        "last_words": last_words,
        "suspect_hit": hit,
        "suspect_count": len(hit_timestamps),
        "suspect_words": hit_words[:10],
        "suspect_timestamps": hit_timestamps,
        "segment_boundary_repeats": boundary_repeats,
    }
    if chunk_info:
        chunks = chunk_info.get("chunks", [])
        boundaries = []
        for hit in hit_timestamps:
            if hit.get("start") is None:
                continue
            nearest = _nearest_chunk_boundary(float(hit["start"]), chunks)
            if nearest:
                boundaries.append(
                    {
                        "word": hit.get("word"),
                        "start": hit.get("start"),
                        "nearest_boundary_s": nearest["boundary_s"],
                        "delta_s": nearest["delta_s"],
                        "chunk_index": nearest.get("chunk_index"),
                    }
                )
        summary["suspect_nearest_boundaries"] = boundaries
        summary["chunk_count"] = len(chunks)
        if suspect_boundary_s is not None:
            near = [b for b in boundaries if b.get("delta_s") is not None and b["delta_s"] <= suspect_boundary_s]
            summary["suspect_boundary_s"] = suspect_boundary_s
            summary["suspect_near_boundary_count"] = len(near)
    return summary


def build_cases(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    cases = []
    diarizations = [args.diarization] if args.diarization != "both" else [True, False]
    force_vads = [args.force_vad] if args.force_vad != "both" else ["on", "off"]
    timestamps = parse_csv(args.timestamps)
    chunk_modes = parse_csv(args.chunk_mode)
    for diar in diarizations:
        for fv in force_vads:
            for ts in timestamps:
                for cm in chunk_modes:
                    if diar and ts != "word":
                        continue
                    label = f"diar={diar}_ts={ts}_cm={cm}_fv={fv}"
                    cases.append(
                        (
                            label,
                            {
                                "diarization": diar,
                                "timestamps": ts,
                                "chunk_mode": cm,
                                "chunk_only": False,
                                "force_vad": fv,
                                "trace_audio": args.trace_audio,
                            },
                        )
                    )
    if args.chunk_only:
        cases.append(
            (
                "chunk_only",
                {
                    "diarization": False,
                    "timestamps": "none",
                    "chunk_mode": args.chunk_mode.split(",")[0],
                    "chunk_only": True,
                    "force_vad": args.force_vad if args.force_vad != "both" else "off",
                    "trace_audio": args.trace_audio,
                },
            )
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASR API with multiple lever settings.")
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument("--url", default=DEFAULT_URL, help="ASR API URL.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to write outputs.")
    parser.add_argument("--diarization", default="true", choices=["true", "false", "both"])
    parser.add_argument("--timestamps", default="word", help="CSV: word,segment,none")
    parser.add_argument("--chunk-mode", default="memory", help="CSV: memory,file")
    parser.add_argument("--force-vad", default="off", choices=["on", "off", "both"])
    parser.add_argument("--trace-audio", action="store_true")
    parser.add_argument("--chunk-only", action="store_true")
    parser.add_argument(
        "--suspect-words",
        default="Aaron",
        help="Comma-separated list of suspect words that indicate failure.",
    )
    parser.add_argument(
        "--suspect-boundary-s",
        type=float,
        default=2.0,
        help="Count suspect words within this many seconds of a chunk boundary (default: 2.0).",
    )
    parser.add_argument(
        "--sweep-keys",
        default="",
        help="Comma-separated request fields to sweep (e.g., vad_uniform_chunk_s,vad_uniform_overlap_s).",
    )
    parser.add_argument(
        "--sweep-values",
        default="",
        help="Comma-separated values to sweep for the first sweep key.",
    )
    parser.add_argument(
        "--sweep-values-2",
        default="",
        help="Comma-separated values to sweep for the second sweep key.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diarization = args.diarization == "true"
    args.diarization = diarization if args.diarization != "both" else "both"

    cases = build_cases(args)
    sweep_keys = parse_csv(args.sweep_keys)
    sweep_values_1 = parse_csv(args.sweep_values) if sweep_keys else [""]
    sweep_values_2 = parse_csv(args.sweep_values_2) if len(sweep_keys) > 1 else [""]
    chunk_case = {
        "diarization": False,
        "timestamps": "none",
        "chunk_mode": args.chunk_mode.split(",")[0],
        "chunk_only": True,
        "force_vad": args.force_vad if args.force_vad != "both" else "off",
        "trace_audio": args.trace_audio,
    }
    try:
        chunk_info = call_api(args.url, audio_path, **chunk_case)
    except requests.HTTPError as exc:
        chunk_info = None
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"Chunk-only request failed: HTTP {status}")
    summary = {}
    suspect_words = parse_csv(args.suspect_words)
    for label, params in cases:
        for v1 in sweep_values_1:
            for v2 in sweep_values_2:
                overrides = {}
                sweep_suffix = ""
                if sweep_keys:
                    if v1 != "":
                        overrides[sweep_keys[0]] = v1
                        sweep_suffix += f"_{sweep_keys[0]}={v1}"
                    if len(sweep_keys) > 1 and v2 != "":
                        overrides[sweep_keys[1]] = v2
                        sweep_suffix += f"_{sweep_keys[1]}={v2}"
                run_label = f"{label}{sweep_suffix}"
                print(f"Running {run_label}...")
                try:
                    result = call_api(args.url, audio_path, **params, overrides=overrides)
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else "?"
                    body = exc.response.text if exc.response is not None else ""
                    summary[run_label] = {"error": f"HTTP {status}", "body": body[:500]}
                    print(f"  -> HTTP {status}")
                    continue
                out_path = out_dir / f"{run_label}.json"
                out_path.write_text(json.dumps(result, indent=2))
                summary[run_label] = summarize(
                    result,
                    suspect_words,
                    chunk_info,
                    suspect_boundary_s=args.suspect_boundary_s if chunk_info else None,
                )
                print(f"  -> {out_path}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
