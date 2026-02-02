#!/usr/bin/env python3
"""POST an audio file to the ASR endpoint and return VAD chunk boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import httpx


def _resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix.lower() != ".wav":
        wav_fallback = path.with_suffix(".wav")
        if wav_fallback.exists():
            print(
                f"input not found: {path}; falling back to {wav_fallback}",
                file=sys.stderr,
            )
            return wav_fallback
    raise FileNotFoundError(f"Input file not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe ASR endpoint for VAD chunk boundaries (no ASR)."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1/audio/transcriptions",
        help="ASR endpoint URL (default: http://localhost:8000/v1/audio/transcriptions)",
    )
    parser.add_argument(
        "--input",
        default="Examples/MoreOrLessFull.mp3",
        help="Input audio file (default: Examples/MoreOrLessFull.mp3)",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=["memory", "file"],
        default="memory",
        help="Chunk mode to request (default: memory)",
    )
    parser.add_argument(
        "--out",
        default="test/tmp/asr_chunk_probe.json",
        help="Output JSON path (default: test/tmp/asr_chunk_probe.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = _resolve_input(Path(args.input))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    form = {
        "response_format": "verbose_json",
        "diarization": "false",
        "timestamps": "none",
        "language": "en",
        "chunk_mode": args.chunk_mode,
        "chunk_only": "true",
    }

    content_type = "audio/mpeg"
    if input_path.suffix.lower() == ".wav":
        content_type = "audio/wav"
    elif input_path.suffix.lower() == ".flac":
        content_type = "audio/flac"

    with input_path.open("rb") as fh:
        files = {"file": (input_path.name, fh, content_type)}
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(args.endpoint, data=form, files=files)

    if resp.status_code != 200:
        print(f"request failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        return 1

    payload = resp.json()
    chunks = payload.get("chunks", [])
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"saved chunk debug to {out_path}")
    print(f"chunk count: {len(chunks)}")
    if chunks:
        first = chunks[0]
        last = chunks[-1]
        print(
            "first chunk: "
            f"{first.get('start_s'):.2f}-{first.get('end_s'):.2f}s "
            f"(effective {first.get('effective_start_s'):.2f}-"
            f"{first.get('effective_end_s'):.2f}s)"
        )
        print(
            "last chunk: "
            f"{last.get('start_s'):.2f}-{last.get('end_s'):.2f}s "
            f"(effective {last.get('effective_start_s'):.2f}-"
            f"{last.get('effective_end_s'):.2f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
