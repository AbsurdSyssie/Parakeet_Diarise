#!/usr/bin/env python3
"""Render verbose transcription JSON as readable speaker-labelled text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def timestamp(seconds: float) -> str:
    total_ms = round(float(seconds) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    lines = [
        f"Duration: {timestamp(data.get('duration', 0))}",
        f"Speakers: {', '.join(data.get('speakers', [])) or 'none'}",
        "",
    ]
    for segment in data.get("segments", []):
        lines.append(
            f"[{timestamp(segment['start'])} - {timestamp(segment['end'])}] "
            f"{segment.get('speaker', 'UNKNOWN')}: {str(segment.get('text', '')).strip()}"
        )
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
