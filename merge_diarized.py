#!/usr/bin/env python3
"""Assign speakers to merged ASR words and output diarized segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diarize_align import assign_speakers, group_words_into_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ASR words with diarization turns into speaker segments."
    )
    parser.add_argument(
        "--words-json",
        default="Output/merged_words.json",
        help="Merged words JSON (default: Output/merged_words.json)",
    )
    parser.add_argument(
        "--turns-json",
        default="Output/diarization_turns.json",
        help="Diarization turns JSON (default: Output/diarization_turns.json)",
    )
    parser.add_argument(
        "--out-json",
        default="Output/diarized_segments.json",
        help="Output diarized segments JSON (default: Output/diarized_segments.json)",
    )
    parser.add_argument(
        "--min-overlap-s",
        type=float,
        default=0.03,
        help="Minimum overlap to assign speaker (default: 0.03)",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=0.6,
        help="Max gap between words to keep segment (default: 0.6)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    words = json.loads(Path(args.words_json).read_text())
    turns = json.loads(Path(args.turns_json).read_text())

    words_with_speaker = assign_speakers(words, turns, min_overlap_s=args.min_overlap_s)
    segments = group_words_into_segments(words_with_speaker, max_gap_s=args.max_gap_s)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(segments, indent=2))

    print(f"Wrote {len(segments)} segments to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
