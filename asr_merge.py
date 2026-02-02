#!/usr/bin/env python3
"""Helpers for merging chunked ASR timestamps into a global timeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

_CHUNK_RE = re.compile(r"^chunk_(\d+)_([0-9.]+)-([0-9.]+)\.wav$")


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: int
    orig_start_s: float
    orig_end_s: float
    pad_left_s: float
    pad_right_s: float

    @property
    def effective_start_s(self) -> float:
        return max(0.0, self.orig_start_s - self.pad_left_s)

    @property
    def effective_end_s(self) -> float:
        return self.orig_end_s + self.pad_right_s


def parse_chunk_filename(filename: str) -> tuple[int, float, float]:
    """Parse chunk filename format: chunk_0001_0.28-3.75.wav."""
    match = _CHUNK_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized chunk filename: {filename}")
    chunk_id = int(match.group(1))
    start_s = float(match.group(2))
    end_s = float(match.group(3))
    return chunk_id, start_s, end_s


def build_chunk_meta(filename: str, pad_left_s: float, pad_right_s: float) -> ChunkMeta:
    chunk_id, start_s, end_s = parse_chunk_filename(filename)
    return ChunkMeta(
        chunk_id=chunk_id,
        orig_start_s=start_s,
        orig_end_s=end_s,
        pad_left_s=pad_left_s,
        pad_right_s=pad_right_s,
    )


def offset_words(words: Iterable[Dict[str, Any]], offset_s: float) -> List[Dict[str, Any]]:
    """Offset word timestamps by a fixed amount in seconds."""
    out = []
    for word in words:
        shifted = dict(word)
        shifted["start"] = float(word["start"]) + offset_s
        shifted["end"] = float(word["end"]) + offset_s
        out.append(shifted)
    return out


def dedup_overlaps(
    words: Iterable[Dict[str, Any]],
    tolerance_s: float = 0.02,
) -> List[Dict[str, Any]]:
    """Drop overlapping words with a simple last-end threshold."""
    out: List[Dict[str, Any]] = []
    last_end = -1.0
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        if end <= last_end + tolerance_s:
            continue
        out.append(word)
        last_end = end
    return out


def sort_words(words: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort words by start then end time."""
    return sorted(words, key=lambda w: (float(w["start"]), float(w["end"])))
