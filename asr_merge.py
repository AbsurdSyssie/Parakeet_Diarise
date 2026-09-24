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


def _normalized_word(word: Dict[str, Any]) -> str:
    return re.sub(r"^\W+|\W+$", "", str(word.get("word", "")).casefold())


def _seam_overlap_size(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    seam_s: float,
    limit: int = 16,
) -> int:
    """Return a timestamp-supported textual overlap at a chunk seam."""
    max_size = min(len(left), len(right), limit)
    for size in range(max_size, 0, -1):
        left_words = left[-size:]
        right_words = right[:size]
        if [_normalized_word(word) for word in left_words] != [
            _normalized_word(word) for word in right_words
        ]:
            continue
        left_start = float(left_words[0]["start"])
        left_end = float(left_words[-1]["end"])
        right_start = float(right_words[0]["start"])
        right_end = float(right_words[-1]["end"])
        overlap_s = min(left_end, right_end) - max(left_start, right_start)
        # A single repeated word is only safe to remove when its timestamp
        # actually overlaps. Multi-word matches tolerate minor timestamp jitter.
        left_midpoint = (float(left_words[-1]["start"]) + left_end) / 2.0
        right_midpoint = (right_start + float(right_words[0]["end"])) / 2.0
        at_seam = abs(left_midpoint - seam_s) <= 0.40 and abs(right_midpoint - seam_s) <= 0.40
        if (
            overlap_s > 0
            or (size >= 2 and right_start - left_end <= 0.10)
            or (size == 1 and at_seam and right_start - left_end <= 0.20)
        ):
            return size
    return 0


def reconcile_chunk_words(
    chunk_words: List[List[Dict[str, Any]]],
    chunk_spans: List[tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Merge adjacent ASR chunks without repeating their shared context.

    Each chunk owns word midpoints between the neighbouring chunk seams. A
    timestamp-supported suffix/prefix match then removes residual duplicate
    decoding caused by timestamp jitter around a seam.
    """
    if len(chunk_words) != len(chunk_spans):
        raise ValueError("chunk_words and chunk_spans must have the same length")
    if not chunk_words:
        return []

    boundaries = [
        (float(chunk_spans[idx][1]) + float(chunk_spans[idx + 1][0])) / 2.0
        for idx in range(len(chunk_spans) - 1)
    ]
    owned: List[List[Dict[str, Any]]] = []
    for idx, words in enumerate(chunk_words):
        left_boundary = boundaries[idx - 1] if idx > 0 else float("-inf")
        right_boundary = boundaries[idx] if idx < len(boundaries) else float("inf")
        kept = []
        for word in sort_words(words):
            midpoint = (float(word["start"]) + float(word["end"])) / 2.0
            if midpoint > left_boundary and midpoint <= right_boundary:
                kept.append(word)
        owned.append(kept)

    for idx in range(len(owned) - 1):
        overlap_size = _seam_overlap_size(owned[idx], owned[idx + 1], boundaries[idx])
        if overlap_size:
            del owned[idx + 1][:overlap_size]

    return sort_words(word for words in owned for word in words)
