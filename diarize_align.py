#!/usr/bin/env python3
"""Assign speaker labels to ASR words and group into segments."""

from __future__ import annotations

from typing import List, Dict, Any


def assign_speakers(
    words: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    min_overlap_s: float = 0.01,
    max_gap_s: float | None = None,
) -> List[Dict[str, Any]]:
    """Assign speaker to each word by max overlap with diarization turns."""
    if not words or not turns:
        return [{**w, "speaker": "UNKNOWN"} for w in words]

    words_out = []
    turn_idx = 0
    for word in words:
        ws = float(word["start"])
        we = float(word["end"])

        while turn_idx < len(turns) and float(turns[turn_idx]["end"]) < ws:
            turn_idx += 1

        best_overlap = 0.0
        best_speaker = "UNKNOWN"

        check_idx = turn_idx
        while check_idx < len(turns):
            ts = float(turns[check_idx]["start"])
            te = float(turns[check_idx]["end"])
            if ts > we:
                break
            overlap = max(0.0, min(we, te) - max(ws, ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turns[check_idx]["speaker"]
            check_idx += 1

        if best_overlap < min_overlap_s:
            best_speaker = "UNKNOWN"
            if max_gap_s is not None and turns:
                prev_turn = turns[turn_idx - 1] if turn_idx > 0 else None
                next_turn = turns[turn_idx] if turn_idx < len(turns) else None
                candidates = [t for t in (prev_turn, next_turn) if t is not None]
                best_gap = None
                for t in candidates:
                    ts = float(t["start"])
                    te = float(t["end"])
                    if we < ts:
                        gap = ts - we
                    elif ws > te:
                        gap = ws - te
                    else:
                        gap = 0.0
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_speaker = t["speaker"]
                if best_gap is None or best_gap > max_gap_s:
                    best_speaker = "UNKNOWN"

        words_out.append({**word, "speaker": best_speaker})

    return words_out


def group_words_into_segments(
    words: List[Dict[str, Any]],
    max_gap_s: float = 0.8,
    unknown_merge_max_s: float = 0.6,
    unknown_merge_max_words: int = 2,
) -> List[Dict[str, Any]]:
    """Group words into speaker-homogeneous segments."""
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    current = {
        "speaker": words[0].get("speaker", "UNKNOWN"),
        "start": float(words[0]["start"]),
        "end": float(words[0]["end"]),
        "text": words[0]["word"],
    }

    for word in words[1:]:
        speaker = word.get("speaker", "UNKNOWN")
        start = float(word["start"])
        end = float(word["end"])
        gap = start - float(current["end"])

        if speaker == current["speaker"] and gap <= max_gap_s:
            current["end"] = end
            current["text"] += " " + word["word"]
        else:
            segments.append(current)
            current = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": word["word"],
            }

    segments.append(current)
    if not segments or unknown_merge_max_s <= 0:
        return segments

    def should_merge_unknown(seg: Dict[str, Any]) -> bool:
        duration = float(seg["end"]) - float(seg["start"])
        word_count = len(str(seg.get("text", "")).split())
        return duration <= unknown_merge_max_s or word_count <= unknown_merge_max_words

    merged: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(segments):
        seg = segments[idx]
        if seg.get("speaker") == "UNKNOWN" and should_merge_unknown(seg):
            prev = merged[-1] if merged else None
            next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
            if prev and next_seg and next_seg.get("speaker") == prev.get("speaker"):
                prev["end"] = next_seg["end"]
                prev["text"] += " " + seg["text"] + " " + next_seg["text"]
                idx += 2
                continue
            if prev and next_seg:
                gap_prev = float(seg["start"]) - float(prev["end"])
                gap_next = float(next_seg["start"]) - float(seg["end"])
                if gap_next < gap_prev:
                    next_seg["start"] = seg["start"]
                    next_seg["text"] = seg["text"] + " " + next_seg["text"]
                    idx += 1
                    continue
                prev["end"] = seg["end"]
                prev["text"] += " " + seg["text"]
                idx += 1
                continue
            if prev:
                prev["end"] = seg["end"]
                prev["text"] += " " + seg["text"]
                idx += 1
                continue
            if next_seg:
                next_seg["start"] = seg["start"]
                next_seg["text"] = seg["text"] + " " + next_seg["text"]
                idx += 1
                continue
        merged.append(seg)
        idx += 1

    return merged
