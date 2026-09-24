#!/usr/bin/env python3
"""Assign speaker labels to ASR words and group into segments."""

from __future__ import annotations

from typing import List, Dict, Any, Sequence

import torch


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


def assign_speakers_from_probabilities(
    words: List[Dict[str, Any]],
    probabilities: torch.Tensor | Sequence[Sequence[float]],
    *,
    audio_duration_s: float,
    frame_duration_s: float | None = None,
    num_speakers: int | None = None,
    confidence_threshold: float = 0.35,
    margin_threshold: float = 0.10,
    unknown_threshold: float = 0.15,
    neighbour_bonus: float = 0.15,
) -> List[Dict[str, Any]]:
    """Assign words by integrating Sortformer speaker posteriors over each word.

    The returned confidence is the winning mean posterior and margin is the
    difference from the runner-up. Low-confidence words may use adjacent,
    confidently-labelled words as a weak prior; very-low-confidence words stay
    UNKNOWN.
    """
    if not words:
        return []
    probs = torch.as_tensor(probabilities, dtype=torch.float32).detach().cpu()
    if probs.ndim == 3 and probs.shape[0] == 1:
        probs = probs[0]
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] == 0 or audio_duration_s <= 0:
        return [
            {**word, "speaker": "UNKNOWN", "speaker_confidence": 0.0, "speaker_margin": 0.0}
            for word in words
        ]

    if num_speakers is not None:
        if num_speakers < 1 or num_speakers > probs.shape[1]:
            raise ValueError(f"num_speakers must be between 1 and {probs.shape[1]}")
        active = torch.topk(probs.mean(dim=0), k=num_speakers).indices.sort().values
        probs = probs[:, active]
        speaker_indices = active.tolist()
    else:
        speaker_indices = list(range(probs.shape[1]))

    frame_s = frame_duration_s or (audio_duration_s / probs.shape[0])
    scored: List[Dict[str, Any]] = []
    for word in words:
        ws = max(0.0, float(word["start"]))
        we = min(audio_duration_s, float(word["end"]))
        if we <= ws:
            scores = torch.zeros(probs.shape[1])
        else:
            first = max(0, int(ws // frame_s))
            last = min(probs.shape[0] - 1, int(max(ws, we - 1e-9) // frame_s))
            weights = []
            frames = []
            for frame_idx in range(first, last + 1):
                overlap = max(0.0, min(we, (frame_idx + 1) * frame_s) - max(ws, frame_idx * frame_s))
                if overlap > 0:
                    weights.append(overlap)
                    frames.append(probs[frame_idx])
            scores = (
                torch.stack(frames).mul(torch.tensor(weights).unsqueeze(1)).sum(dim=0) / sum(weights)
                if frames
                else torch.zeros(probs.shape[1])
            )
        ranked = torch.argsort(scores, descending=True)
        best_local = int(ranked[0])
        confidence = float(scores[best_local])
        runner_up = float(scores[int(ranked[1])]) if len(ranked) > 1 else 0.0
        margin = confidence - runner_up
        confident = confidence >= confidence_threshold and margin >= margin_threshold
        scored.append(
            {
                **word,
                "speaker": f"SPEAKER_{speaker_indices[best_local]:02d}" if confident else "UNKNOWN",
                "speaker_confidence": round(confidence, 4),
                "speaker_margin": round(margin, 4),
                "_speaker_scores": scores,
            }
        )

    # Smooth only ambiguous words that still have meaningful speaker evidence.
    for idx, word in enumerate(scored):
        if word["speaker"] != "UNKNOWN" or word["speaker_confidence"] < unknown_threshold:
            continue
        adjusted = word["_speaker_scores"].clone()
        for neighbour_idx in (idx - 1, idx + 1):
            if 0 <= neighbour_idx < len(scored):
                label = scored[neighbour_idx]["speaker"]
                if label.startswith("SPEAKER_"):
                    channel = int(label.rsplit("_", 1)[1])
                    if channel in speaker_indices:
                        adjusted[speaker_indices.index(channel)] += neighbour_bonus
        ranked = torch.argsort(adjusted, descending=True)
        best_local = int(ranked[0])
        adjusted_margin = float(adjusted[best_local]) - (
            float(adjusted[int(ranked[1])]) if len(ranked) > 1 else 0.0
        )
        if adjusted_margin >= margin_threshold:
            word["speaker"] = f"SPEAKER_{speaker_indices[best_local]:02d}"

    for word in scored:
        word.pop("_speaker_scores", None)
    return scored


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
