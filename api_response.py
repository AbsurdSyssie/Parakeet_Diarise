#!/usr/bin/env python3
"""API response helpers for transcription outputs."""

from __future__ import annotations

from typing import List, Dict, Any, Optional


def build_response(
    words: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    language: str,
    duration: float,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    if text is None:
        text = " ".join([w["word"] for w in words])
    speakers = sorted({w.get("speaker") for w in words if w.get("speaker")})
    return {
        "text": text,
        "language": language,
        "duration": duration,
        "words": words,
        "segments": segments,
        "speakers": speakers,
    }
