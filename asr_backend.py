#!/usr/bin/env python3
"""Compatibility facade for the ASR backend package.

New code should import from `asr`. This module keeps the original public
imports working for existing callers.
"""

from asr import (
    ASRConfig,
    FASTER_WHISPER_MODEL_ALIASES,
    NEMO_MODEL_ALIASES,
    TRANSFORMERS_ASR_MODEL_ALIASES,
    WHISPER_MODEL_ALIASES,
    FasterWhisperASRBackend,
    GraniteASRBackend,
    NemotronASRBackend,
    SimpleHypothesis,
    TransformersASRBackend,
    WhisperASRBackend,
    load_asr_backend,
    load_nemo_asr_model,
    resolve_asr_model,
)

__all__ = [
    "ASRConfig",
    "SimpleHypothesis",
    "NEMO_MODEL_ALIASES",
    "WHISPER_MODEL_ALIASES",
    "FASTER_WHISPER_MODEL_ALIASES",
    "TRANSFORMERS_ASR_MODEL_ALIASES",
    "resolve_asr_model",
    "WhisperASRBackend",
    "FasterWhisperASRBackend",
    "GraniteASRBackend",
    "NemotronASRBackend",
    "TransformersASRBackend",
    "load_nemo_asr_model",
    "load_asr_backend",
]
