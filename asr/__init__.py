"""ASR backends and model resolution."""

from .faster_whisper import FasterWhisperASRBackend
from .granite import GraniteASRBackend
from .loader import load_asr_backend
from .nemo import load_nemo_asr_model
from .nemotron import NemotronASRBackend
from .registry import (
    FASTER_WHISPER_MODEL_ALIASES,
    MODEL_REGISTRY,
    ModelSpec,
    NEMO_MODEL_ALIASES,
    TRANSFORMERS_ASR_MODEL_ALIASES,
    WHISPER_MODEL_ALIASES,
    resolve_asr_model,
)
from .transformers import TransformersASRBackend
from .types import ASRConfig, SimpleHypothesis
from .whisper import WhisperASRBackend

__all__ = [
    "ASRConfig",
    "SimpleHypothesis",
    "ModelSpec",
    "MODEL_REGISTRY",
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
