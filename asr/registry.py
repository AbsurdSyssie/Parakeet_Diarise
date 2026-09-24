"""Canonical ASR model registry and alias resolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str
    model_name: str
    description: str
    language: str = "en"
    task: str = "transcribe"
    trust_remote_code: bool = False
    return_timestamps: bool = False
    chunk_length_s: int | None = None
    stride_length_s: int | tuple[int, int] | None = None
    supports_word_timestamps: bool = True
    supports_segment_timestamps: bool = True
    supports_diarization: bool = True
    selectable: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "parakeet-0.6b": ModelSpec(
        id="parakeet-0.6b",
        backend="nemo",
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        description="NVIDIA Parakeet TDT 0.6B v3 via NeMo",
    ),
    "parakeet-1.1b": ModelSpec(
        id="parakeet-1.1b",
        backend="nemo",
        model_name="nvidia/parakeet-tdt-1.1b",
        description="NVIDIA Parakeet TDT 1.1B via NeMo",
    ),
    "nemotron-3.5-asr-streaming-0.6b": ModelSpec(
        id="nemotron-3.5-asr-streaming-0.6b",
        backend="nemo",
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        description="NVIDIA Nemotron 3.5 ASR Streaming 0.6B via NeMo",
    ),
    "medical-whisper-large-v3": ModelSpec(
        id="medical-whisper-large-v3",
        backend="whisper",
        model_name="Na0s/Medical-Whisper-Large-v3",
        description="Medical Whisper Large v3 via Transformers pipeline",
    ),
    "faster-whisper-large-v3": ModelSpec(
        id="faster-whisper-large-v3",
        backend="faster-whisper",
        model_name="Systran/faster-whisper-large-v3",
        description="SYSTRAN faster-whisper Large v3 via CTranslate2",
    ),
    "cohere-transcribe-03-2026": ModelSpec(
        id="cohere-transcribe-03-2026",
        backend="transformers-asr",
        model_name="CohereLabs/cohere-transcribe-03-2026",
        description="Cohere Transcribe via Transformers custom ASR backend",
        supports_word_timestamps=False,
        supports_segment_timestamps=False,
        supports_diarization=False,
    ),
    "granite-speech-4.1-2b-nar": ModelSpec(
        id="granite-speech-4.1-2b-nar",
        backend="transformers-asr",
        model_name="ibm-granite/granite-speech-4.1-2b-nar",
        description="IBM Granite Speech 4.1 2B NAR via Transformers custom model",
        trust_remote_code=True,
        supports_word_timestamps=False,
        supports_segment_timestamps=False,
        supports_diarization=False,
    ),
}


NEMO_MODEL_ALIASES = {
    spec.id: spec.model_name
    for spec in MODEL_REGISTRY.values()
    if spec.backend == "nemo"
}

WHISPER_MODEL_ALIASES = {
    spec.id: spec.model_name
    for spec in MODEL_REGISTRY.values()
    if spec.backend == "whisper"
}
WHISPER_MODEL_ALIASES["whisper-medical-large-v3"] = MODEL_REGISTRY[
    "medical-whisper-large-v3"
].model_name

FASTER_WHISPER_MODEL_ALIASES = {
    spec.id: spec.model_name
    for spec in MODEL_REGISTRY.values()
    if spec.backend == "faster-whisper"
}

TRANSFORMERS_ASR_MODEL_ALIASES = {
    spec.id: spec.model_name
    for spec in MODEL_REGISTRY.values()
    if spec.backend == "transformers-asr"
}
# Nemotron can also be loaded through the generic Transformers adapter.
TRANSFORMERS_ASR_MODEL_ALIASES["nemotron-3.5-asr-streaming-0.6b"] = MODEL_REGISTRY[
    "nemotron-3.5-asr-streaming-0.6b"
].model_name


def resolve_asr_model(backend: str, model_key: str) -> str:
    """Resolve a short alias or direct Hugging Face/NVIDIA model name."""
    normalized_backend = (backend or "nemo").strip().lower()
    normalized_key = (model_key or "").strip()

    if normalized_backend == "nemo":
        if not normalized_key:
            return NEMO_MODEL_ALIASES["parakeet-0.6b"]
        return NEMO_MODEL_ALIASES.get(normalized_key, normalized_key)

    if normalized_backend in {"whisper", "transformers-whisper"}:
        if not normalized_key:
            return WHISPER_MODEL_ALIASES["medical-whisper-large-v3"]
        return WHISPER_MODEL_ALIASES.get(normalized_key, normalized_key)

    if normalized_backend == "faster-whisper":
        if not normalized_key:
            return FASTER_WHISPER_MODEL_ALIASES["faster-whisper-large-v3"]
        return FASTER_WHISPER_MODEL_ALIASES.get(normalized_key, normalized_key)

    if normalized_backend in {"transformers-asr", "hf-asr"}:
        if not normalized_key:
            return TRANSFORMERS_ASR_MODEL_ALIASES["cohere-transcribe-03-2026"]
        return TRANSFORMERS_ASR_MODEL_ALIASES.get(normalized_key, normalized_key)

    raise ValueError(
        "ASR_BACKEND must be 'nemo', 'whisper', 'faster-whisper', or 'transformers-asr' "
        f"(got {backend!r})"
    )
