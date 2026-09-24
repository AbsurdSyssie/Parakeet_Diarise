"""ASR backend aliases and model resolution."""

from __future__ import annotations


NEMO_MODEL_ALIASES = {
    "parakeet-0.6b": "nvidia/parakeet-tdt-0.6b-v3",
    "parakeet-1.1b": "nvidia/parakeet-tdt-1.1b",
    "nemotron-3.5-asr-streaming-0.6b": "nvidia/nemotron-3.5-asr-streaming-0.6b",
}

WHISPER_MODEL_ALIASES = {
    "medical-whisper-large-v3": "Na0s/Medical-Whisper-Large-v3",
    "whisper-medical-large-v3": "Na0s/Medical-Whisper-Large-v3",
}

FASTER_WHISPER_MODEL_ALIASES = {
    "faster-whisper-large-v3": "Systran/faster-whisper-large-v3",
}

TRANSFORMERS_ASR_MODEL_ALIASES = {
    "cohere-transcribe-03-2026": "CohereLabs/cohere-transcribe-03-2026",
    "granite-speech-4.1-2b-nar": "ibm-granite/granite-speech-4.1-2b-nar",
    "nemotron-3.5-asr-streaming-0.6b": "nvidia/nemotron-3.5-asr-streaming-0.6b",
}


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
