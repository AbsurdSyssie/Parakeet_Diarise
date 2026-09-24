#!/usr/bin/env python3
"""Curated ASR model registry and capability rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from asr import ASRConfig, resolve_asr_model
from settings import AppSettings, SETTINGS


@dataclass(frozen=True)
class ModelRegistryEntry:
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


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "parakeet-0.6b": ModelRegistryEntry(
        id="parakeet-0.6b",
        backend="nemo",
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        description="NVIDIA Parakeet TDT 0.6B v3 via NeMo",
    ),
    "parakeet-1.1b": ModelRegistryEntry(
        id="parakeet-1.1b",
        backend="nemo",
        model_name="nvidia/parakeet-tdt-1.1b",
        description="NVIDIA Parakeet TDT 1.1B via NeMo",
    ),
    "nemotron-3.5-asr-streaming-0.6b": ModelRegistryEntry(
        id="nemotron-3.5-asr-streaming-0.6b",
        backend="nemo",
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        description="NVIDIA Nemotron 3.5 ASR Streaming 0.6B via NeMo",
    ),
    "medical-whisper-large-v3": ModelRegistryEntry(
        id="medical-whisper-large-v3",
        backend="whisper",
        model_name="Na0s/Medical-Whisper-Large-v3",
        description="Medical Whisper Large v3 via Transformers pipeline",
    ),
    "faster-whisper-large-v3": ModelRegistryEntry(
        id="faster-whisper-large-v3",
        backend="faster-whisper",
        model_name="Systran/faster-whisper-large-v3",
        description="SYSTRAN faster-whisper Large v3 via CTranslate2",
    ),
    "cohere-transcribe-03-2026": ModelRegistryEntry(
        id="cohere-transcribe-03-2026",
        backend="transformers-asr",
        model_name="CohereLabs/cohere-transcribe-03-2026",
        description="Cohere Transcribe via Transformers custom ASR backend",
        trust_remote_code=SETTINGS.asr_trust_remote_code,
        supports_word_timestamps=False,
        supports_segment_timestamps=False,
        supports_diarization=False,
    ),
    "granite-speech-4.1-2b-nar": ModelRegistryEntry(
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


def default_model_key_for_backend(backend: str) -> str:
    normalized = (backend or "nemo").strip().lower()
    if normalized == "nemo":
        return "parakeet-0.6b"
    if normalized in {"whisper", "transformers-whisper"}:
        return "medical-whisper-large-v3"
    if normalized == "faster-whisper":
        return "faster-whisper-large-v3"
    if normalized in {"transformers-asr", "hf-asr"}:
        return "cohere-transcribe-03-2026"
    return "parakeet-0.6b"


def startup_model_key(settings: AppSettings = SETTINGS) -> str:
    return settings.asr_model or default_model_key_for_backend(settings.asr_backend)


def startup_model_name(settings: AppSettings = SETTINGS) -> str:
    return resolve_asr_model(settings.asr_backend, startup_model_key(settings))


def find_model_entry(
    model_id: str,
    backend: str | None = None,
) -> ModelRegistryEntry | None:
    normalized = (model_id or "").strip()
    normalized_backend = (backend or "").strip().lower()
    for entry in MODEL_REGISTRY.values():
        if normalized not in {entry.id, entry.model_name}:
            continue
        if normalized_backend and normalized_backend != entry.backend:
            continue
        return entry
    return None


def model_entry_for_config(config: ASRConfig | None) -> ModelRegistryEntry | None:
    if config is None:
        return None
    return find_model_entry(config.model_key, config.backend) or find_model_entry(
        config.model_name,
        config.backend,
    )


def public_model_entry(entry: ModelRegistryEntry) -> dict[str, Any]:
    data = asdict(entry)
    data["object"] = "model"
    data["owned_by"] = entry.backend
    return data


def public_config(config: ASRConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    entry = model_entry_for_config(config)
    return {
        "backend": config.backend,
        "model_key": config.model_key,
        "model_name": config.model_name,
        "language": config.language,
        "task": config.task,
        "trust_remote_code": config.trust_remote_code,
        "return_timestamps": config.return_timestamps,
        "chunk_length_s": config.chunk_length_s,
        "stride_length_s": config.stride_length_s,
        "attention_implementation": config.attention_implementation,
        "supports_word_timestamps": bool(entry.supports_word_timestamps) if entry else None,
        "supports_segment_timestamps": bool(entry.supports_segment_timestamps) if entry else None,
        "supports_diarization": bool(entry.supports_diarization) if entry else None,
    }


def config_from_entry(
    entry: ModelRegistryEntry,
    *,
    settings: AppSettings = SETTINGS,
    language: str | None = None,
    return_timestamps: bool | None = None,
) -> ASRConfig:
    requested_timestamps = (
        entry.return_timestamps if return_timestamps is None else bool(return_timestamps)
    )
    if (
        requested_timestamps
        and entry.backend not in {"transformers-asr", "hf-asr"}
        and not (entry.supports_word_timestamps or entry.supports_segment_timestamps)
    ):
        requested_timestamps = False

    return ASRConfig(
        backend=entry.backend,
        model_key=entry.id,
        model_name=entry.model_name,
        hf_token=settings.hf_token,
        language=(language or entry.language or settings.asr_language),
        task=entry.task or settings.asr_task,
        trust_remote_code=entry.trust_remote_code,
        return_timestamps=requested_timestamps,
        chunk_length_s=entry.chunk_length_s,
        stride_length_s=entry.stride_length_s,
        attention_implementation=settings.asr_attention_implementation,
    )


def initial_config(settings: AppSettings = SETTINGS) -> ASRConfig:
    model_key = startup_model_key(settings)
    model_name = startup_model_name(settings)
    entry = find_model_entry(model_key, settings.asr_backend) or find_model_entry(
        model_name,
        settings.asr_backend,
    )

    if entry is None:
        if settings.asr_backend in {"transformers-asr", "hf-asr"}:
            return ASRConfig(
                backend=settings.asr_backend,
                model_key=model_key,
                model_name=model_name,
                hf_token=settings.hf_token,
                language=settings.asr_language,
                task=settings.asr_task,
                trust_remote_code=settings.asr_trust_remote_code,
                return_timestamps=settings.asr_return_timestamps,
                chunk_length_s=settings.asr_chunk_length_s,
                stride_length_s=settings.asr_stride_length_s,
                attention_implementation=settings.asr_attention_implementation,
            )
        raise RuntimeError(
            "ASR_MODEL must be one of the curated /v1/models entries "
            f"(got backend={settings.asr_backend!r}, model={model_key!r})"
        )

    config = config_from_entry(
        entry,
        settings=settings,
        language=settings.asr_language,
        return_timestamps=(
            settings.asr_return_timestamps if settings.asr_return_timestamps else None
        ),
    )
    if config.backend in {"transformers-asr", "hf-asr"}:
        config.trust_remote_code = (
            config.trust_remote_code or settings.asr_trust_remote_code
        )
        config.return_timestamps = settings.asr_return_timestamps
        config.chunk_length_s = settings.asr_chunk_length_s
        config.stride_length_s = settings.asr_stride_length_s
        config.attention_implementation = settings.asr_attention_implementation
    return config


def effective_timestamps(config: ASRConfig, requested: str) -> str:
    if requested == "none":
        return "none"
    if config.backend in {"transformers-asr", "hf-asr"}:
        return requested if config.return_timestamps else "none"

    entry = model_entry_for_config(config)
    if entry is None:
        return requested

    if requested == "word":
        if entry.supports_word_timestamps and config.return_timestamps:
            return "word"
        if entry.supports_segment_timestamps and config.return_timestamps:
            return "segment"
        return "none"

    if requested == "segment":
        if entry.supports_segment_timestamps and config.return_timestamps:
            return "segment"
        return "none"

    return requested


def validate_request(
    config: ASRConfig,
    *,
    diarization: bool,
    timestamps: str,
) -> None:
    if diarization and timestamps != "word":
        raise ValueError("diarization requires timestamps=word")
    if not diarization:
        return
    if config.backend in {"transformers-asr", "hf-asr"} and not config.return_timestamps:
        raise ValueError(
            "diarization requires ASR_RETURN_TIMESTAMPS=1 for transformers-asr"
        )

    entry = model_entry_for_config(config)
    if entry is not None and not entry.supports_diarization:
        raise ValueError(f"diarization is not supported by active model {entry.id!r}")

    if effective_timestamps(config, timestamps) != "word":
        raise ValueError(
            "diarization requires an active model with word timestamps enabled"
        )
