#!/usr/bin/env python3
"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()


def env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


@dataclass(frozen=True)
class AppSettings:
    api_port: int
    hf_token: str | None
    asr_backend: str
    asr_model: str
    asr_language: str
    asr_task: str
    asr_trust_remote_code: bool
    asr_return_timestamps: bool
    asr_chunk_length_s: int | None
    asr_stride_length_s: int | None
    asr_attention_implementation: str | None
    diarization_model: str
    vad_sample_rate: int
    disable_cuda_graphs: bool

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            api_port=env_int("API_PORT", "8000"),
            hf_token=(
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_TOKEN")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
            ),
            asr_backend=os.getenv("ASR_BACKEND", "nemo").strip().lower(),
            asr_model=os.getenv("ASR_MODEL", "").strip(),
            asr_language=os.getenv("ASR_LANGUAGE", "en").strip(),
            asr_task=os.getenv("ASR_TASK", "transcribe").strip(),
            asr_trust_remote_code=env_bool("ASR_TRUST_REMOTE_CODE", "0"),
            asr_return_timestamps=env_bool("ASR_RETURN_TIMESTAMPS", "0"),
            asr_chunk_length_s=optional_env_int("ASR_CHUNK_LENGTH_S"),
            asr_stride_length_s=optional_env_int("ASR_STRIDE_LENGTH_S"),
            asr_attention_implementation=(
                os.getenv("ASR_ATTENTION_IMPLEMENTATION", "").strip() or None
            ),
            diarization_model=(
                os.getenv("DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization").strip()
                or "nvidia/Nemotron-3-Diarization"
            ),
            vad_sample_rate=env_int("VAD_SAMPLE_RATE", "16000"),
            disable_cuda_graphs=env_bool("DISABLE_CUDA_GRAPHS", "0"),
        )


SETTINGS = AppSettings.from_env()
