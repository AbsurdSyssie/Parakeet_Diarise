"""ASR backend dispatch."""

from __future__ import annotations

from .faster_whisper import FasterWhisperASRBackend
from .granite import GraniteASRBackend
from .nemo import load_nemo_asr_model
from .nemotron import NemotronASRBackend
from .transformers import TransformersASRBackend
from .types import ASRConfig
from .whisper import WhisperASRBackend


def load_asr_backend(config: ASRConfig):
    backend = config.backend.strip().lower()
    if backend == "nemo":
        return load_nemo_asr_model(config)
    if backend in {"whisper", "transformers-whisper"}:
        return WhisperASRBackend(
            model_name=config.model_name,
            hf_token=config.hf_token,
            language=config.language,
            task=config.task,
        )
    if backend == "faster-whisper":
        return FasterWhisperASRBackend(
            model_name=config.model_name,
            hf_token=config.hf_token,
            language=config.language,
            task=config.task,
        )
    if backend in {"transformers-asr", "hf-asr"}:
        if config.model_name.lower() == "ibm-granite/granite-speech-4.1-2b-nar":
            return GraniteASRBackend(
                model_name=config.model_name,
                hf_token=config.hf_token,
                trust_remote_code=config.trust_remote_code,
                device=config.device,
                attention_implementation=config.attention_implementation,
            )
        if "nemotron-3.5-asr" in config.model_name.lower():
            return NemotronASRBackend(
                model_name=config.model_name,
                hf_token=config.hf_token,
                trust_remote_code=config.trust_remote_code,
                return_timestamps=config.return_timestamps,
                chunk_length_s=config.chunk_length_s,
                stride_length_s=config.stride_length_s,
                language=config.language,
            )
        return TransformersASRBackend(
            model_name=config.model_name,
            hf_token=config.hf_token,
            trust_remote_code=config.trust_remote_code,
            return_timestamps=config.return_timestamps,
            chunk_length_s=config.chunk_length_s,
            stride_length_s=config.stride_length_s,
            language=config.language,
        )
    raise ValueError(
        "ASR_BACKEND must be 'nemo', 'whisper', 'faster-whisper', or 'transformers-asr' "
        f"(got {config.backend!r})"
    )
