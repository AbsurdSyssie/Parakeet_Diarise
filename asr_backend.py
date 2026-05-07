#!/usr/bin/env python3
"""ASR backend loading and compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import nemo.collections.asr as nemo_asr


NEMO_MODEL_ALIASES = {
    "parakeet-0.6b": "nvidia/parakeet-tdt-0.6b-v3",
    "parakeet-1.1b": "nvidia/parakeet-tdt-1.1b",
}

WHISPER_MODEL_ALIASES = {
    "medical-whisper-large-v3": "Na0s/Medical-Whisper-Large-v3",
    "whisper-medical-large-v3": "Na0s/Medical-Whisper-Large-v3",
}


@dataclass
class ASRConfig:
    backend: str
    model_key: str
    model_name: str
    hf_token: str | None = None
    language: str = "en"
    task: str = "transcribe"


@dataclass
class SimpleHypothesis:
    text: str
    timestamp: dict[str, list[dict[str, Any]]]


def resolve_asr_model(backend: str, model_key: str) -> str:
    """Resolve a short alias or direct Hugging Face/NVIDIA model name."""
    normalized_backend = (backend or "nemo").strip().lower()
    normalized_key = (model_key or "").strip()

    if normalized_backend == "nemo":
        if not normalized_key:
            return NEMO_MODEL_ALIASES["parakeet-0.6b"]
        return NEMO_MODEL_ALIASES.get(normalized_key, normalized_key)

    if normalized_backend in {"whisper", "transformers", "transformers-whisper"}:
        if not normalized_key:
            return WHISPER_MODEL_ALIASES["medical-whisper-large-v3"]
        return WHISPER_MODEL_ALIASES.get(normalized_key, normalized_key)

    raise ValueError(
        "ASR_BACKEND must be 'nemo' or 'whisper' "
        f"(got {backend!r})"
    )


class WhisperASRBackend:
    """Transformers Whisper adapter with a NeMo-like transcribe interface."""

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        language: str = "en",
        task: str = "transcribe",
    ) -> None:
        from transformers import pipeline

        self.model_name = model_name
        self.language = language
        self.task = task
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipe_kwargs: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
        }
        if hf_token:
            pipe_kwargs["token"] = hf_token

        self.pipe = pipeline(**pipe_kwargs)

    def to(self, device: str | torch.device):
        # Transformers pipeline placement is configured at load time.
        return self

    def transcribe(
        self,
        inputs: Iterable[Any],
        timestamps: bool = False,
        verbose: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        return_hypotheses: bool = True,
        **_: Any,
    ) -> list[SimpleHypothesis] | list[str]:
        prepared = [self._prepare_input(item) for item in inputs]
        return_timestamps: bool | str = "word" if timestamps else False
        generate_kwargs = {"task": self.task}
        if self.language:
            generate_kwargs["language"] = self.language
        if timestamps:
            # Force WhisperTimeStampLogitsProcessor in generate(). The pipeline's
            # return_timestamps="word" should also do this, but passing it here
            # makes the generation config explicit for Whisper backends.
            generate_kwargs["return_timestamps"] = True
            generate_kwargs["return_token_timestamps"] = True

        pipe_call_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "return_timestamps": return_timestamps,
            "generate_kwargs": generate_kwargs,
        }
        if timestamps:
            # Keep chunks under Whisper's 30s timestamp window and add a small
            # right stride so cut-off words near chunk edges still get endings.
            pipe_call_kwargs["chunk_length_s"] = 30
            pipe_call_kwargs["stride_length_s"] = (0, 2)

        outputs = self.pipe(prepared, **pipe_call_kwargs)
        if isinstance(outputs, dict):
            outputs = [outputs]

        hypotheses = [self._to_hypothesis(output, timestamps=timestamps) for output in outputs]
        if return_hypotheses:
            return hypotheses
        return [hyp.text for hyp in hypotheses]

    def _prepare_input(self, item: Any) -> Any:
        if isinstance(item, (str, Path)):
            return str(item)

        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.dim() == 2 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            return {"array": tensor.float().numpy(), "sampling_rate": 16000}

        return item

    def _to_hypothesis(self, output: dict[str, Any], timestamps: bool) -> SimpleHypothesis:
        text = str(output.get("text") or "").strip()
        if not timestamps:
            return SimpleHypothesis(text=text, timestamp={"word": [], "segment": []})

        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        for chunk in output.get("chunks") or []:
            chunk_text = str(chunk.get("text") or "").strip()
            ts = chunk.get("timestamp") or (None, None)
            if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue
            start, end = ts[0], ts[1]
            if start is None:
                continue
            try:
                start_f = float(start)
            except (TypeError, ValueError):
                continue
            try:
                end_f = float(end) if end is not None else start_f + 0.01
            except (TypeError, ValueError):
                end_f = start_f + 0.01
            if end_f <= start_f:
                end_f = start_f + 0.01
            if not chunk_text:
                continue

            words.append({"word": chunk_text, "start": start_f, "end": end_f})
            segments.append({"segment": chunk_text, "start": start_f, "end": end_f})

        return SimpleHypothesis(text=text, timestamp={"word": words, "segment": segments})


def load_nemo_asr_model(config: ASRConfig):
    kwargs = {"model_name": config.model_name}
    if config.hf_token:
        kwargs["use_auth_token"] = config.hf_token

    try:
        return nemo_asr.models.ASRModel.from_pretrained(**kwargs)
    except TypeError:
        kwargs.pop("use_auth_token", None)
        return nemo_asr.models.ASRModel.from_pretrained(**kwargs)


def load_asr_backend(config: ASRConfig):
    backend = config.backend.strip().lower()
    if backend == "nemo":
        return load_nemo_asr_model(config)
    if backend in {"whisper", "transformers", "transformers-whisper"}:
        return WhisperASRBackend(
            model_name=config.model_name,
            hf_token=config.hf_token,
            language=config.language,
            task=config.task,
        )
    raise ValueError(
        "ASR_BACKEND must be 'nemo' or 'whisper' "
        f"(got {config.backend!r})"
    )
