"""Generic Hugging Face Transformers ASR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from .types import SimpleHypothesis


class TransformersASRBackend:
    """Generic Hugging Face Transformers ASR adapter."""

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        trust_remote_code: bool = False,
        return_timestamps: bool = False,
        chunk_length_s: int | None = None,
        stride_length_s: int | tuple[int, int] | None = None,
        language: str = "en",
    ) -> None:
        self.model_name = model_name
        self.return_timestamps = return_timestamps
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.language = language
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Generic/custom ASR models are not guaranteed to be fp16-safe. Cohere's
        # remote code masks attention with -1e9, which overflows float16.
        self.torch_dtype = torch.float32
        self._is_cohere = model_name.lower() == "coherelabs/cohere-transcribe-03-2026"

        if self._is_cohere:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            model_kwargs: dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": self.torch_dtype,
            }
            if hf_token:
                model_kwargs["token"] = hf_token

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )
            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs,
            ).to(self.device)
            # Cohere's processor returns 'length', which its generate path uses
            # even though the base Transformers validator does not see it.
            self.model._validate_model_kwargs = lambda kwargs, model_kwargs=None: None
            return

        from transformers import pipeline

        pipe_kwargs: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": trust_remote_code,
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
        if self._is_cohere:
            hypotheses = [self._transcribe_cohere(item) for item in prepared]
            if return_hypotheses:
                return hypotheses
            return [hyp.text for hyp in hypotheses]

        # Some custom ASR pipelines emit variable-length feature tensors that do
        # not collate correctly in batches. Keep the generic path conservative.
        pipe_call_kwargs: dict[str, Any] = {"batch_size": 1}

        should_return_timestamps = timestamps and self.return_timestamps
        if should_return_timestamps:
            pipe_call_kwargs["return_timestamps"] = True
        if self.chunk_length_s:
            pipe_call_kwargs["chunk_length_s"] = self.chunk_length_s
        if self.stride_length_s:
            pipe_call_kwargs["stride_length_s"] = self.stride_length_s

        outputs = self.pipe(prepared, **pipe_call_kwargs)
        if isinstance(outputs, dict):
            outputs = [outputs]

        hypotheses = [
            self._to_hypothesis(output, timestamps=should_return_timestamps)
            for output in outputs
        ]

        if return_hypotheses:
            return hypotheses
        return [hyp.text for hyp in hypotheses]

    def _transcribe_cohere(self, audio: Any) -> SimpleHypothesis:
        proc_inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language=self.language,
        )
        audio_chunk_index = proc_inputs.get("audio_chunk_index")
        proc_inputs = self._prepare_cohere_inputs(proc_inputs)

        with torch.no_grad():
            output_ids = self.model.generate(**proc_inputs, max_new_tokens=256)

        decode_kwargs: dict[str, Any] = {"skip_special_tokens": True}
        if audio_chunk_index is not None:
            decode_kwargs["audio_chunk_index"] = audio_chunk_index
            decode_kwargs["language"] = self.language

        text = self.processor.decode(output_ids, **decode_kwargs)
        if isinstance(text, list):
            text = " ".join(str(item).strip() for item in text if str(item).strip())
        return SimpleHypothesis(text=str(text).strip(), timestamp={"word": [], "segment": []})

    def _prepare_cohere_inputs(self, proc_inputs: Any) -> dict[str, Any]:
        target_device = getattr(self.model, "device", torch.device(self.device))
        target_dtype = getattr(self.model, "dtype", self.torch_dtype)

        moved: dict[str, Any] = {}
        for key, value in dict(proc_inputs).items():
            if isinstance(value, torch.Tensor):
                if torch.is_floating_point(value) or torch.is_complex(value):
                    value = value.to(target_device, dtype=target_dtype)
                else:
                    value = value.to(target_device)
            moved[key] = value

        self._normalize_cohere_input_features(moved)
        return moved

    def _normalize_cohere_input_features(self, proc_inputs: dict[str, Any]) -> None:
        features = proc_inputs.get("input_features")
        if not isinstance(features, torch.Tensor) or features.dim() != 3:
            return

        feature_size = self._cohere_feature_size(default=128)
        if features.shape[1] == feature_size and features.shape[-1] != feature_size:
            proc_inputs["input_features"] = features.transpose(1, 2).contiguous()

    def _cohere_feature_size(self, default: int = 128) -> int:
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        feature_size = getattr(feature_extractor, "feature_size", None)
        if feature_size is None:
            feature_size = getattr(feature_extractor, "num_mel_bins", None)
        try:
            return int(feature_size)
        except (TypeError, ValueError):
            return default

    def _prepare_input(self, item: Any) -> Any:
        if isinstance(item, (str, Path)):
            if self._is_cohere:
                from transformers.audio_utils import load_audio

                return load_audio(str(item), sampling_rate=16000)
            return str(item)

        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.dim() == 2 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            if self._is_cohere:
                return tensor.float().numpy()
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

            if not chunk_text:
                continue
            if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue

            start, end = ts[0], ts[1]
            if start is None or end is None:
                continue

            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError):
                continue

            if end_f <= start_f:
                continue

            words.append({"word": chunk_text, "start": start_f, "end": end_f})
            segments.append({"segment": chunk_text, "start": start_f, "end": end_f})

        return SimpleHypothesis(text=text, timestamp={"word": words, "segment": segments})
