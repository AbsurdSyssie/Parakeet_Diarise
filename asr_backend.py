#!/usr/bin/env python3
"""ASR backend loading and compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torchaudio
import nemo.collections.asr as nemo_asr


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


@dataclass
class ASRConfig:
    backend: str
    model_key: str
    model_name: str
    hf_token: str | None = None
    language: str = "en"
    task: str = "transcribe"
    trust_remote_code: bool = False
    return_timestamps: bool = False
    chunk_length_s: int | None = None
    stride_length_s: int | tuple[int, int] | None = None
    device: str | None = None
    attention_implementation: str | None = None


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


class FasterWhisperASRBackend:
    """faster-whisper / CTranslate2 adapter with a NeMo-like transcribe interface."""

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        language: str = "en",
        task: str = "transcribe",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper backend requires the 'faster-whisper' package. "
                "Install it before selecting Systran/faster-whisper models."
            ) from exc

        self.model_name = model_name
        self.language = language or None
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"

        model_kwargs: dict[str, Any] = {
            "device": self.device,
            "compute_type": self.compute_type,
        }
        if hf_token:
            model_kwargs["token"] = hf_token

        self.model = WhisperModel(model_name, **model_kwargs)

    def to(self, device: str | torch.device):
        # CTranslate2 device placement is configured at load time.
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
        hypotheses: list[SimpleHypothesis] = []
        for item in inputs:
            audio = self._prepare_input(item)
            segments_iter, _info = self.model.transcribe(
                audio,
                language=self.language,
                task=self.task,
                word_timestamps=timestamps,
            )
            segments = list(segments_iter)
            text = " ".join((segment.text or "").strip() for segment in segments).strip()

            words: list[dict[str, Any]] = []
            segment_items: list[dict[str, Any]] = []
            if timestamps:
                for segment in segments:
                    segment_text = (segment.text or "").strip()
                    start = float(segment.start or 0.0)
                    end = float(segment.end or start)
                    if segment_text and end > start:
                        segment_items.append({"segment": segment_text, "start": start, "end": end})
                    for word in getattr(segment, "words", None) or []:
                        word_text = (word.word or "").strip()
                        if not word_text:
                            continue
                        word_start = float(word.start or 0.0)
                        word_end = float(word.end or word_start)
                        if word_end <= word_start:
                            word_end = word_start + 0.01
                        words.append({"word": word_text, "start": word_start, "end": word_end})

            hypotheses.append(
                SimpleHypothesis(
                    text=text,
                    timestamp={"word": words, "segment": segment_items},
                )
            )

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
            return tensor.float().numpy()

        return item


class GraniteASRBackend:
    """IBM Granite Speech NAR adapter with the shared transcribe interface."""

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        trust_remote_code: bool = False,
        device: str | None = None,
        attention_implementation: str | None = None,
    ) -> None:
        if not trust_remote_code:
            raise RuntimeError(
                "IBM Granite Speech requires trust_remote_code=True because its "
                "model and processor implementations are stored in the model repository."
            )

        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.attention_implementation = attention_implementation or "sdpa"
        if self.attention_implementation == "flash_attention_2":
            if not self.device.startswith("cuda"):
                raise RuntimeError("flash_attention_2 requires a CUDA device")
            if importlib.util.find_spec("flash_attn") is None:
                raise RuntimeError(
                    "flash_attention_2 was requested but flash-attn is not installed. "
                    "Install the optional Granite GPU dependency or use sdpa."
                )

        common_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if hf_token:
            common_kwargs["token"] = hf_token

        self.processor = AutoProcessor.from_pretrained(model_name, **common_kwargs)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            attn_implementation=self.attention_implementation,
            **common_kwargs,
        ).eval().to(self.device)

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        requested = (device or "auto").strip().lower()
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Granite but no CUDA device is available")
        if requested == "cuda":
            return "cuda:0"
        if requested == "cpu" or requested.startswith("cuda:"):
            return requested
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")

    def to(self, device: str | torch.device):
        target = self._resolve_device(str(device))
        if target == self.device:
            return self
        self.device = target
        self.torch_dtype = torch.bfloat16 if target.startswith("cuda") else torch.float32
        self.model = self.model.to(device=target, dtype=self.torch_dtype)
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
        del timestamps, verbose, num_workers
        prepared = [self._prepare_audio(item) for item in inputs]
        hypotheses: list[SimpleHypothesis] = []
        effective_batch_size = max(1, int(batch_size))

        for start in range(0, len(prepared), effective_batch_size):
            batch = prepared[start : start + effective_batch_size]
            model_inputs = self.processor(audios=batch, device=self.device)
            output = self.model.transcribe(**model_inputs)
            texts = self.processor.batch_decode(output.preds)
            hypotheses.extend(
                SimpleHypothesis(
                    text=str(text).strip(),
                    timestamp={"word": [], "segment": []},
                )
                for text in texts
            )

        if return_hypotheses:
            return hypotheses
        return [hyp.text for hyp in hypotheses]

    def _prepare_audio(self, item: Any) -> torch.Tensor:
        if isinstance(item, (str, Path)):
            try:
                waveform, sample_rate = torchaudio.load(str(item))
            except ImportError as exc:
                if "TorchCodec" not in str(exc):
                    raise
                import soundfile as sf

                data, sample_rate = sf.read(str(item), always_2d=True, dtype="float32")
                waveform = torch.from_numpy(data.T).contiguous()
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            item = waveform

        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().float()
            if tensor.dim() == 2:
                tensor = tensor.mean(dim=0)
            if tensor.dim() != 1:
                raise ValueError(f"Expected mono audio tensor, got shape {tuple(tensor.shape)}")
            return tensor

        try:
            tensor = torch.as_tensor(item, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Unsupported Granite audio input type: {type(item).__name__}") from exc
        if tensor.dim() != 1:
            raise ValueError(f"Expected mono audio array, got shape {tuple(tensor.shape)}")
        return tensor


class NemotronASRBackend:
    """NVIDIA Nemotron 3.5 ASR Streaming adapter with the shared transcribe interface."""

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
        if not trust_remote_code:
            raise RuntimeError(
                "Nemotron ASR requires trust_remote_code=True because its "
                "model and processor implementations are stored in the model repository."
            )

        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.return_timestamps = return_timestamps
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.language = language
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32

        common_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if hf_token:
            common_kwargs["token"] = hf_token

        self.processor = AutoProcessor.from_pretrained(model_name, **common_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **common_kwargs).to(self.device)
        self.model.eval()

    def to(self, device: str | torch.device):
        target = str(device)
        if target == self.device:
            return self
        self.device = target
        self.model = self.model.to(device)
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
        hypotheses: list[SimpleHypothesis] = []

        for audio in prepared:
            # Skip warmup dummy input (all zeros)
            if isinstance(audio, np.ndarray) and audio.size > 0 and np.all(audio == 0):
                hypotheses.append(SimpleHypothesis(text="", timestamp={"word": [], "segment": []}))
                continue
            if isinstance(audio, torch.Tensor) and audio.numel() > 0 and torch.all(audio == 0):
                hypotheses.append(SimpleHypothesis(text="", timestamp={"word": [], "segment": []}))
                continue

            # Process audio
            proc_inputs = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt")
            # Fix num_lookahead_tokens to be a tensor
            proc_inputs["num_lookahead_tokens"] = torch.tensor(proc_inputs["num_lookahead_tokens"])
            proc_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in proc_inputs.items()}

            # Generate
            with torch.no_grad():
                if timestamps and self.return_timestamps:
                    outputs = self.model.generate(
                        **proc_inputs,
                        max_new_tokens=256,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    sequences = outputs.sequences
                    durations = outputs.durations if hasattr(outputs, "durations") else None
                else:
                    outputs = self.model.generate(**proc_inputs, max_new_tokens=256)
                    sequences = outputs
                    durations = None

            # Decode
            if durations is not None:
                text = self.processor.decode(sequences, durations=durations, skip_special_tokens=True)
            else:
                text = self.processor.decode(sequences, skip_special_tokens=True)

            # Parse timestamps from decode output
            words: list[dict[str, Any]] = []
            segments: list[dict[str, Any]] = []

            if timestamps and self.return_timestamps and isinstance(text, tuple) and len(text) == 2:
                # text is (text_list, timestamps_list)
                text_str = text[0][0] if isinstance(text[0], list) else text[0]
                ts_list = text[1][0] if isinstance(text[1], list) else text[1]
                for ts in ts_list:
                    if isinstance(ts, dict) and "token" in ts and "start" in ts and "end" in ts:
                        words.append({"word": ts["token"], "start": float(ts["start"]), "end": float(ts["end"])})
                # Also create segments from words (group by punctuation or fixed intervals)
                if words:
                    # Simple segmentation: split by sentence-ending punctuation
                    segment_text = ""
                    segment_start = words[0]["start"]
                    for w in words:
                        segment_text += w["word"]
                        if w["word"].endswith((".", "!", "?")):
                            segments.append({"segment": segment_text.strip(), "start": segment_start, "end": w["end"]})
                            segment_text = ""
                            if w != words[-1]:
                                segment_start = words[words.index(w) + 1]["start"]
                    if segment_text:
                        segments.append({"segment": segment_text.strip(), "start": segment_start, "end": words[-1]["end"]})
            else:
                text_str = text[0] if isinstance(text, list) else text

            hypotheses.append(
                SimpleHypothesis(
                    text=text_str,
                    timestamp={"word": words, "segment": segments},
                )
            )

        if return_hypotheses:
            return hypotheses
        return [hyp.text for hyp in hypotheses]

    def _prepare_input(self, item: Any) -> Any:
        if isinstance(item, (str, Path)):
            import soundfile as sf
            audio, sr = sf.read(str(item), dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio.T).contiguous()
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != 16000:
                import torchaudio
                audio = torchaudio.functional.resample(audio, sr, 16000)
            if audio.dim() == 2 and audio.size(0) == 1:
                audio = audio.squeeze(0)
            return audio.numpy()

        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.dim() == 2 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            return tensor.float().numpy()

        return item


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
