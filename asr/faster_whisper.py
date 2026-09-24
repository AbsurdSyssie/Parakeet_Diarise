"""faster-whisper ASR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from .types import SimpleHypothesis


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
