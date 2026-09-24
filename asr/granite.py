"""IBM Granite Speech ASR adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio

from .types import SimpleHypothesis


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
