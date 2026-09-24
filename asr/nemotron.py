"""NVIDIA Nemotron ASR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .types import SimpleHypothesis


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
