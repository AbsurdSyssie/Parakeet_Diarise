#!/usr/bin/env python3
"""Transcribe Silero VAD chunks and merge timestamps into a global timeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
import types

try:
    import torch
except Exception:  # pragma: no cover - optional for unit tests
    torch = None

try:
    import nemo.collections.asr as nemo_asr
except Exception:  # pragma: no cover - optional for unit tests
    nemo_asr = None

from asr_merge import build_chunk_meta, dedup_overlaps, offset_words, sort_words

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe VAD chunks and merge word timestamps."
    )
    parser.add_argument(
        "--chunk-dir",
        default="Output/vad_chunks",
        help="Directory with VAD chunk WAVs (default: Output/vad_chunks)",
    )
    parser.add_argument(
        "--out-json",
        default="Output/merged_words.json",
        help="Output JSON for merged words (default: Output/merged_words.json)",
    )
    parser.add_argument(
        "--out-text",
        default="Output/merged_text.txt",
        help="Output text file (default: Output/merged_text.txt)",
    )
    parser.add_argument(
        "--pad-left-s",
        type=float,
        default=0.2,
        help="Left padding added during VAD in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--pad-right-s",
        type=float,
        default=0.2,
        help="Right padding added during VAD in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Chunk batch size for ASR (default: 4)",
    )
    return parser.parse_args()


def _patch_transcribe_dataloader_no_lhotse(asr_model) -> None:
    def _setup_transcribe_dataloader_no_lhotse(self, config):
        if "manifest_filepath" in config:
            manifest_filepath = config["manifest_filepath"]
            batch_size = config["batch_size"]
        else:
            manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
            batch_size = min(config["batch_size"], len(config["paths2audio_files"]))

        dl_config = {
            "use_lhotse": False,
            "manifest_filepath": manifest_filepath,
            "sample_rate": self.preprocessor._sample_rate,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": config.get("num_workers", min(batch_size, os.cpu_count() - 1)),
            "pin_memory": True,
            "channel_selector": config.get("channel_selector", None),
            "use_start_end_token": self.cfg.validation_ds.get("use_start_end_token", False),
        }

        if config.get("augmentor"):
            dl_config["augmentor"] = config.get("augmentor")

        return self._setup_dataloader_from_config(config=dl_config)

    asr_model._setup_transcribe_dataloader = types.MethodType(
        _setup_transcribe_dataloader_no_lhotse, asr_model
    )


def _collect_chunks(chunk_dir: Path, pad_left_s: float, pad_right_s: float) -> List[Tuple[Path, float]]:
    chunks: List[Tuple[Path, float]] = []
    for path in sorted(chunk_dir.glob("*.wav")):
        meta = build_chunk_meta(path.name, pad_left_s, pad_right_s)
        chunks.append((path, meta.effective_start_s))
    chunks.sort(key=lambda item: item[1])
    return chunks


def _merge_hypotheses(outputs, offsets):
    merged_words = []
    for hyp, offset_s in zip(outputs, offsets):
        if not hasattr(hyp, "timestamp") or not isinstance(hyp.timestamp, dict):
            continue
        words = hyp.timestamp.get("word") or []
        merged_words.extend(offset_words(words, offset_s))
    return merged_words


def _merge_texts(outputs, offsets) -> str:
    texts = []
    for hyp, offset_s in zip(outputs, offsets):
        text = getattr(hyp, "text", "")
        if text:
            texts.append((offset_s, text.strip()))
    texts.sort(key=lambda item: item[0])
    merged = []
    last = None
    for _, text in texts:
        if not text or text == last:
            continue
        merged.append(text)
        last = text
    return " ".join(merged)


def _merge_segments(outputs, offsets) -> list[dict]:
    segments = []
    for hyp, offset_s in zip(outputs, offsets):
        if not hasattr(hyp, "timestamp") or not isinstance(hyp.timestamp, dict):
            continue
        segs = hyp.timestamp.get("segment") or []
        for seg in segs:
            text = seg.get("segment")
            if not text:
                continue
            segments.append(
                {
                    "start": float(seg["start"]) + offset_s,
                    "end": float(seg["end"]) + offset_s,
                    "text": text,
                }
            )
    segments.sort(key=lambda item: (item["start"], item["end"]))
    deduped = []
    for seg in segments:
        if deduped:
            prev = deduped[-1]
            if seg["text"] == prev["text"] and seg["start"] <= prev["end"]:
                continue
        deduped.append(seg)
    return deduped


def transcribe_chunks_with_model(
    asr_model,
    chunk_dir: Path,
    pad_left_s: float,
    pad_right_s: float,
    batch_size: int,
) -> list[dict]:
    chunk_dir = Path(chunk_dir)
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk dir not found: {chunk_dir}")

    chunks = _collect_chunks(chunk_dir, pad_left_s, pad_right_s)
    if not chunks:
        return []

    merged_words = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        paths = [str(item[0]) for item in batch]
        offsets = [item[1] for item in batch]

        outputs = asr_model.transcribe(
            paths,
            timestamps=True,
            verbose=False,
            batch_size=len(paths),
            num_workers=0,
            return_hypotheses=True,
        )
        merged_words.extend(_merge_hypotheses(outputs, offsets))

    merged_words = dedup_overlaps(sort_words(merged_words))
    return merged_words


def transcribe_chunks_in_memory(
    asr_model,
    chunks: list[dict],
    batch_size: int,
    pad_left_s: float = 0.0,
) -> list[dict]:
    if not chunks:
        return []

    merged_words = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        waveforms = [item["waveform"] for item in batch]
        offsets = [max(0.0, item["start_s"] - pad_left_s) for item in batch]

        outputs = asr_model.transcribe(
            waveforms,
            timestamps=True,
            verbose=False,
            batch_size=len(waveforms),
            num_workers=0,
            return_hypotheses=True,
        )

        merged_words.extend(_merge_hypotheses(outputs, offsets))

    merged_words = dedup_overlaps(sort_words(merged_words))
    return merged_words


def transcribe_chunks_with_model_mode(
    asr_model,
    chunk_dir: Path,
    pad_left_s: float,
    pad_right_s: float,
    batch_size: int,
    timestamps: str,
) -> dict:
    if timestamps not in {"word", "segment", "none"}:
        raise ValueError("timestamps must be 'word', 'segment', or 'none'")
    chunk_dir = Path(chunk_dir)
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk dir not found: {chunk_dir}")

    chunks = _collect_chunks(chunk_dir, pad_left_s, pad_right_s)
    if not chunks:
        return {"words": [], "segments": [], "text": ""}

    words: list[dict] = []
    segments: list[dict] = []
    texts = []
    use_timestamps = timestamps != "none"

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        paths = [str(item[0]) for item in batch]
        offsets = [item[1] for item in batch]

        outputs = asr_model.transcribe(
            paths,
            timestamps=use_timestamps,
            verbose=False,
            batch_size=len(paths),
            num_workers=0,
            return_hypotheses=True,
        )
        texts.append(_merge_texts(outputs, offsets))
        if timestamps == "word":
            words.extend(_merge_hypotheses(outputs, offsets))
        elif timestamps == "segment":
            segments.extend(_merge_segments(outputs, offsets))

    if timestamps == "word":
        words = dedup_overlaps(sort_words(words))
    text = " ".join([t for t in texts if t])
    return {"words": words, "segments": segments, "text": text}


def transcribe_chunks_in_memory_mode(
    asr_model,
    chunks: list[dict],
    batch_size: int,
    timestamps: str,
    pad_left_s: float = 0.0,
) -> dict:
    if timestamps not in {"word", "segment", "none"}:
        raise ValueError("timestamps must be 'word', 'segment', or 'none'")
    if not chunks:
        return {"words": [], "segments": [], "text": ""}

    words: list[dict] = []
    segments: list[dict] = []
    texts = []
    use_timestamps = timestamps != "none"

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        waveforms = [item["waveform"] for item in batch]
        offsets = [max(0.0, item["start_s"] - pad_left_s) for item in batch]

        outputs = asr_model.transcribe(
            waveforms,
            timestamps=use_timestamps,
            verbose=False,
            batch_size=len(waveforms),
            num_workers=0,
            return_hypotheses=True,
        )
        texts.append(_merge_texts(outputs, offsets))
        if timestamps == "word":
            words.extend(_merge_hypotheses(outputs, offsets))
        elif timestamps == "segment":
            segments.extend(_merge_segments(outputs, offsets))

    if timestamps == "word":
        words = dedup_overlaps(sort_words(words))
    text = " ".join([t for t in texts if t])
    return {"words": words, "segments": segments, "text": text}


def transcribe_chunks(
    chunk_dir: Path,
    pad_left_s: float,
    pad_right_s: float,
    batch_size: int,
) -> list[dict]:
    if nemo_asr is None:
        raise RuntimeError("NeMo is required for transcribe_chunks; not available in this environment.")
    if torch is None:
        raise RuntimeError("Torch is required for transcribe_chunks; not available in this environment.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    asr_model = asr_model.to(device)
    _patch_transcribe_dataloader_no_lhotse(asr_model)

    return transcribe_chunks_with_model(
        asr_model=asr_model,
        chunk_dir=chunk_dir,
        pad_left_s=pad_left_s,
        pad_right_s=pad_right_s,
        batch_size=batch_size,
    )


def main() -> int:
    args = parse_args()
    chunk_dir = Path(args.chunk_dir)
    if not chunk_dir.exists():
        raise SystemExit(f"Chunk dir not found: {chunk_dir}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_text = Path(args.out_text)
    out_text.parent.mkdir(parents=True, exist_ok=True)

    merged_words = transcribe_chunks(
        chunk_dir=chunk_dir,
        pad_left_s=args.pad_left_s,
        pad_right_s=args.pad_right_s,
        batch_size=args.batch_size,
    )

    if not merged_words:
        print("No chunks found.")
        return 0

    out_json.write_text(json.dumps(merged_words, indent=2))
    out_text.write_text(" ".join([w["word"] for w in merged_words]))

    print(f"Merged {len(merged_words)} words from {len(list(chunk_dir.glob('*.wav')))} chunks")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
