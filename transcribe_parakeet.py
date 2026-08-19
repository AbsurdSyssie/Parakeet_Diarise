#!/usr/bin/env python3
"""Command-line transcription using any configured ASR backend."""

import argparse
import os
import sys

import torch

from asr_backend import ASRConfig, load_asr_backend, resolve_asr_model
from chunk_transcribe import _patch_transcribe_dataloader_no_lhotse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with a selectable ASR backend."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="example.wav",
        help="Path to an audio file (default: example.wav)",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("ASR_BACKEND", "nemo"),
        choices=["nemo", "whisper", "faster-whisper", "transformers-asr", "hf-asr"],
        help="ASR backend (default: ASR_BACKEND or nemo)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ASR_MODEL", ""),
        help="Model alias or repository name (default: ASR_MODEL or backend default)",
    )
    parser.add_argument(
        "--timestamps",
        choices=["word", "segment", "none"],
        default="word",
        help="Timestamp detail to print (default: word)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=os.getenv("ASR_TRUST_REMOTE_CODE", "0").lower() in {"1", "true", "yes", "on"},
        help="Allow model repository code (automatically enabled for the Granite alias)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default=os.getenv("ASR_ATTENTION_IMPLEMENTATION", "sdpa"),
        help="Transformers attention implementation (default: sdpa)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.audio_path):
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        return 1

    model_name = resolve_asr_model(args.backend, args.model)
    is_granite = model_name.lower() == "ibm-granite/granite-speech-4.1-2b-nar"
    config = ASRConfig(
        backend=args.backend,
        model_key=args.model or model_name,
        model_name=model_name,
        hf_token=os.getenv("HF_TOKEN"),
        trust_remote_code=args.trust_remote_code or is_granite,
        return_timestamps=args.timestamps != "none",
        device=args.device,
        attention_implementation=args.attention_implementation,
    )
    asr_model = load_asr_backend(config)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    asr_model = asr_model.to(device)
    if args.backend == "nemo":
        _patch_transcribe_dataloader_no_lhotse(asr_model)

    outputs = asr_model.transcribe(
        [args.audio_path],
        timestamps=args.timestamps != "none",
        verbose=False,
        batch_size=args.batch_size,
        num_workers=0,
        return_hypotheses=True,
    )

    hyp = outputs[0]
    print("TEXT:")
    print(hyp.text)

    if args.timestamps != "none" and hasattr(hyp, "timestamp") and isinstance(hyp.timestamp, dict):
        print("\nWORD TIMESTAMPS:")
        print(hyp.timestamp.get("word"))
        print("\nSEGMENT TIMESTAMPS:")
        print(hyp.timestamp.get("segment"))
    else:
        print("\nNo timestamp data found in hypothesis.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
