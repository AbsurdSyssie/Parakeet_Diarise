#!/usr/bin/env python3
"""Minimal Parakeet transcription with timestamps."""

import argparse
import os
import sys

import types

import torch
import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file with Parakeet and print timestamps."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="example.wav",
        help="Path to WAV file (default: example.wav)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.audio_path):
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    asr_model = asr_model.to(device)

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

    # Avoid Lhotse path issues by patching the transcribe dataloader to set use_lhotse=False.
    asr_model._setup_transcribe_dataloader = types.MethodType(
        _setup_transcribe_dataloader_no_lhotse, asr_model
    )

    outputs = asr_model.transcribe(
        [args.audio_path],
        timestamps=True,
        verbose=False,
        batch_size=1,
        num_workers=0,
        return_hypotheses=True,
    )

    hyp = outputs[0]
    print("TEXT:")
    print(hyp.text)

    if hasattr(hyp, "timestamp") and isinstance(hyp.timestamp, dict):
        print("\nWORD TIMESTAMPS:")
        print(hyp.timestamp.get("word"))
        print("\nSEGMENT TIMESTAMPS:")
        print(hyp.timestamp.get("segment"))
    else:
        print("\nNo timestamp data found in hypothesis.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
