import argparse
import json
import os
import sys
from pathlib import Path

import requests
import torch
import torchaudio
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from diarization import diarize_waveform
from diarize_align import assign_speakers, group_words_into_segments


DEFAULT_AUDIO = "Examples/MoreOrLess.wav"
DEFAULT_OUT_DIR = "tmp/probes/diarization_align"
ASR_URL = "http://localhost:8000/v1/audio/transcriptions"


def run_asr(audio_path: Path):
    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "audio/wav")}
        data = {
            "response_format": "verbose_json",
            "timestamps": "word",
            "diarization": "false",
        }
        resp = requests.post(ASR_URL, files=files, data=data, timeout=300)
    resp.raise_for_status()
    return resp.json()


def load_audio_mono_16k(path: Path) -> torch.Tensor:
    try:
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    except Exception:
        data, sr = sf.read(str(path), always_2d=True)
        waveform = torch.from_numpy(data.T)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform


def pick_tmp_dir(out_dir: Path) -> Path:
    shm = Path("/dev/shm")
    if shm.exists() and shm.is_dir() and os.access(str(shm), os.W_OK):
        return shm
    return out_dir


def run_diarization(audio_path: Path, tmp_dir: Path):
    waveform = load_audio_mono_16k(audio_path)
    return diarize_waveform(waveform, 16000, tmp_dir)


def main():
    parser = argparse.ArgumentParser(description="Dry-run ASR + speaker diarization + alignment.")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Path to audio file.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for JSON artifacts.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asr = run_asr(audio_path)
    words = asr.get("words", [])
    tmp_dir = pick_tmp_dir(out_dir)
    turns = run_diarization(audio_path, tmp_dir)

    words_with_speaker = assign_speakers(words, turns, max_gap_s=0.5)
    segments = group_words_into_segments(words_with_speaker)

    (out_dir / "asr_words.json").write_text(json.dumps(words, indent=2))
    (out_dir / "diarization_turns.json").write_text(json.dumps(turns, indent=2))
    (out_dir / "aligned_words.json").write_text(json.dumps(words_with_speaker, indent=2))
    (out_dir / "aligned_segments.json").write_text(json.dumps(segments, indent=2))

    print(f"Wrote outputs to {out_dir}")
    print(f"Words: {len(words)} | Turns: {len(turns)} | Segments: {len(segments)}")


if __name__ == "__main__":
    main()
