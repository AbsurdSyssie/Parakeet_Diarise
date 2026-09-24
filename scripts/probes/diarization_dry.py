import argparse
import json
import tempfile
from pathlib import Path
import sys

import torch
import torchaudio
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diarization import diarize_waveform


DEFAULT_AUDIO = "Examples/MoreOrLessFull.wav"
DEFAULT_OUT = "Output/sortformer_segments.json"
FALLBACK_OUT = "tmp/probes/diarization_segments.json"

def load_audio(path: Path, target_sr: int = 16000) -> torch.Tensor:
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        data, sr = sf.read(str(path), always_2d=True)
        waveform = torch.from_numpy(data.T)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform



def main():
    parser = argparse.ArgumentParser(description="Dry-run speaker diarization on a local audio file.")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Path to audio file (WAV recommended).")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to write JSON diarization segments.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    waveform = load_audio(audio_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        segments = diarize_waveform(waveform, 16000, Path(tmpdir))

    out_path = Path(args.out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(segments, indent=2))
    except PermissionError:
        out_path = Path(FALLBACK_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(segments, indent=2))

    print(f"Wrote {len(segments)} segments to {out_path}")
    for seg in segments[:10]:
        print(seg)


if __name__ == "__main__":
    main()
