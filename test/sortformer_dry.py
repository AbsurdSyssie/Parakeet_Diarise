import argparse
import json
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from nemo.collections.asr.models import SortformerEncLabelModel


DEFAULT_AUDIO = "Examples/MoreOrLessFull.wav"
DEFAULT_OUT = "Output/sortformer_segments.json"
FALLBACK_OUT = "test/tmp/sortformer_segments.json"


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


def normalize_segments(raw_segments):
    segments = []
    for seg in raw_segments:
        if isinstance(seg, dict):
            start = float(seg.get("start", seg.get("start_time", 0.0)))
            end = float(seg.get("end", seg.get("end_time", 0.0)))
            speaker = seg.get("speaker", seg.get("speaker_label", seg.get("label", "UNKNOWN")))
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            start = float(seg[0])
            end = float(seg[1])
            speaker = seg[2]
        else:
            # Fallback: keep raw representation
            segments.append({"raw": seg})
            continue
        segments.append({"start": start, "end": end, "speaker": speaker})
    return segments


def main():
    parser = argparse.ArgumentParser(description="Dry-run Sortformer diarization on a local audio file.")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Path to audio file (WAV recommended).")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to write JSON diarization segments.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    audio_input = [str(audio_path)]

    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
    diar_model.eval()

    # Streaming params from docs/sortformer.md quick-start defaults
    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300

    predicted_segments = diar_model.diarize(audio=audio_input, batch_size=1)
    raw = predicted_segments[0] if predicted_segments else []
    segments = normalize_segments(raw)

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
