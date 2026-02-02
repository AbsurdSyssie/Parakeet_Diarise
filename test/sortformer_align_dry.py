import argparse
import json
import os
import sys
from pathlib import Path

import requests
import torch
import torchaudio
import soundfile as sf
from nemo.collections.asr.models import SortformerEncLabelModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from diarize_align import assign_speakers, group_words_into_segments


DEFAULT_AUDIO = "Examples/MoreOrLess.wav"
DEFAULT_OUT_DIR = "test/tmp/sortformer_align"
ASR_URL = "http://localhost:8000/v1/audio/transcriptions"


def normalize_speaker(label: str) -> str:
    if not isinstance(label, str):
        return "UNKNOWN"
    if label.startswith("SPEAKER_"):
        return label
    if label.startswith("speaker_"):
        try:
            idx = int(label.split("_", 1)[1])
            return f"SPEAKER_{idx:02d}"
        except (ValueError, IndexError):
            return label
    return label


def normalize_segments(raw_segments):
    segments = []
    for seg in raw_segments:
        if isinstance(seg, dict):
            start = float(seg.get("start", seg.get("start_time", 0.0)))
            end = float(seg.get("end", seg.get("end_time", 0.0)))
            speaker = seg.get("speaker", seg.get("speaker_label", seg.get("label", "UNKNOWN")))
        elif isinstance(seg, str):
            parts = seg.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
            else:
                segments.append({"raw": seg})
                continue
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            start = float(seg[0])
            end = float(seg[1])
            speaker = seg[2]
        else:
            segments.append({"raw": seg})
            continue
        segments.append({"start": start, "end": end, "speaker": normalize_speaker(speaker)})
    return segments


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


def run_sortformer(audio_path: Path, tmp_dir: Path):
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
    diar_model.eval()
    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300

    waveform = load_audio_mono_16k(audio_path)
    audio_np = waveform.squeeze(0).numpy().astype("float32")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{audio_path.stem}.mono16k.wav"
    sf.write(str(tmp_path), audio_np, 16000)
    try:
        predicted_segments = diar_model.diarize(audio=[str(tmp_path)], batch_size=1)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    raw = predicted_segments[0] if predicted_segments else []
    turns = normalize_segments(raw)
    turns = [t for t in turns if "start" in t and "end" in t]
    turns.sort(key=lambda t: (float(t["start"]), float(t["end"])))
    return turns


def main():
    parser = argparse.ArgumentParser(description="Dry-run ASR + Sortformer diarization + alignment.")
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
    turns = run_sortformer(audio_path, tmp_dir)

    words_with_speaker = assign_speakers(words, turns, max_gap_s=0.5)
    segments = group_words_into_segments(words_with_speaker)

    (out_dir / "asr_words.json").write_text(json.dumps(words, indent=2))
    (out_dir / "sortformer_turns.json").write_text(json.dumps(turns, indent=2))
    (out_dir / "aligned_words.json").write_text(json.dumps(words_with_speaker, indent=2))
    (out_dir / "aligned_segments.json").write_text(json.dumps(segments, indent=2))

    print(f"Wrote outputs to {out_dir}")
    print(f"Words: {len(words)} | Turns: {len(turns)} | Segments: {len(segments)}")


if __name__ == "__main__":
    main()
