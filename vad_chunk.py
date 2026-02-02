#!/usr/bin/env python3
"""Chunk a WAV file into speech segments using Silero VAD."""

import argparse
import time
from pathlib import Path

import os
import torch
import torchaudio

_VAD_MODEL = None
_VAD_UTILS = None
_VAD_LOADED_AT = None


def _resolve_vad_device() -> str:
    forced = os.environ.get("VAD_DEVICE")
    if forced:
        return forced.lower()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _energy_gate_intervals(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: int,
    threshold_db: float,
    min_active_ms: int,
    pad_ms: int,
    merge_gap_ms: int,
) -> tuple[list[dict], float]:
    if waveform.dim() == 2 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)
    if waveform.dim() != 1:
        raise ValueError("Energy gate expects mono waveform.")

    frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))
    total_frames = waveform.numel() // frame_len
    if total_frames == 0:
        return [{"start": 0, "end": waveform.numel()}], 0.0

    trimmed = waveform[: total_frames * frame_len]
    frames = trimmed.view(total_frames, frame_len)
    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-8)
    db = 20.0 * torch.log10(rms + 1e-8)
    active = db > threshold_db
    active_ratio = float(active.sum().item()) / float(active.numel())

    intervals = []
    start_idx = None
    for idx, is_active in enumerate(active.tolist()):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            end_idx = idx
            intervals.append({"start": start_idx * frame_len, "end": end_idx * frame_len})
            start_idx = None
    if start_idx is not None:
        intervals.append({"start": start_idx * frame_len, "end": total_frames * frame_len})

    if not intervals:
        return [], active_ratio

    min_active_samples = int(sample_rate * (min_active_ms / 1000.0))
    intervals = [iv for iv in intervals if (iv["end"] - iv["start"]) >= min_active_samples]
    if not intervals:
        return [], active_ratio

    pad_samples = int(sample_rate * (pad_ms / 1000.0))
    max_len = waveform.numel()
    for iv in intervals:
        iv["start"] = max(0, iv["start"] - pad_samples)
        iv["end"] = min(max_len, iv["end"] + pad_samples)

    if merge_gap_ms > 0 and len(intervals) > 1:
        merge_gap_samples = int(sample_rate * (merge_gap_ms / 1000.0))
        merged = [intervals[0]]
        for iv in intervals[1:]:
            gap = iv["start"] - merged[-1]["end"]
            if gap <= merge_gap_samples:
                merged[-1]["end"] = max(merged[-1]["end"], iv["end"])
            else:
                merged.append(iv)
        intervals = merged

    return intervals, active_ratio


def _uniform_intervals(
    total_samples: int,
    sample_rate: int,
    chunk_s: float,
    overlap_s: float,
) -> list[dict]:
    if total_samples <= 0:
        return []
    chunk_samples = int(sample_rate * chunk_s)
    overlap_samples = int(sample_rate * overlap_s)
    if chunk_samples <= 0:
        return []
    step = max(1, chunk_samples - overlap_samples)
    intervals = []
    cur = 0
    while cur < total_samples:
        end = min(cur + chunk_samples, total_samples)
        intervals.append({"start": cur, "end": end})
        if end >= total_samples:
            break
        cur = cur + step
    return intervals


def _load_vad(device: str):
    global _VAD_MODEL, _VAD_UTILS, _VAD_LOADED_AT
    if _VAD_MODEL is None or _VAD_UTILS is None:
        start = time.perf_counter()
        _VAD_MODEL, _VAD_UTILS = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _VAD_LOADED_AT = time.perf_counter() - start
        print(f"silero_vad load in {_VAD_LOADED_AT:.2f}s (device={device})")
    else:
        print(f"silero_vad cache hit (device={device})")
    _VAD_MODEL = _VAD_MODEL.to(device)
    return _VAD_MODEL, _VAD_UTILS


def _load_audio_mono_16k(input_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(input_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.dim() == 2 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)
    return waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Silero VAD and save speech chunks as WAV files."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="Examples/MoreOrLessFull.wav",
        help="Path to input WAV (default: Examples/MoreOrLessFull.wav)",
    )
    parser.add_argument(
        "--out-dir",
        default="Output/vad_chunks",
        help="Directory for chunked WAVs (default: Output/vad_chunks)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for VAD (default: 16000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.38,
        help="Speech probability threshold (default: 0.38)",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=250,
        help="Minimum speech duration in ms (default: 250)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=200,
        help="Minimum silence duration to split segments in ms (default: 200)",
    )
    parser.add_argument(
        "--merge-gap-ms",
        type=int,
        default=200,
        help="Merge gaps shorter than this in ms (default: 200)",
    )
    parser.add_argument(
        "--target-min-s",
        type=float,
        default=10.0,
        help="Target minimum chunk length in seconds (default: 10)",
    )
    parser.add_argument(
        "--target-max-s",
        type=float,
        default=20.0,
        help="Target maximum chunk length in seconds (default: 20)",
    )
    parser.add_argument(
        "--hard-max-s",
        type=float,
        default=30.0,
        help="Hard maximum chunk length in seconds (default: 30)",
    )
    parser.add_argument(
        "--overlap-s",
        type=float,
        default=1.0,
        help="Overlap between hard-cut chunks in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=0,
        help="Pad speech segments on each side in ms (default: 0)",
    )
    return parser.parse_args()


def run_vad_chunks(
    input_path: Path,
    out_dir: Path,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    merge_gap_ms: int,
    target_min_s: float,
    target_max_s: float,
    hard_max_s: float,
    overlap_s: float,
    speech_pad_ms: int,
) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    audio = _load_audio_mono_16k(input_path, sample_rate)
    return run_vad_chunks_from_waveform(
        waveform=audio,
        out_dir=out_dir,
        sample_rate=sample_rate,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        merge_gap_ms=merge_gap_ms,
        target_min_s=target_min_s,
        target_max_s=target_max_s,
        hard_max_s=hard_max_s,
        overlap_s=overlap_s,
        speech_pad_ms=speech_pad_ms,
    )


def run_vad_chunks_from_waveform(
    waveform: torch.Tensor,
    out_dir: Path,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    merge_gap_ms: int,
    target_min_s: float,
    target_max_s: float,
    hard_max_s: float,
    overlap_s: float,
    speech_pad_ms: int,
    chunk_waveform: torch.Tensor | None = None,
    chunk_sample_rate: int | None = None,
    energy_gate_override: bool | None = None,
    energy_overrides: dict | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_vad_device()
    energy_overrides = energy_overrides or {}
    if energy_gate_override is None:
        use_energy_gate = energy_overrides.get(
            "energy_gate",
            os.environ.get("VAD_ENERGY_GATE", "0") == "1",
        )
    else:
        use_energy_gate = bool(energy_gate_override)
    energy_frame_ms = int(energy_overrides.get("energy_frame_ms", os.environ.get("VAD_ENERGY_FRAME_MS", "100")))
    energy_threshold_db = float(energy_overrides.get("energy_db", os.environ.get("VAD_ENERGY_DB", "-40")))
    energy_min_active_ms = int(
        energy_overrides.get("energy_min_active_ms", os.environ.get("VAD_ENERGY_MIN_ACTIVE_MS", "300"))
    )
    energy_merge_gap_ms = int(
        energy_overrides.get("energy_merge_gap_ms", os.environ.get("VAD_ENERGY_MERGE_GAP_MS", "500"))
    )
    energy_active_ratio_skip = float(
        energy_overrides.get("energy_active_skip", os.environ.get("VAD_ENERGY_ACTIVE_SKIP", "0.7"))
    )
    uniform_chunk_s = float(energy_overrides.get("uniform_chunk_s", os.environ.get("VAD_UNIFORM_CHUNK_S", "45")))
    uniform_overlap_s = float(
        energy_overrides.get("uniform_overlap_s", os.environ.get("VAD_UNIFORM_OVERLAP_S", "0.5"))
    )

    model, utils = _load_vad(device)
    (get_speech_timestamps, save_audio, _, _, _) = utils

    audio = waveform
    if use_energy_gate:
        print(
            "energy gate config: "
            f"db={energy_threshold_db}, frame_ms={energy_frame_ms}, "
            f"min_active_ms={energy_min_active_ms}, merge_gap_ms={energy_merge_gap_ms}, "
            f"skip_ratio={energy_active_ratio_skip}, chunk_s={uniform_chunk_s}, "
            f"overlap_s={uniform_overlap_s}"
        )
        intervals, active_ratio = _energy_gate_intervals(
            waveform=audio.detach().cpu(),
            sample_rate=sample_rate,
            frame_ms=energy_frame_ms,
            threshold_db=energy_threshold_db,
            min_active_ms=energy_min_active_ms,
            pad_ms=speech_pad_ms,
            merge_gap_ms=energy_merge_gap_ms,
        )
        if active_ratio >= energy_active_ratio_skip:
            speech_timestamps = _uniform_intervals(
                total_samples=audio.shape[-1],
                sample_rate=sample_rate,
                chunk_s=uniform_chunk_s,
                overlap_s=uniform_overlap_s,
            )
            print(
                f"energy gate skip (ratio={active_ratio:.2f}) -> "
                f"{len(speech_timestamps)} uniform intervals"
            )
        else:
            print(f"energy gate enabled: {len(intervals)} intervals (ratio={active_ratio:.2f})")
            speech_timestamps = []
            for iv in intervals:
                segment = audio[iv["start"] : iv["end"]]
                audio_vad = segment.to(device) if device == "cuda" else segment
                seg_ts = get_speech_timestamps(
                    audio_vad,
                    model,
                    sampling_rate=sample_rate,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_ms,
                    min_silence_duration_ms=min_silence_ms,
                    speech_pad_ms=speech_pad_ms,
                )
                for ts in seg_ts:
                    speech_timestamps.append(
                        {"start": ts["start"] + iv["start"], "end": ts["end"] + iv["start"]}
                    )
    else:
        intervals = [{"start": 0, "end": audio.shape[-1]}]
        speech_timestamps = []
        for iv in intervals:
            segment = audio[iv["start"] : iv["end"]]
            audio_vad = segment.to(device) if device == "cuda" else segment
            seg_ts = get_speech_timestamps(
                audio_vad,
                model,
                sampling_rate=sample_rate,
                threshold=threshold,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
            )
            for ts in seg_ts:
                speech_timestamps.append(
                    {"start": ts["start"] + iv["start"], "end": ts["end"] + iv["start"]}
                )

    if speech_timestamps:
        merge_gap_samples = int(sample_rate * (merge_gap_ms / 1000.0))
        merged = [speech_timestamps[0]]
        for ts in speech_timestamps[1:]:
            gap = ts["start"] - merged[-1]["end"]
            if gap <= merge_gap_samples:
                merged[-1]["end"] = max(merged[-1]["end"], ts["end"])
            else:
                merged.append(ts)
        speech_timestamps = merged

    if speech_timestamps:
        target_min_samples = int(sample_rate * target_min_s)
        target_max_samples = int(sample_rate * target_max_s)
        hard_max_samples = int(sample_rate * hard_max_s)

        chunks = []
        cur_start = speech_timestamps[0]["start"]
        cur_end = speech_timestamps[0]["end"]

        for ts in speech_timestamps[1:]:
            next_start = ts["start"]
            next_end = ts["end"]
            gap = next_start - cur_end

            if (next_end - cur_start) <= target_max_samples:
                cur_end = next_end
                continue

            cur_len = cur_end - cur_start
            if cur_len < target_min_samples and (next_end - cur_start) <= hard_max_samples:
                cur_end = next_end
                continue
            if cur_len > hard_max_samples and hard_max_samples > 0:
                overlap_samples = int(sample_rate * overlap_s)
                split = cur_start + hard_max_samples
                chunks.append({"start": cur_start, "end": split})
                cur_start = max(cur_start, split - overlap_samples)
                cur_end = next_end
            else:
                split = cur_end + gap // 2
                chunks.append({"start": cur_start, "end": split})
                cur_start = split
                cur_end = next_end

        chunks.append({"start": cur_start, "end": cur_end})

        # Enforce hard max even when VAD returns a single long interval (e.g., music).
        enforced = []
        overlap_samples = int(sample_rate * overlap_s)
        step = hard_max_samples - overlap_samples if hard_max_samples > overlap_samples else hard_max_samples
        for ch in chunks:
            ch_start = ch["start"]
            ch_end = ch["end"]
            if hard_max_samples > 0 and (ch_end - ch_start) > hard_max_samples:
                cur = ch_start
                while cur < ch_end:
                    seg_end = min(cur + hard_max_samples, ch_end)
                    enforced.append({"start": cur, "end": seg_end})
                    if seg_end >= ch_end:
                        break
                    cur = max(cur, seg_end - overlap_samples)
            else:
                enforced.append(ch)

        speech_timestamps = enforced

    chunk_paths = []
    if not speech_timestamps:
        return chunk_paths

    source_waveform = chunk_waveform if chunk_waveform is not None else audio
    source_rate = chunk_sample_rate if chunk_sample_rate is not None else sample_rate
    pad_samples = int(source_rate * (speech_pad_ms / 1000.0))
    max_len = source_waveform.shape[-1]

    for idx, ts in enumerate(speech_timestamps, start=1):
        start_s = ts["start"] / sample_rate
        end_s = ts["end"] / sample_rate
        start = max(0, int(start_s * source_rate) - pad_samples)
        end = min(max_len, int(end_s * source_rate) + pad_samples)
        chunk = source_waveform[start:end]
        out_path = out_dir / f"chunk_{idx:04d}_{start_s:.3f}-{end_s:.3f}.wav"
        save_audio(str(out_path), chunk, sampling_rate=source_rate)
        chunk_paths.append(out_path)

    return chunk_paths


def run_vad_chunks_in_memory(
    input_path: Path,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    merge_gap_ms: int,
    target_min_s: float,
    target_max_s: float,
    hard_max_s: float,
    overlap_s: float,
    speech_pad_ms: int,
) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    audio = _load_audio_mono_16k(input_path, sample_rate)

    return run_vad_chunks_in_memory_from_waveform(
        waveform=audio,
        sample_rate=sample_rate,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        merge_gap_ms=merge_gap_ms,
        target_min_s=target_min_s,
        target_max_s=target_max_s,
        hard_max_s=hard_max_s,
        overlap_s=overlap_s,
        speech_pad_ms=speech_pad_ms,
    )


def run_vad_chunks_in_memory_from_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    merge_gap_ms: int,
    target_min_s: float,
    target_max_s: float,
    hard_max_s: float,
    overlap_s: float,
    speech_pad_ms: int,
    chunk_waveform: torch.Tensor | None = None,
    chunk_sample_rate: int | None = None,
    energy_gate_override: bool | None = None,
    energy_overrides: dict | None = None,
) -> list[dict]:
    device = _resolve_vad_device()
    energy_overrides = energy_overrides or {}
    if energy_gate_override is None:
        use_energy_gate = energy_overrides.get(
            "energy_gate",
            os.environ.get("VAD_ENERGY_GATE", "0") == "1",
        )
    else:
        use_energy_gate = bool(energy_gate_override)
    energy_frame_ms = int(energy_overrides.get("energy_frame_ms", os.environ.get("VAD_ENERGY_FRAME_MS", "100")))
    energy_threshold_db = float(energy_overrides.get("energy_db", os.environ.get("VAD_ENERGY_DB", "-40")))
    energy_min_active_ms = int(
        energy_overrides.get("energy_min_active_ms", os.environ.get("VAD_ENERGY_MIN_ACTIVE_MS", "300"))
    )
    energy_merge_gap_ms = int(
        energy_overrides.get("energy_merge_gap_ms", os.environ.get("VAD_ENERGY_MERGE_GAP_MS", "500"))
    )
    energy_active_ratio_skip = float(
        energy_overrides.get("energy_active_skip", os.environ.get("VAD_ENERGY_ACTIVE_SKIP", "0.7"))
    )
    uniform_chunk_s = float(energy_overrides.get("uniform_chunk_s", os.environ.get("VAD_UNIFORM_CHUNK_S", "45")))
    uniform_overlap_s = float(
        energy_overrides.get("uniform_overlap_s", os.environ.get("VAD_UNIFORM_OVERLAP_S", "0.5"))
    )

    model, utils = _load_vad(device)
    (get_speech_timestamps, _, _, _, _) = utils

    audio = waveform
    if use_energy_gate:
        print(
            "energy gate config: "
            f"db={energy_threshold_db}, frame_ms={energy_frame_ms}, "
            f"min_active_ms={energy_min_active_ms}, merge_gap_ms={energy_merge_gap_ms}, "
            f"skip_ratio={energy_active_ratio_skip}, chunk_s={uniform_chunk_s}, "
            f"overlap_s={uniform_overlap_s}"
        )
        intervals, active_ratio = _energy_gate_intervals(
            waveform=audio.detach().cpu(),
            sample_rate=sample_rate,
            frame_ms=energy_frame_ms,
            threshold_db=energy_threshold_db,
            min_active_ms=energy_min_active_ms,
            pad_ms=speech_pad_ms,
            merge_gap_ms=energy_merge_gap_ms,
        )
        if active_ratio >= energy_active_ratio_skip:
            speech_timestamps = _uniform_intervals(
                total_samples=audio.shape[-1],
                sample_rate=sample_rate,
                chunk_s=uniform_chunk_s,
                overlap_s=uniform_overlap_s,
            )
            print(
                f"energy gate skip (ratio={active_ratio:.2f}) -> "
                f"{len(speech_timestamps)} uniform intervals"
            )
        else:
            print(f"energy gate enabled: {len(intervals)} intervals (ratio={active_ratio:.2f})")
            speech_timestamps = []
            for iv in intervals:
                segment = audio[iv["start"] : iv["end"]]
                audio_vad = segment.to(device) if device == "cuda" else segment
                seg_ts = get_speech_timestamps(
                    audio_vad,
                    model,
                    sampling_rate=sample_rate,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_ms,
                    min_silence_duration_ms=min_silence_ms,
                    speech_pad_ms=speech_pad_ms,
                )
                for ts in seg_ts:
                    speech_timestamps.append(
                        {"start": ts["start"] + iv["start"], "end": ts["end"] + iv["start"]}
                    )
    else:
        intervals = [{"start": 0, "end": audio.shape[-1]}]
        speech_timestamps = []
        for iv in intervals:
            segment = audio[iv["start"] : iv["end"]]
            audio_vad = segment.to(device) if device == "cuda" else segment
            seg_ts = get_speech_timestamps(
                audio_vad,
                model,
                sampling_rate=sample_rate,
                threshold=threshold,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
            )
            for ts in seg_ts:
                speech_timestamps.append(
                    {"start": ts["start"] + iv["start"], "end": ts["end"] + iv["start"]}
                )

    if speech_timestamps:
        merge_gap_samples = int(sample_rate * (merge_gap_ms / 1000.0))
        merged = [speech_timestamps[0]]
        for ts in speech_timestamps[1:]:
            gap = ts["start"] - merged[-1]["end"]
            if gap <= merge_gap_samples:
                merged[-1]["end"] = max(merged[-1]["end"], ts["end"])
            else:
                merged.append(ts)
        speech_timestamps = merged

    if speech_timestamps:
        target_min_samples = int(sample_rate * target_min_s)
        target_max_samples = int(sample_rate * target_max_s)
        hard_max_samples = int(sample_rate * hard_max_s)

        chunks = []
        cur_start = speech_timestamps[0]["start"]
        cur_end = speech_timestamps[0]["end"]

        for ts in speech_timestamps[1:]:
            next_start = ts["start"]
            next_end = ts["end"]
            gap = next_start - cur_end

            if (next_end - cur_start) <= target_max_samples:
                cur_end = next_end
                continue

            cur_len = cur_end - cur_start
            if cur_len < target_min_samples and (next_end - cur_start) <= hard_max_samples:
                cur_end = next_end
                continue

            if cur_len > hard_max_samples and hard_max_samples > 0:
                overlap_samples = int(sample_rate * overlap_s)
                split = cur_start + hard_max_samples
                chunks.append({"start": cur_start, "end": split})
                cur_start = max(cur_start, split - overlap_samples)
                cur_end = next_end
            else:
                split = cur_end + gap // 2
                chunks.append({"start": cur_start, "end": split})
                cur_start = split
                cur_end = next_end

        chunks.append({"start": cur_start, "end": cur_end})

        enforced = []
        overlap_samples = int(sample_rate * overlap_s)
        for ch in chunks:
            ch_start = ch["start"]
            ch_end = ch["end"]
            if hard_max_samples > 0 and (ch_end - ch_start) > hard_max_samples:
                cur = ch_start
                while cur < ch_end:
                    seg_end = min(cur + hard_max_samples, ch_end)
                    enforced.append({"start": cur, "end": seg_end})
                    if seg_end >= ch_end:
                        break
                    cur = max(cur, seg_end - overlap_samples)
            else:
                enforced.append(ch)

        speech_timestamps = enforced

    chunks = []
    if not speech_timestamps:
        return chunks

    source_waveform = chunk_waveform if chunk_waveform is not None else waveform
    source_rate = chunk_sample_rate if chunk_sample_rate is not None else sample_rate
    pad_samples = int(source_rate * (speech_pad_ms / 1000.0))
    max_len = source_waveform.shape[-1]

    for ts in speech_timestamps:
        start_s = ts["start"] / sample_rate
        end_s = ts["end"] / sample_rate
        start = max(0, int(start_s * source_rate) - pad_samples)
        end = min(max_len, int(end_s * source_rate) + pad_samples)
        chunk = source_waveform[start:end].detach().cpu()
        chunks.append(
            {
                "waveform": chunk,
                "sample_rate": source_rate,
                "start_s": start_s,
                "end_s": end_s,
            }
        )

    return chunks


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)

    chunk_paths = run_vad_chunks(
        input_path=input_path,
        out_dir=out_dir,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        merge_gap_ms=args.merge_gap_ms,
        target_min_s=args.target_min_s,
        target_max_s=args.target_max_s,
        hard_max_s=args.hard_max_s,
        overlap_s=args.overlap_s,
        speech_pad_ms=args.speech_pad_ms,
    )

    if not chunk_paths:
        print("No speech segments detected.")
        return 0

    print(f"Wrote {len(chunk_paths)} chunks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
