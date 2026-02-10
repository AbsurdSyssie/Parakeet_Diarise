#!/usr/bin/env python3
"""ASR API: VAD chunking + Parakeet transcription + optional diarization."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import os
from typing import Optional, Literal

import torchaudio
import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api_response import build_response
from asr_merge import parse_chunk_filename
from chunk_transcribe import (
    transcribe_chunks_in_memory_mode,
    transcribe_chunks_with_model_mode,
    _patch_transcribe_dataloader_no_lhotse,
)
from diarize_align import assign_speakers, group_words_into_segments
from vad_chunk import (
    run_vad_chunks_from_waveform,
    run_vad_chunks_in_memory_from_waveform,
)

app = FastAPI()
_ASR_MODEL = None
_DIAR_MODEL = None
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", "16000"))


def _get_env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _get_env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _get_env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _disable_cuda_graphs(asr_model, verbose: bool = False) -> bool:
    disabled = False
    if hasattr(asr_model, "cfg") and hasattr(asr_model.cfg, "decoding"):
        if hasattr(asr_model.cfg.decoding, "greedy"):
            from omegaconf import open_dict

            with open_dict(asr_model.cfg.decoding.greedy):
                asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
                if verbose:
                    print("  Disabled CUDA graphs in cfg.decoding.greedy")
                disabled = True

    if hasattr(asr_model, "decoding") and hasattr(asr_model.decoding, "decoding"):
        dc = asr_model.decoding.decoding
        if hasattr(dc, "use_cuda_graph_decoder"):
            dc.use_cuda_graph_decoder = False
            if verbose:
                print("  Disabled use_cuda_graph_decoder in decoding.decoding")
            disabled = True
        if hasattr(dc, "decoding_computer"):
            dcomp = dc.decoding_computer
            if hasattr(dcomp, "allow_cuda_graphs"):
                dcomp.allow_cuda_graphs = False
            if hasattr(dcomp, "disable_cuda_graphs"):
                dcomp.disable_cuda_graphs()
            if hasattr(dcomp, "cuda_graphs_mode"):
                dcomp.cuda_graphs_mode = None
            if verbose:
                print("  Disabled CUDA graphs in decoding_computer")
            disabled = True

    for attr_name in ["joint", "joint_0", "joint_1", "joint_2", "joint_3"]:
        if hasattr(asr_model, attr_name):
            joint = getattr(asr_model, attr_name)
            if hasattr(joint, "decoding") and hasattr(joint.decoding, "decoding"):
                jdc = joint.decoding.decoding
                if hasattr(jdc, "use_cuda_graph_decoder"):
                    jdc.use_cuda_graph_decoder = False
                if hasattr(jdc, "decoding_computer"):
                    jdcomp = jdc.decoding_computer
                    if hasattr(jdcomp, "allow_cuda_graphs"):
                        jdcomp.allow_cuda_graphs = False
                    if hasattr(jdcomp, "cuda_graphs_mode"):
                        jdcomp.cuda_graphs_mode = None
                if verbose:
                    print(f"  Disabled CUDA graphs in {attr_name}")
                disabled = True

    return disabled


def cuda_mem():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def _chunk_debug_from_in_memory(chunks: list[dict], duration: float, pad_s: float) -> list[dict]:
    out = []
    for idx, chunk in enumerate(chunks, start=1):
        start_s = float(chunk["start_s"])
        end_s = float(chunk["end_s"])
        effective_start_s = max(0.0, start_s - pad_s)
        effective_end_s = min(duration, end_s + pad_s)
        out.append(
            {
                "index": idx,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "effective_start_s": effective_start_s,
                "effective_end_s": effective_end_s,
                "effective_duration_s": effective_end_s - effective_start_s,
                "sample_rate": int(chunk.get("sample_rate", 0)),
                "num_samples": int(chunk["waveform"].shape[-1]),
            }
        )
    return out


def _log_chunk_durations(
    chunks: list[dict],
    duration: float,
    pad_s: float,
    trace_id: str,
) -> None:
    if not chunks:
        return
    sample_rate = int(chunks[0].get("sample_rate", 0))
    if sample_rate <= 0:
        return
    for label, chunk in (("first", chunks[0]), ("last", chunks[-1])):
        start_s = float(chunk["start_s"])
        end_s = float(chunk["end_s"])
        effective_start_s = max(0.0, start_s - pad_s)
        effective_end_s = min(duration, end_s + pad_s)
        expected = effective_end_s - effective_start_s
        actual = float(chunk["waveform"].shape[-1]) / sample_rate
        delta = actual - expected
        print(
            f"[{trace_id}] chunk_duration {label} "
            f"expected_s={expected:.3f} actual_s={actual:.3f} delta_s={delta:.3f}"
        )


def _chunk_debug_from_files(
    chunk_paths: list[Path],
    duration: float,
    pad_s: float,
) -> list[dict]:
    out = []
    for idx, path in enumerate(sorted(chunk_paths), start=1):
        chunk_id, start_s, end_s = parse_chunk_filename(path.name)
        effective_start_s = max(0.0, start_s - pad_s)
        effective_end_s = min(duration, end_s + pad_s)
        out.append(
            {
                "index": idx,
                "chunk_id": chunk_id,
                "filename": path.name,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "effective_start_s": effective_start_s,
                "effective_end_s": effective_end_s,
                "effective_duration_s": effective_end_s - effective_start_s,
            }
        )
    return out


@app.get("/health")
def health():
    return {
        "ok": True,
        "asr_loaded": _ASR_MODEL is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_mem": cuda_mem(),
    }


@app.on_event("startup")
def _startup_load_model() -> None:
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    asr_model = asr_model.to(device)
    _patch_transcribe_dataloader_no_lhotse(asr_model)
    if _get_env_bool("DISABLE_CUDA_GRAPHS", "0"):
        if _disable_cuda_graphs(asr_model, verbose=True):
            print("  CUDA graphs disabled for decoding")

    # Warmup to amortize first-request overhead.
    dummy = torch.zeros(16000, dtype=torch.float32)
    asr_model.transcribe(
        [dummy],
        timestamps=False,
        verbose=False,
        batch_size=1,
        num_workers=0,
    )

    _ASR_MODEL = asr_model


def _get_diar_model():
    global _DIAR_MODEL
    if _DIAR_MODEL is not None:
        return _DIAR_MODEL

    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1"
    )
    model.eval()
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    model.sortformer_modules.chunk_len = 340
    model.sortformer_modules.chunk_right_context = 40
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.spkcache_update_period = 300

    _DIAR_MODEL = model
    return model


def _normalize_speaker(label: str) -> str:
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


def _run_sortformer(waveform: torch.Tensor, sr: int, tmpdir: Path) -> list[dict]:
    model = _get_diar_model()
    if waveform.dim() == 1:
        wav = waveform.unsqueeze(0)
    else:
        wav = waveform
    mono_path = tmpdir / "diarize_mono16k.wav"
    torchaudio.save(str(mono_path), wav, sr)
    predicted_segments = model.diarize(audio=[str(mono_path)], batch_size=1)
    raw = predicted_segments[0] if predicted_segments else []
    turns = []
    for seg in raw:
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
                continue
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            start = float(seg[0])
            end = float(seg[1])
            speaker = seg[2]
        else:
            continue
        turns.append({"start": start, "end": end, "speaker": _normalize_speaker(speaker)})
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    response_format: Literal["verbose_json"] = Form(
        "verbose_json",
        description="Only supported response format.",
    ),
    diarization: bool = Form(
        False,
        description="Enable Sortformer diarization. Requires timestamps=word.",
    ),
    timestamps: Literal["word", "segment", "none"] = Form(
        "word",
        description="word, segment, or none. word required for diarization.",
    ),
    language: str = Form(
        "en",
        description="Currently only 'en' is supported.",
    ),
    chunk_mode: Literal["memory", "file"] = Form(
        "memory",
        description="memory (default) or file (writes chunk WAVs to disk).",
    ),
    chunk_only: bool = Form(
        False,
        description="Return VAD chunks only; skips ASR.",
    ),
    trace_audio: bool = Form(
        False,
        description="Emit per-request trace logs for ingest/decode/VAD.",
    ),
    force_vad: Literal["off", "on"] = Form(
        "on",
        description="off or on (default; force Silero VAD, ignore energy gate).",
    ),
    vad_sample_rate: Optional[int] = Form(
        None,
        description="Override VAD sample rate (default env VAD_SAMPLE_RATE=16000).",
    ),
    vad_threshold: Optional[float] = Form(
        None,
        description="Silero threshold (default env VAD_THRESHOLD=0.30).",
    ),
    vad_min_speech_ms: Optional[int] = Form(
        None,
        description="Minimum speech duration (ms). Default env VAD_MIN_SPEECH_MS=150.",
    ),
    vad_min_silence_ms: Optional[int] = Form(
        None,
        description="Minimum silence duration (ms). Default env VAD_MIN_SILENCE_MS=220.",
    ),
    vad_merge_gap_ms: Optional[int] = Form(
        None,
        description="Gap to merge speech segments (ms). Default env VAD_MERGE_GAP_MS=200.",
    ),
    vad_target_min_s: Optional[float] = Form(
        None,
        description="Target min chunk length (s). Default env VAD_TARGET_MIN_S=10.0.",
    ),
    vad_target_max_s: Optional[float] = Form(
        None,
        description="Target max chunk length (s). Default env VAD_TARGET_MAX_S=20.0.",
    ),
    vad_hard_max_s: Optional[float] = Form(
        None,
        description="Hard max chunk length (s). Default env VAD_HARD_MAX_S=30.0.",
    ),
    vad_overlap_s: Optional[float] = Form(
        None,
        description="Chunk overlap (s). Default env VAD_OVERLAP_S=1.0.",
    ),
    vad_speech_pad_ms: Optional[int] = Form(
        None,
        description="Pad speech edges (ms). Default env VAD_SPEECH_PAD_MS=250.",
    ),
    vad_energy_gate: Optional[bool] = Form(
        None,
        description="Enable/disable energy gate (default env VAD_ENERGY_GATE=0).",
    ),
    vad_energy_db: Optional[float] = Form(
        None,
        description="Energy gate threshold (dB). Default env VAD_ENERGY_DB=-35.",
    ),
    vad_energy_frame_ms: Optional[int] = Form(
        None,
        description="Energy gate frame size (ms). Default env VAD_ENERGY_FRAME_MS=100.",
    ),
    vad_energy_min_active_ms: Optional[int] = Form(
        None,
        description="Energy gate min active duration (ms). Default env VAD_ENERGY_MIN_ACTIVE_MS=500.",
    ),
    vad_energy_merge_gap_ms: Optional[int] = Form(
        None,
        description="Energy gate merge gap (ms). Default env VAD_ENERGY_MERGE_GAP_MS=800.",
    ),
    vad_energy_active_skip: Optional[float] = Form(
        None,
        description="Skip Silero if active ratio >= this. Default env VAD_ENERGY_ACTIVE_SKIP=0.85.",
    ),
    vad_uniform_chunk_s: Optional[float] = Form(
        None,
        description="Uniform chunk length (s) when energy gate skips. Default env VAD_UNIFORM_CHUNK_S=30.",
    ),
    vad_uniform_overlap_s: Optional[float] = Form(
        None,
        description="Uniform chunk overlap (s). Default env VAD_UNIFORM_OVERLAP_S=1.0.",
    ),
):
    if response_format != "verbose_json":
        raise HTTPException(status_code=400, detail="Only verbose_json is supported")
    if timestamps not in {"word", "segment", "none"}:
        raise HTTPException(
            status_code=400,
            detail="timestamps must be 'word', 'segment', or 'none'",
        )
    if chunk_mode not in {"memory", "file"}:
        raise HTTPException(status_code=400, detail="chunk_mode must be 'memory' or 'file'")
    if diarization and timestamps != "word":
        raise HTTPException(
            status_code=400,
            detail="diarization requires timestamps=word",
        )
    if force_vad not in {"off", "on"}:
        raise HTTPException(
            status_code=400,
            detail="force_vad must be 'off' or 'on'",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        start_req = time.perf_counter()
        trace_id = None
        tmp_path = Path(tmpdir) / file.filename
        file_start = time.perf_counter()
        file_bytes = await file.read()
        tmp_path.write_bytes(file_bytes)
        file_elapsed = time.perf_counter() - file_start
        if trace_audio:
            trace_id = f"trace-{int(start_req * 1000)}"
            print(
                f"[{trace_id}] ingest file={file.filename!r} bytes={len(file_bytes)} "
                f"write_s={file_elapsed:.3f}"
            )

        decode_start = time.perf_counter()
        waveform, sr = torchaudio.load(str(tmp_path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)
        decode_elapsed = time.perf_counter() - decode_start
        duration = waveform.shape[-1] / sr if waveform.numel() else 0.0
        if trace_audio:
            channels = 1 if waveform.dim() == 1 else waveform.size(0)
            num_samples = int(waveform.shape[-1])
            print(
                f"[{trace_id}] decode sr={sr} channels={channels} "
                f"samples={num_samples} duration_s={duration:.3f} "
                f"decode_s={decode_elapsed:.3f}"
            )

        vad_waveform = waveform
        vad_sr = vad_sample_rate if vad_sample_rate is not None else VAD_SAMPLE_RATE
        if vad_sr != sr:
            vad_waveform = torchaudio.functional.resample(waveform, sr, vad_sr)
        if trace_audio:
            vad_samples = int(vad_waveform.shape[-1])
            vad_duration = vad_samples / vad_sr if vad_samples else 0.0
            print(
                f"[{trace_id}] vad_resample sr={vad_sr} "
                f"samples={vad_samples} duration_s={vad_duration:.3f}"
            )

        vad_threshold = vad_threshold if vad_threshold is not None else _get_env_float("VAD_THRESHOLD", "0.30")
        vad_min_speech_ms = (
            vad_min_speech_ms if vad_min_speech_ms is not None else _get_env_int("VAD_MIN_SPEECH_MS", "150")
        )
        vad_min_silence_ms = (
            vad_min_silence_ms if vad_min_silence_ms is not None else _get_env_int("VAD_MIN_SILENCE_MS", "220")
        )
        vad_merge_gap_ms = (
            vad_merge_gap_ms if vad_merge_gap_ms is not None else _get_env_int("VAD_MERGE_GAP_MS", "200")
        )
        vad_target_min_s = (
            vad_target_min_s if vad_target_min_s is not None else _get_env_float("VAD_TARGET_MIN_S", "10.0")
        )
        vad_target_max_s = (
            vad_target_max_s if vad_target_max_s is not None else _get_env_float("VAD_TARGET_MAX_S", "20.0")
        )
        vad_hard_max_s = (
            vad_hard_max_s if vad_hard_max_s is not None else _get_env_float("VAD_HARD_MAX_S", "30.0")
        )
        vad_overlap_s = (
            vad_overlap_s if vad_overlap_s is not None else _get_env_float("VAD_OVERLAP_S", "1.0")
        )
        vad_speech_pad_ms = (
            vad_speech_pad_ms if vad_speech_pad_ms is not None else _get_env_int("VAD_SPEECH_PAD_MS", "250")
        )
        pad_s = vad_speech_pad_ms / 1000.0
        energy_gate_override = None if force_vad == "off" else False
        energy_overrides = {}
        if vad_energy_gate is not None:
            energy_overrides["energy_gate"] = vad_energy_gate
        if vad_energy_db is not None:
            energy_overrides["energy_db"] = vad_energy_db
        if vad_energy_frame_ms is not None:
            energy_overrides["energy_frame_ms"] = vad_energy_frame_ms
        if vad_energy_min_active_ms is not None:
            energy_overrides["energy_min_active_ms"] = vad_energy_min_active_ms
        if vad_energy_merge_gap_ms is not None:
            energy_overrides["energy_merge_gap_ms"] = vad_energy_merge_gap_ms
        if vad_energy_active_skip is not None:
            energy_overrides["energy_active_skip"] = vad_energy_active_skip
        if vad_uniform_chunk_s is not None:
            energy_overrides["uniform_chunk_s"] = vad_uniform_chunk_s
        if vad_uniform_overlap_s is not None:
            energy_overrides["uniform_overlap_s"] = vad_uniform_overlap_s

        print(
            "vad settings: "
            f"vad_sr={vad_sr}, thr={vad_threshold}, min_speech_ms={vad_min_speech_ms}, "
            f"min_silence_ms={vad_min_silence_ms}, merge_gap_ms={vad_merge_gap_ms}, "
            f"target_min_s={vad_target_min_s}, target_max_s={vad_target_max_s}, "
            f"hard_max_s={vad_hard_max_s}, overlap_s={vad_overlap_s}, "
            f"speech_pad_ms={vad_speech_pad_ms}, force_vad={force_vad}, "
            f"energy_gate={energy_overrides.get('energy_gate')}, "
            f"energy_db={energy_overrides.get('energy_db')}, "
            f"energy_frame_ms={energy_overrides.get('energy_frame_ms')}, "
            f"energy_min_active_ms={energy_overrides.get('energy_min_active_ms')}, "
            f"energy_merge_gap_ms={energy_overrides.get('energy_merge_gap_ms')}, "
            f"energy_active_skip={energy_overrides.get('energy_active_skip')}, "
            f"uniform_chunk_s={energy_overrides.get('uniform_chunk_s')}, "
            f"uniform_overlap_s={energy_overrides.get('uniform_overlap_s')}"
        )

        print(f"cuda mem before vad: {cuda_mem()}")
        if chunk_mode == "memory":
            vad_start = time.perf_counter()
            chunks = run_vad_chunks_in_memory_from_waveform(
                waveform=vad_waveform,
                sample_rate=vad_sr,
                threshold=vad_threshold,
                min_speech_ms=vad_min_speech_ms,
                min_silence_ms=vad_min_silence_ms,
                merge_gap_ms=vad_merge_gap_ms,
                target_min_s=vad_target_min_s,
                target_max_s=vad_target_max_s,
                hard_max_s=vad_hard_max_s,
                overlap_s=vad_overlap_s,
                speech_pad_ms=vad_speech_pad_ms,
                chunk_waveform=waveform,
                chunk_sample_rate=sr,
                energy_gate_override=energy_gate_override,
                energy_overrides=energy_overrides,
            )
            vad_elapsed = time.perf_counter() - vad_start

            if not chunks:
                if trace_audio:
                    print(f"[{trace_id}] vad_chunks count=0 mode=memory")
                if chunk_only:
                    return {
                        "ok": True,
                        "chunk_only": True,
                        "chunk_mode": chunk_mode,
                        "duration": duration,
                        "sample_rate": sr,
                        "vad_sample_rate": vad_sr,
                        "vad_params": {
                            "threshold": vad_threshold,
                            "min_speech_ms": vad_min_speech_ms,
                            "min_silence_ms": vad_min_silence_ms,
                            "merge_gap_ms": vad_merge_gap_ms,
                            "target_min_s": vad_target_min_s,
                            "target_max_s": vad_target_max_s,
                            "hard_max_s": vad_hard_max_s,
                            "overlap_s": vad_overlap_s,
                            "speech_pad_ms": vad_speech_pad_ms,
                            "force_vad": force_vad,
                            "energy_gate": vad_energy_gate,
                            "energy_db": vad_energy_db,
                            "energy_frame_ms": vad_energy_frame_ms,
                            "energy_min_active_ms": vad_energy_min_active_ms,
                            "energy_merge_gap_ms": vad_energy_merge_gap_ms,
                            "energy_active_skip": vad_energy_active_skip,
                            "uniform_chunk_s": vad_uniform_chunk_s,
                            "uniform_overlap_s": vad_uniform_overlap_s,
                        },
                        "chunks": [],
                    }
                return build_response([], [], language=language, duration=duration, text="")

            if trace_audio and chunks:
                first = chunks[0]
                last = chunks[-1]
                print(
                    f"[{trace_id}] vad_chunks count={len(chunks)} mode=memory "
                    f"first={first['start_s']:.2f}-{first['end_s']:.2f} "
                    f"last={last['start_s']:.2f}-{last['end_s']:.2f}"
                )
                _log_chunk_durations(chunks, duration, pad_s, trace_id)

            if chunk_only:
                return {
                    "ok": True,
                    "chunk_only": True,
                    "chunk_mode": chunk_mode,
                    "duration": duration,
                    "sample_rate": sr,
                    "vad_sample_rate": vad_sr,
                        "vad_params": {
                            "threshold": vad_threshold,
                            "min_speech_ms": vad_min_speech_ms,
                            "min_silence_ms": vad_min_silence_ms,
                            "merge_gap_ms": vad_merge_gap_ms,
                            "target_min_s": vad_target_min_s,
                            "target_max_s": vad_target_max_s,
                            "hard_max_s": vad_hard_max_s,
                            "overlap_s": vad_overlap_s,
                            "speech_pad_ms": vad_speech_pad_ms,
                            "force_vad": force_vad,
                        },
                        "chunks": _chunk_debug_from_in_memory(chunks, duration, pad_s),
                    }

            start_asr = time.perf_counter()
            result = transcribe_chunks_in_memory_mode(
                asr_model=_ASR_MODEL,
                chunks=chunks,
                batch_size=4,
                timestamps=timestamps,
                pad_left_s=pad_s,
            )
            elapsed = time.perf_counter() - start_asr
            print(
                f"file write {file_elapsed:.2f}s; decode {decode_elapsed:.2f}s; "
                f"vad {vad_elapsed:.2f}s; chunks transcribed in {elapsed:.2f}s"
            )
            print(f"cuda mem after asr: {cuda_mem()}")
        else:
            chunk_dir = Path(tmpdir) / "chunks"
            vad_start = time.perf_counter()
            chunk_paths = run_vad_chunks_from_waveform(
                waveform=vad_waveform,
                out_dir=chunk_dir,
                sample_rate=vad_sr,
                threshold=vad_threshold,
                min_speech_ms=vad_min_speech_ms,
                min_silence_ms=vad_min_silence_ms,
                merge_gap_ms=vad_merge_gap_ms,
                target_min_s=vad_target_min_s,
                target_max_s=vad_target_max_s,
                hard_max_s=vad_hard_max_s,
                overlap_s=vad_overlap_s,
                speech_pad_ms=vad_speech_pad_ms,
                chunk_waveform=waveform,
                chunk_sample_rate=sr,
                energy_gate_override=energy_gate_override,
                energy_overrides=energy_overrides,
            )
            vad_elapsed = time.perf_counter() - vad_start

            if not chunk_paths:
                if trace_audio:
                    print(f"[{trace_id}] vad_chunks count=0 mode=file")
                if chunk_only:
                    return {
                        "ok": True,
                        "chunk_only": True,
                        "chunk_mode": chunk_mode,
                        "duration": duration,
                        "sample_rate": sr,
                        "vad_sample_rate": vad_sr,
                        "vad_params": {
                            "threshold": vad_threshold,
                            "min_speech_ms": vad_min_speech_ms,
                            "min_silence_ms": vad_min_silence_ms,
                            "merge_gap_ms": vad_merge_gap_ms,
                            "target_min_s": vad_target_min_s,
                            "target_max_s": vad_target_max_s,
                            "hard_max_s": vad_hard_max_s,
                            "overlap_s": vad_overlap_s,
                            "speech_pad_ms": vad_speech_pad_ms,
                            "force_vad": force_vad,
                        },
                        "chunks": [],
                    }
                return build_response([], [], language=language, duration=duration, text="")

            if trace_audio and chunk_paths:
                first_id, first_start, first_end = parse_chunk_filename(chunk_paths[0].name)
                last_id, last_start, last_end = parse_chunk_filename(chunk_paths[-1].name)
                print(
                    f"[{trace_id}] vad_chunks count={len(chunk_paths)} mode=file "
                    f"first={first_start:.2f}-{first_end:.2f} "
                    f"last={last_start:.2f}-{last_end:.2f}"
                )

            if chunk_only:
                return {
                    "ok": True,
                    "chunk_only": True,
                    "chunk_mode": chunk_mode,
                    "duration": duration,
                    "sample_rate": sr,
                    "vad_sample_rate": vad_sr,
                        "vad_params": {
                            "threshold": vad_threshold,
                            "min_speech_ms": vad_min_speech_ms,
                            "min_silence_ms": vad_min_silence_ms,
                            "merge_gap_ms": vad_merge_gap_ms,
                            "target_min_s": vad_target_min_s,
                            "target_max_s": vad_target_max_s,
                            "hard_max_s": vad_hard_max_s,
                            "overlap_s": vad_overlap_s,
                            "speech_pad_ms": vad_speech_pad_ms,
                            "force_vad": force_vad,
                        },
                        "chunks": _chunk_debug_from_files(chunk_paths, duration, pad_s),
                    }

            start_asr = time.perf_counter()
            result = transcribe_chunks_with_model_mode(
                asr_model=_ASR_MODEL,
                chunk_dir=chunk_dir,
                pad_left_s=pad_s,
                pad_right_s=pad_s,
                batch_size=4,
                timestamps=timestamps,
            )
            elapsed = time.perf_counter() - start_asr
            print(
                f"file write {file_elapsed:.2f}s; decode {decode_elapsed:.2f}s; "
                f"vad {vad_elapsed:.2f}s; chunks transcribed in {elapsed:.2f}s"
            )
            print(f"cuda mem after asr: {cuda_mem()}")

        text_override = result["text"]
        if timestamps == "word":
            words = result["words"]
        else:
            words = []

        if diarization:
            start_diar = time.perf_counter()
            turns = _run_sortformer(waveform, sr, Path(tmpdir))
            diar_elapsed = time.perf_counter() - start_diar
            print(f"diarization in {diar_elapsed:.2f}s")
            words_with_speaker = assign_speakers(words, turns)
        else:
            words_with_speaker = [
                {**w, "speaker": w.get("speaker") or "UNKNOWN"} for w in words
            ]

        start_merge = time.perf_counter()
        if timestamps == "word":
            segments = group_words_into_segments(words_with_speaker)
            for idx, seg in enumerate(segments):
                seg.setdefault("id", idx)
            text_override = None
        elif timestamps == "segment":
            segments = result["segments"]
            for idx, seg in enumerate(segments):
                seg.setdefault("id", idx)
                seg.setdefault("speaker", "UNKNOWN")
        else:
            segments = []
        merge_elapsed = time.perf_counter() - start_merge
        total_elapsed = time.perf_counter() - start_req
        print(f"merge in {merge_elapsed:.2f}s; total request {total_elapsed:.2f}s")
        return build_response(
            words_with_speaker,
            segments,
            language=language,
            duration=duration,
            text=text_override,
        )
