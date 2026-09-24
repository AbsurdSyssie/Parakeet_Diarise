#!/usr/bin/env python3
"""ASR API: VAD chunking + runtime-selectable ASR + optional diarization."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Optional

import torchaudio
import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile

from api_response import build_response
from asr_merge import parse_chunk_filename
from chunk_transcribe import (
    transcribe_chunks_in_memory_mode,
    transcribe_chunks_with_model_mode,
)
from diarization import DIARIZATION_MODEL, diarize_waveform, is_diarizer_loaded
from diarize_align import assign_speakers, group_words_into_segments
from model_manager import (
    MODEL_MANAGER,
    ModelSwitchError,
    ModelUnavailableError,
    cuda_mem,
)
from model_registry import (
    MODEL_REGISTRY,
    config_from_entry,
    effective_timestamps,
    find_model_entry,
    initial_config,
    public_config,
    public_model_entry,
    validate_request,
)
from settings import SETTINGS, env_float, env_int
from vad_chunk import run_vad_chunks_from_waveform, run_vad_chunks_in_memory_from_waveform


app = FastAPI()
STARTUP_CONFIG = initial_config()


def _load_audio_waveform(path: str) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(path)
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise

        import soundfile as sf

        data, sr = sf.read(path, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data.T).contiguous()
        return waveform, int(sr)


def _active_model_or_503():
    try:
        return MODEL_MANAGER.require_active()
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _validate_request_for_config(config, *, diarization: bool, timestamps: str) -> None:
    try:
        validate_request(config, diarization=diarization, timestamps=timestamps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def _chunk_debug_from_files(chunk_paths: list[Path], duration: float, pad_s: float) -> list[dict]:
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


def _log_chunk_durations(chunks: list[dict], duration: float, pad_s: float, trace_id: str) -> None:
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


@app.get("/health")
def health():
    return {
        "ok": MODEL_MANAGER.model is not None and MODEL_MANAGER.state.get("state") == "ready",
        "model_state": MODEL_MANAGER.state.get("state"),
        "asr_loaded": MODEL_MANAGER.model is not None,
        "asr_backend": STARTUP_CONFIG.backend,
        "asr_model_key": STARTUP_CONFIG.model_key,
        "asr_model_name": STARTUP_CONFIG.model_name,
        "asr_trust_remote_code": STARTUP_CONFIG.trust_remote_code,
        "asr_return_timestamps": STARTUP_CONFIG.return_timestamps,
        "asr_chunk_length_s": STARTUP_CONFIG.chunk_length_s,
        "asr_stride_length_s": STARTUP_CONFIG.stride_length_s,
        "asr_attention_implementation": STARTUP_CONFIG.attention_implementation,
        "startup_model": public_config(STARTUP_CONFIG),
        "active_model": public_config(MODEL_MANAGER.active_config),
        "model_switch": dict(MODEL_MANAGER.state),
        "available_models_count": len(MODEL_REGISTRY),
        "hf_token_configured": bool(SETTINGS.hf_token),
        "diarization_model": DIARIZATION_MODEL,
        "diarization_loaded": is_diarizer_loaded(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_mem": cuda_mem(),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [public_model_entry(entry) for entry in MODEL_REGISTRY.values()],
    }


@app.get("/v1/models/current")
def get_current_model():
    return {
        "object": "model.current",
        "active_model": public_config(MODEL_MANAGER.active_config),
        "asr_loaded": MODEL_MANAGER.model is not None,
        "model_state": dict(MODEL_MANAGER.state),
        "cuda_mem": cuda_mem(),
    }


@app.post("/v1/models/current")
def set_current_model(payload: dict[str, Any] = Body(...)):
    model_id = str(payload.get("model") or payload.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="Request body must include 'model' or 'model_id'")

    entry = find_model_entry(model_id)
    if entry is None or not entry.selectable:
        raise HTTPException(status_code=404, detail=f"Unknown or unselectable model: {model_id!r}")

    language = payload.get("language")
    if language is not None:
        language = str(language).strip() or None

    return_timestamps = payload.get("return_timestamps")
    if return_timestamps is None and "timestamps" in payload:
        timestamps_value = payload.get("timestamps")
        if isinstance(timestamps_value, str):
            return_timestamps = timestamps_value.strip().lower() not in {"0", "false", "none", "off", "no"}
        else:
            return_timestamps = bool(timestamps_value)

    config = config_from_entry(
        entry,
        language=language,
        return_timestamps=return_timestamps,
    )
    try:
        return MODEL_MANAGER.switch(config)
    except ModelSwitchError as exc:
        raise HTTPException(status_code=500, detail=exc.detail) from exc


@app.on_event("startup")
def _startup_load_model() -> None:
    MODEL_MANAGER.startup(STARTUP_CONFIG)



def _build_vad_params(
    *,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_merge_gap_ms: int,
    vad_target_min_s: float,
    vad_target_max_s: float,
    vad_hard_max_s: float,
    vad_overlap_s: float,
    vad_speech_pad_ms: int,
    force_vad: str,
    vad_energy_gate: Optional[bool],
    vad_energy_db: Optional[float],
    vad_energy_frame_ms: Optional[int],
    vad_energy_min_active_ms: Optional[int],
    vad_energy_merge_gap_ms: Optional[int],
    vad_energy_active_skip: Optional[float],
    vad_uniform_chunk_s: Optional[float],
    vad_uniform_overlap_s: Optional[float],
) -> dict:
    return {
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
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    response_format: Literal["verbose_json"] = Form("verbose_json", description="Only supported response format."),
    diarization: bool = Form(False, description="Enable speaker diarization. Requires timestamps=word."),
    timestamps: Literal["word", "segment", "none"] = Form("word", description="word, segment, or none. word required for diarization."),
    language: str = Form("en", description="Currently only response metadata; active model language is selected at model load."),
    chunk_mode: Literal["memory", "file"] = Form("memory", description="memory (default) or file (writes chunk WAVs to disk)."),
    chunk_only: bool = Form(False, description="Return VAD chunks only; skips ASR."),
    trace_audio: bool = Form(False, description="Emit per-request trace logs for ingest/decode/VAD."),
    force_vad: Literal["off", "on"] = Form("on", description="off or on (default; force Silero VAD, ignore energy gate)."),
    vad_sample_rate: Optional[int] = Form(None, description="Override VAD sample rate (default env SETTINGS.vad_sample_rate=16000)."),
    vad_threshold: Optional[float] = Form(None, description="Silero threshold (default env VAD_THRESHOLD=0.30)."),
    vad_min_speech_ms: Optional[int] = Form(None, description="Minimum speech duration (ms). Default env VAD_MIN_SPEECH_MS=150."),
    vad_min_silence_ms: Optional[int] = Form(None, description="Minimum silence duration (ms). Default env VAD_MIN_SILENCE_MS=220."),
    vad_merge_gap_ms: Optional[int] = Form(None, description="Gap to merge speech segments (ms). Default env VAD_MERGE_GAP_MS=200."),
    vad_target_min_s: Optional[float] = Form(None, description="Target min chunk length (s). Default env VAD_TARGET_MIN_S=10.0."),
    vad_target_max_s: Optional[float] = Form(None, description="Target max chunk length (s). Default env VAD_TARGET_MAX_S=20.0."),
    vad_hard_max_s: Optional[float] = Form(None, description="Hard max chunk length (s). Default env VAD_HARD_MAX_S=30.0."),
    vad_overlap_s: Optional[float] = Form(None, description="Chunk overlap (s). Default env VAD_OVERLAP_S=1.0."),
    vad_speech_pad_ms: Optional[int] = Form(None, description="Pad speech edges (ms). Default env VAD_SPEECH_PAD_MS=250."),
    vad_energy_gate: Optional[bool] = Form(None, description="Enable/disable energy gate (default env VAD_ENERGY_GATE=0)."),
    vad_energy_db: Optional[float] = Form(None, description="Energy gate threshold (dB). Default env VAD_ENERGY_DB=-35."),
    vad_energy_frame_ms: Optional[int] = Form(None, description="Energy gate frame size (ms). Default env VAD_ENERGY_FRAME_MS=100."),
    vad_energy_min_active_ms: Optional[int] = Form(None, description="Energy gate min active duration (ms). Default env VAD_ENERGY_MIN_ACTIVE_MS=500."),
    vad_energy_merge_gap_ms: Optional[int] = Form(None, description="Energy gate merge gap (ms). Default env VAD_ENERGY_MERGE_GAP_MS=800."),
    vad_energy_active_skip: Optional[float] = Form(None, description="Skip Silero if active ratio >= this. Default env VAD_ENERGY_ACTIVE_SKIP=0.85."),
    vad_uniform_chunk_s: Optional[float] = Form(None, description="Uniform chunk length (s) when energy gate skips. Default env VAD_UNIFORM_CHUNK_S=30."),
    vad_uniform_overlap_s: Optional[float] = Form(None, description="Uniform chunk overlap (s). Default env VAD_UNIFORM_OVERLAP_S=1.0."),
):
    if response_format != "verbose_json":
        raise HTTPException(status_code=400, detail="Only verbose_json is supported")
    if timestamps not in {"word", "segment", "none"}:
        raise HTTPException(status_code=400, detail="timestamps must be 'word', 'segment', or 'none'")
    if chunk_mode not in {"memory", "file"}:
        raise HTTPException(status_code=400, detail="chunk_mode must be 'memory' or 'file'")
    if force_vad not in {"off", "on"}:
        raise HTTPException(status_code=400, detail="force_vad must be 'off' or 'on'")

    with MODEL_MANAGER.lock:
        _, initial_config = _active_model_or_503()
        _validate_request_for_config(initial_config, diarization=diarization, timestamps=timestamps)

    with tempfile.TemporaryDirectory() as tmpdir:
        start_req = time.perf_counter()
        trace_id = f"trace-{int(start_req * 1000)}" if trace_audio else None
        tmp_path = Path(tmpdir) / file.filename

        file_start = time.perf_counter()
        file_bytes = await file.read()
        tmp_path.write_bytes(file_bytes)
        file_elapsed = time.perf_counter() - file_start
        if trace_audio:
            print(f"[{trace_id}] ingest file={file.filename!r} bytes={len(file_bytes)} write_s={file_elapsed:.3f}")

        decode_start = time.perf_counter()
        waveform, sr = _load_audio_waveform(str(tmp_path))
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
            print(
                f"[{trace_id}] decode sr={sr} channels={channels} "
                f"samples={int(waveform.shape[-1])} duration_s={duration:.3f} decode_s={decode_elapsed:.3f}"
            )

        vad_waveform = waveform
        vad_sr = vad_sample_rate if vad_sample_rate is not None else SETTINGS.vad_sample_rate
        if vad_sr != sr:
            vad_waveform = torchaudio.functional.resample(waveform, sr, vad_sr)

        vad_threshold = vad_threshold if vad_threshold is not None else env_float("VAD_THRESHOLD", "0.30")
        vad_min_speech_ms = vad_min_speech_ms if vad_min_speech_ms is not None else env_int("VAD_MIN_SPEECH_MS", "150")
        vad_min_silence_ms = vad_min_silence_ms if vad_min_silence_ms is not None else env_int("VAD_MIN_SILENCE_MS", "220")
        vad_merge_gap_ms = vad_merge_gap_ms if vad_merge_gap_ms is not None else env_int("VAD_MERGE_GAP_MS", "200")
        vad_target_min_s = vad_target_min_s if vad_target_min_s is not None else env_float("VAD_TARGET_MIN_S", "10.0")
        vad_target_max_s = vad_target_max_s if vad_target_max_s is not None else env_float("VAD_TARGET_MAX_S", "20.0")
        vad_hard_max_s = vad_hard_max_s if vad_hard_max_s is not None else env_float("VAD_HARD_MAX_S", "30.0")
        vad_overlap_s = vad_overlap_s if vad_overlap_s is not None else env_float("VAD_OVERLAP_S", "1.0")
        vad_speech_pad_ms = vad_speech_pad_ms if vad_speech_pad_ms is not None else env_int("VAD_SPEECH_PAD_MS", "250")
        pad_s = vad_speech_pad_ms / 1000.0

        energy_gate_override = None if force_vad == "off" else False
        energy_overrides = {}
        for key, value in {
            "energy_gate": vad_energy_gate,
            "energy_db": vad_energy_db,
            "energy_frame_ms": vad_energy_frame_ms,
            "energy_min_active_ms": vad_energy_min_active_ms,
            "energy_merge_gap_ms": vad_energy_merge_gap_ms,
            "energy_active_skip": vad_energy_active_skip,
            "uniform_chunk_s": vad_uniform_chunk_s,
            "uniform_overlap_s": vad_uniform_overlap_s,
        }.items():
            if value is not None:
                energy_overrides[key] = value

        vad_params = _build_vad_params(
            vad_threshold=vad_threshold,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_merge_gap_ms=vad_merge_gap_ms,
            vad_target_min_s=vad_target_min_s,
            vad_target_max_s=vad_target_max_s,
            vad_hard_max_s=vad_hard_max_s,
            vad_overlap_s=vad_overlap_s,
            vad_speech_pad_ms=vad_speech_pad_ms,
            force_vad=force_vad,
            vad_energy_gate=vad_energy_gate,
            vad_energy_db=vad_energy_db,
            vad_energy_frame_ms=vad_energy_frame_ms,
            vad_energy_min_active_ms=vad_energy_min_active_ms,
            vad_energy_merge_gap_ms=vad_energy_merge_gap_ms,
            vad_energy_active_skip=vad_energy_active_skip,
            vad_uniform_chunk_s=vad_uniform_chunk_s,
            vad_uniform_overlap_s=vad_uniform_overlap_s,
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
            if trace_audio and chunks:
                print(
                    f"[{trace_id}] vad_chunks count={len(chunks)} mode=memory "
                    f"first={chunks[0]['start_s']:.2f}-{chunks[0]['end_s']:.2f} "
                    f"last={chunks[-1]['start_s']:.2f}-{chunks[-1]['end_s']:.2f}"
                )
                _log_chunk_durations(chunks, duration, pad_s, trace_id)

            if chunk_only or not chunks:
                if not chunks and not chunk_only:
                    return build_response([], [], language=language, duration=duration, text="")
                return {
                    "ok": True,
                    "chunk_only": True,
                    "chunk_mode": chunk_mode,
                    "duration": duration,
                    "sample_rate": sr,
                    "vad_sample_rate": vad_sr,
                    "vad_params": vad_params,
                    "chunks": _chunk_debug_from_in_memory(chunks, duration, pad_s) if chunks else [],
                }

            start_asr = time.perf_counter()
            with MODEL_MANAGER.lock:
                asr_model, active_config = _active_model_or_503()
                _validate_request_for_config(active_config, diarization=diarization, timestamps=timestamps)
                effective_timestamps = effective_timestamps(active_config, timestamps)
                result = transcribe_chunks_in_memory_mode(
                    asr_model=asr_model,
                    chunks=chunks,
                    batch_size=4,
                    timestamps=effective_timestamps,
                    pad_left_s=pad_s,
                )
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
            if trace_audio and chunk_paths:
                _, first_start, first_end = parse_chunk_filename(chunk_paths[0].name)
                _, last_start, last_end = parse_chunk_filename(chunk_paths[-1].name)
                print(
                    f"[{trace_id}] vad_chunks count={len(chunk_paths)} mode=file "
                    f"first={first_start:.2f}-{first_end:.2f} last={last_start:.2f}-{last_end:.2f}"
                )

            if chunk_only or not chunk_paths:
                if not chunk_paths and not chunk_only:
                    return build_response([], [], language=language, duration=duration, text="")
                return {
                    "ok": True,
                    "chunk_only": True,
                    "chunk_mode": chunk_mode,
                    "duration": duration,
                    "sample_rate": sr,
                    "vad_sample_rate": vad_sr,
                    "vad_params": vad_params,
                    "chunks": _chunk_debug_from_files(chunk_paths, duration, pad_s) if chunk_paths else [],
                }

            start_asr = time.perf_counter()
            with MODEL_MANAGER.lock:
                asr_model, active_config = _active_model_or_503()
                _validate_request_for_config(active_config, diarization=diarization, timestamps=timestamps)
                effective_timestamps = effective_timestamps(active_config, timestamps)
                result = transcribe_chunks_with_model_mode(
                    asr_model=asr_model,
                    chunk_dir=chunk_dir,
                    pad_left_s=pad_s,
                    pad_right_s=pad_s,
                    batch_size=4,
                    timestamps=effective_timestamps,
                )

        elapsed = time.perf_counter() - start_asr
        print(
            f"file write {file_elapsed:.2f}s; decode {decode_elapsed:.2f}s; "
            f"vad {vad_elapsed:.2f}s; chunks transcribed in {elapsed:.2f}s"
        )
        print(f"cuda mem after asr: {cuda_mem()}")

        text_override = result["text"]
        words = result["words"] if effective_timestamps == "word" else []

        if diarization:
            start_diar = time.perf_counter()
            turns = diarize_waveform(waveform, sr, Path(tmpdir))
            diar_elapsed = time.perf_counter() - start_diar
            print(f"diarization in {diar_elapsed:.2f}s")
            words_with_speaker = assign_speakers(words, turns)
        else:
            words_with_speaker = [{**w, "speaker": w.get("speaker") or "UNKNOWN"} for w in words]

        start_merge = time.perf_counter()
        if effective_timestamps == "word":
            segments = group_words_into_segments(words_with_speaker)
            for idx, seg in enumerate(segments):
                seg.setdefault("id", idx)
            text_override = None
        elif effective_timestamps == "segment":
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
