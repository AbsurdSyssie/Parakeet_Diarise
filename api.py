#!/usr/bin/env python3
"""ASR API: VAD chunking + runtime-selectable ASR + optional diarization."""

from __future__ import annotations

import gc
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False
import torchaudio
import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile

from api_response import build_response
from asr_backend import ASRConfig, load_asr_backend, resolve_asr_model
from asr_merge import parse_chunk_filename
from chunk_transcribe import (
    _patch_transcribe_dataloader_no_lhotse,
    transcribe_chunks_in_memory_mode,
    transcribe_chunks_with_model_mode,
)
from diarize_align import assign_speakers, group_words_into_segments
from vad_chunk import run_vad_chunks_from_waveform, run_vad_chunks_in_memory_from_waveform

load_dotenv()

app = FastAPI()
_MODEL_LOCK = threading.RLock()
_ASR_MODEL = None
_DIAR_MODEL = None
_ACTIVE_CONFIG: ASRConfig | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelRegistryEntry:
    id: str
    backend: str
    model_name: str
    description: str
    language: str = "en"
    task: str = "transcribe"
    trust_remote_code: bool = False
    return_timestamps: bool = False
    chunk_length_s: int | None = None
    stride_length_s: int | tuple[int, int] | None = None
    supports_word_timestamps: bool = True
    supports_segment_timestamps: bool = True
    supports_diarization: bool = True
    selectable: bool = True


def _get_env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _get_env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _get_env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
)
ASR_BACKEND = os.getenv("ASR_BACKEND", "nemo").strip().lower()
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en").strip()
ASR_TASK = os.getenv("ASR_TASK", "transcribe").strip()
ASR_TRUST_REMOTE_CODE = _get_env_bool("ASR_TRUST_REMOTE_CODE", "0")
ASR_RETURN_TIMESTAMPS = _get_env_bool("ASR_RETURN_TIMESTAMPS", "0")
ASR_CHUNK_LENGTH_S = _get_optional_env_int("ASR_CHUNK_LENGTH_S")
ASR_STRIDE_LENGTH_S = _get_optional_env_int("ASR_STRIDE_LENGTH_S")
VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", "16000"))

MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "parakeet-0.6b": ModelRegistryEntry(
        id="parakeet-0.6b",
        backend="nemo",
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        description="NVIDIA Parakeet TDT 0.6B v3 via NeMo",
        supports_word_timestamps=True,
        supports_segment_timestamps=True,
        supports_diarization=True,
    ),
    "parakeet-1.1b": ModelRegistryEntry(
        id="parakeet-1.1b",
        backend="nemo",
        model_name="nvidia/parakeet-tdt-1.1b",
        description="NVIDIA Parakeet TDT 1.1B via NeMo",
        supports_word_timestamps=True,
        supports_segment_timestamps=True,
        supports_diarization=True,
    ),
    "medical-whisper-large-v3": ModelRegistryEntry(
        id="medical-whisper-large-v3",
        backend="whisper",
        model_name="Na0s/Medical-Whisper-Large-v3",
        description="Medical Whisper Large v3 via Transformers pipeline",
        supports_word_timestamps=True,
        supports_segment_timestamps=True,
        supports_diarization=True,
    ),
    "faster-whisper-large-v3": ModelRegistryEntry(
        id="faster-whisper-large-v3",
        backend="faster-whisper",
        model_name="Systran/faster-whisper-large-v3",
        description="SYSTRAN faster-whisper Large v3 via CTranslate2",
        supports_word_timestamps=True,
        supports_segment_timestamps=True,
        supports_diarization=True,
    ),
    "cohere-transcribe-03-2026": ModelRegistryEntry(
        id="cohere-transcribe-03-2026",
        backend="transformers-asr",
        model_name="CohereLabs/cohere-transcribe-03-2026",
        description="Cohere Transcribe via Transformers custom ASR backend",
        trust_remote_code=ASR_TRUST_REMOTE_CODE,
        return_timestamps=False,
        supports_word_timestamps=False,
        supports_segment_timestamps=False,
        supports_diarization=False,
    ),
}


def _default_model_key_for_backend(backend: str) -> str:
    if backend == "nemo":
        return "parakeet-0.6b"
    if backend in {"whisper", "transformers-whisper"}:
        return "medical-whisper-large-v3"
    if backend == "faster-whisper":
        return "faster-whisper-large-v3"
    if backend in {"transformers-asr", "hf-asr"}:
        return "cohere-transcribe-03-2026"
    return "parakeet-0.6b"


MODEL_KEY = os.getenv("ASR_MODEL", _default_model_key_for_backend(ASR_BACKEND)).strip()
MODEL_NAME = resolve_asr_model(ASR_BACKEND, MODEL_KEY)

_MODEL_STATE: dict[str, Any] = {
    "state": "starting",
    "in_progress": False,
    "started_at": None,
    "completed_at": None,
    "last_error": None,
    "last_requested_model": MODEL_KEY,
    "switch_count": 0,
}


def _find_model_entry(model_id: str, backend: str | None = None) -> ModelRegistryEntry | None:
    normalized = (model_id or "").strip()
    normalized_backend = (backend or "").strip().lower()
    for entry in MODEL_REGISTRY.values():
        if normalized not in {entry.id, entry.model_name}:
            continue
        if normalized_backend and normalized_backend != entry.backend:
            continue
        return entry
    return None


def _model_entry_for_config(config: ASRConfig | None) -> ModelRegistryEntry | None:
    if config is None:
        return None
    return _find_model_entry(config.model_key, backend=config.backend) or _find_model_entry(
        config.model_name, backend=config.backend
    )


def _public_model_entry(entry: ModelRegistryEntry) -> dict[str, Any]:
    data = asdict(entry)
    data["object"] = "model"
    data["owned_by"] = entry.backend
    return data


def _public_config(config: ASRConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    entry = _model_entry_for_config(config)
    return {
        "backend": config.backend,
        "model_key": config.model_key,
        "model_name": config.model_name,
        "language": config.language,
        "task": config.task,
        "trust_remote_code": config.trust_remote_code,
        "return_timestamps": config.return_timestamps,
        "chunk_length_s": config.chunk_length_s,
        "stride_length_s": config.stride_length_s,
        "supports_word_timestamps": bool(entry.supports_word_timestamps) if entry else None,
        "supports_segment_timestamps": bool(entry.supports_segment_timestamps) if entry else None,
        "supports_diarization": bool(entry.supports_diarization) if entry else None,
    }


def _config_from_entry(
    entry: ModelRegistryEntry,
    *,
    language: str | None = None,
    return_timestamps: bool | None = None,
) -> ASRConfig:
    requested_timestamps = entry.return_timestamps if return_timestamps is None else bool(return_timestamps)
    if (
        requested_timestamps
        and entry.backend not in {"transformers-asr", "hf-asr"}
        and not (entry.supports_word_timestamps or entry.supports_segment_timestamps)
    ):
        requested_timestamps = False
    return ASRConfig(
        backend=entry.backend,
        model_key=entry.id,
        model_name=entry.model_name,
        hf_token=HF_TOKEN,
        language=(language or entry.language or ASR_LANGUAGE),
        task=entry.task or ASR_TASK,
        trust_remote_code=entry.trust_remote_code,
        return_timestamps=requested_timestamps,
        chunk_length_s=entry.chunk_length_s,
        stride_length_s=entry.stride_length_s,
    )


def _initial_config_from_env() -> ASRConfig:
    entry = _find_model_entry(MODEL_KEY, backend=ASR_BACKEND) or _find_model_entry(
        MODEL_NAME, backend=ASR_BACKEND
    )
    if entry is None:
        if ASR_BACKEND in {"transformers-asr", "hf-asr"}:
            return ASRConfig(
                backend=ASR_BACKEND,
                model_key=MODEL_KEY,
                model_name=MODEL_NAME,
                hf_token=HF_TOKEN,
                language=ASR_LANGUAGE,
                task=ASR_TASK,
                trust_remote_code=ASR_TRUST_REMOTE_CODE,
                return_timestamps=ASR_RETURN_TIMESTAMPS,
                chunk_length_s=ASR_CHUNK_LENGTH_S,
                stride_length_s=ASR_STRIDE_LENGTH_S,
            )
        raise RuntimeError(
            "ASR_MODEL must be one of the curated /v1/models entries "
            f"(got backend={ASR_BACKEND!r}, model={MODEL_KEY!r})"
        )
    config = _config_from_entry(
        entry,
        language=ASR_LANGUAGE,
        return_timestamps=ASR_RETURN_TIMESTAMPS if ASR_RETURN_TIMESTAMPS else None,
    )
    if config.backend in {"transformers-asr", "hf-asr"}:
        config.trust_remote_code = ASR_TRUST_REMOTE_CODE
        config.return_timestamps = ASR_RETURN_TIMESTAMPS
        config.chunk_length_s = ASR_CHUNK_LENGTH_S
        config.stride_length_s = ASR_STRIDE_LENGTH_S
    return config


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


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


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


def _load_model_from_config(config: ASRConfig, *, warmup: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "Loading ASR model: "
        f"backend={config.backend!r}, key={config.model_key!r}, model_name={config.model_name!r}"
    )
    asr_model = load_asr_backend(config)
    asr_model = asr_model.to(device)

    if config.backend == "nemo":
        _patch_transcribe_dataloader_no_lhotse(asr_model)
        if _get_env_bool("DISABLE_CUDA_GRAPHS", "0"):
            if _disable_cuda_graphs(asr_model, verbose=True):
                print("  CUDA graphs disabled for decoding")

    if warmup:
        dummy = torch.zeros(16000, dtype=torch.float32)
        asr_model.transcribe(
            [dummy],
            timestamps=False,
            verbose=False,
            batch_size=1,
            num_workers=0,
            return_hypotheses=True,
        )

    return asr_model


def _unload_models(*, unload_diarization: bool = False) -> None:
    global _ASR_MODEL, _DIAR_MODEL
    old_asr = _ASR_MODEL
    _ASR_MODEL = None
    if old_asr is not None:
        del old_asr

    if unload_diarization:
        old_diar = _DIAR_MODEL
        _DIAR_MODEL = None
        if old_diar is not None:
            del old_diar

    _cleanup_cuda()


def _effective_timestamps(config: ASRConfig, requested: str) -> str:
    if requested == "none":
        return "none"
    if config.backend in {"transformers-asr", "hf-asr"}:
        if not config.return_timestamps:
            return "none"
        return requested
    entry = _model_entry_for_config(config)
    if entry is None:
        return requested
    if requested == "word":
        if entry.supports_word_timestamps and config.return_timestamps:
            return "word"
        if entry.supports_segment_timestamps and config.return_timestamps:
            return "segment"
        return "none"
    if requested == "segment":
        if entry.supports_segment_timestamps and config.return_timestamps:
            return "segment"
        return "none"
    return requested


def _validate_request_for_config(config: ASRConfig, *, diarization: bool, timestamps: str) -> None:
    if diarization and timestamps != "word":
        raise HTTPException(status_code=400, detail="diarization requires timestamps=word")
    if not diarization:
        return
    if config.backend in {"transformers-asr", "hf-asr"} and not config.return_timestamps:
        raise HTTPException(
            status_code=400,
            detail="diarization requires ASR_RETURN_TIMESTAMPS=1 for transformers-asr",
        )
    entry = _model_entry_for_config(config)
    if entry is not None and not entry.supports_diarization:
        raise HTTPException(
            status_code=400,
            detail=f"diarization is not supported by active model {entry.id!r}",
        )
    if _effective_timestamps(config, timestamps) != "word":
        raise HTTPException(
            status_code=400,
            detail="diarization requires an active model with word timestamps enabled",
        )


def _active_model_or_503() -> tuple[Any, ASRConfig]:
    if _ASR_MODEL is None or _ACTIVE_CONFIG is None:
        raise HTTPException(status_code=503, detail="ASR model is not loaded")
    return _ASR_MODEL, _ACTIVE_CONFIG


def _switch_model(config: ASRConfig) -> dict[str, Any]:
    global _ASR_MODEL, _ACTIVE_CONFIG
    with _MODEL_LOCK:
        previous_config = _ACTIVE_CONFIG
        previous_public = _public_config(previous_config)
        if previous_config == config and _ASR_MODEL is not None:
            return {
                "ok": True,
                "changed": False,
                "active_model": _public_config(_ACTIVE_CONFIG),
                "cuda_mem": cuda_mem(),
            }

        _MODEL_STATE.update(
            {
                "state": "loading",
                "in_progress": True,
                "started_at": _utcnow(),
                "completed_at": None,
                "last_error": None,
                "last_requested_model": config.model_key,
            }
        )

        _unload_models(unload_diarization=True)
        load_started = time.perf_counter()
        try:
            new_model = _load_model_from_config(config, warmup=False)
            load_s = time.perf_counter() - load_started
            _ASR_MODEL = new_model
            _ACTIVE_CONFIG = config
            _MODEL_STATE.update(
                {
                    "state": "ready",
                    "in_progress": False,
                    "completed_at": _utcnow(),
                    "last_error": None,
                    "switch_count": int(_MODEL_STATE.get("switch_count", 0)) + 1,
                }
            )
            return {
                "ok": True,
                "changed": True,
                "previous_model": previous_public,
                "active_model": _public_config(_ACTIVE_CONFIG),
                "load_s": round(load_s, 3),
                "cuda_mem": cuda_mem(),
            }
        except Exception as exc:
            new_error = str(exc)
            _MODEL_STATE["last_error"] = new_error
            if previous_config is not None:
                try:
                    restored_model = _load_model_from_config(previous_config, warmup=False)
                    _ASR_MODEL = restored_model
                    _ACTIVE_CONFIG = previous_config
                    _MODEL_STATE.update(
                        {
                            "state": "ready",
                            "in_progress": False,
                            "completed_at": _utcnow(),
                            "last_error": new_error,
                        }
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "message": "requested model failed to load; previous model was restored",
                            "error": new_error,
                            "active_model": _public_config(_ACTIVE_CONFIG),
                        },
                    ) from exc
                except HTTPException:
                    raise
                except Exception as restore_exc:
                    _MODEL_STATE.update(
                        {
                            "state": "error",
                            "in_progress": False,
                            "completed_at": _utcnow(),
                            "last_error": f"{new_error}; restore failed: {restore_exc}",
                        }
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "message": "requested model failed to load and previous model restore failed",
                            "error": new_error,
                            "restore_error": str(restore_exc),
                        },
                    ) from exc

            _MODEL_STATE.update(
                {
                    "state": "error",
                    "in_progress": False,
                    "completed_at": _utcnow(),
                    "last_error": new_error,
                }
            )
            raise HTTPException(status_code=500, detail=new_error) from exc


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
        "ok": _ASR_MODEL is not None and _MODEL_STATE.get("state") == "ready",
        "model_state": _MODEL_STATE.get("state"),
        "asr_loaded": _ASR_MODEL is not None,
        "asr_backend": ASR_BACKEND,
        "asr_model_key": MODEL_KEY,
        "asr_model_name": MODEL_NAME,
        "asr_trust_remote_code": ASR_TRUST_REMOTE_CODE,
        "asr_return_timestamps": ASR_RETURN_TIMESTAMPS,
        "asr_chunk_length_s": ASR_CHUNK_LENGTH_S,
        "asr_stride_length_s": ASR_STRIDE_LENGTH_S,
        "active_model": _public_config(_ACTIVE_CONFIG),
        "model_switch": dict(_MODEL_STATE),
        "available_models_count": len(MODEL_REGISTRY),
        "hf_token_configured": bool(HF_TOKEN),
        "cuda_available": torch.cuda.is_available(),
        "cuda_mem": cuda_mem(),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [_public_model_entry(entry) for entry in MODEL_REGISTRY.values()],
    }


@app.get("/v1/models/current")
def get_current_model():
    return {
        "object": "model.current",
        "active_model": _public_config(_ACTIVE_CONFIG),
        "asr_loaded": _ASR_MODEL is not None,
        "model_state": dict(_MODEL_STATE),
        "cuda_mem": cuda_mem(),
    }


@app.post("/v1/models/current")
def set_current_model(payload: dict[str, Any] = Body(...)):
    model_id = str(payload.get("model") or payload.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="Request body must include 'model' or 'model_id'")

    entry = _find_model_entry(model_id)
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

    config = _config_from_entry(entry, language=language, return_timestamps=return_timestamps)
    return _switch_model(config)


@app.on_event("startup")
def _startup_load_model() -> None:
    global _ASR_MODEL, _ACTIVE_CONFIG
    with _MODEL_LOCK:
        if _ASR_MODEL is not None:
            return
        config = _initial_config_from_env()
        _MODEL_STATE.update(
            {
                "state": "loading",
                "in_progress": True,
                "started_at": _utcnow(),
                "completed_at": None,
                "last_error": None,
                "last_requested_model": config.model_key,
            }
        )
        try:
            _ASR_MODEL = _load_model_from_config(config, warmup=True)
            _ACTIVE_CONFIG = config
            _MODEL_STATE.update(
                {
                    "state": "ready",
                    "in_progress": False,
                    "completed_at": _utcnow(),
                    "last_error": None,
                }
            )
        except Exception as exc:
            _MODEL_STATE.update(
                {
                    "state": "error",
                    "in_progress": False,
                    "completed_at": _utcnow(),
                    "last_error": str(exc),
                }
            )
            raise


def _get_diar_model():
    global _DIAR_MODEL
    if _DIAR_MODEL is not None:
        return _DIAR_MODEL

    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
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
    wav = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
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
            if len(parts) < 3:
                continue
            start = float(parts[0])
            end = float(parts[1])
            speaker = parts[2]
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            start = float(seg[0])
            end = float(seg[1])
            speaker = seg[2]
        else:
            continue
        turns.append({"start": start, "end": end, "speaker": _normalize_speaker(speaker)})
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


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
    diarization: bool = Form(False, description="Enable Sortformer diarization. Requires timestamps=word."),
    timestamps: Literal["word", "segment", "none"] = Form("word", description="word, segment, or none. word required for diarization."),
    language: str = Form("en", description="Currently only response metadata; active model language is selected at model load."),
    chunk_mode: Literal["memory", "file"] = Form("memory", description="memory (default) or file (writes chunk WAVs to disk)."),
    chunk_only: bool = Form(False, description="Return VAD chunks only; skips ASR."),
    trace_audio: bool = Form(False, description="Emit per-request trace logs for ingest/decode/VAD."),
    force_vad: Literal["off", "on"] = Form("on", description="off or on (default; force Silero VAD, ignore energy gate)."),
    vad_sample_rate: Optional[int] = Form(None, description="Override VAD sample rate (default env VAD_SAMPLE_RATE=16000)."),
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

    with _MODEL_LOCK:
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
        vad_sr = vad_sample_rate if vad_sample_rate is not None else VAD_SAMPLE_RATE
        if vad_sr != sr:
            vad_waveform = torchaudio.functional.resample(waveform, sr, vad_sr)

        vad_threshold = vad_threshold if vad_threshold is not None else _get_env_float("VAD_THRESHOLD", "0.30")
        vad_min_speech_ms = vad_min_speech_ms if vad_min_speech_ms is not None else _get_env_int("VAD_MIN_SPEECH_MS", "150")
        vad_min_silence_ms = vad_min_silence_ms if vad_min_silence_ms is not None else _get_env_int("VAD_MIN_SILENCE_MS", "220")
        vad_merge_gap_ms = vad_merge_gap_ms if vad_merge_gap_ms is not None else _get_env_int("VAD_MERGE_GAP_MS", "200")
        vad_target_min_s = vad_target_min_s if vad_target_min_s is not None else _get_env_float("VAD_TARGET_MIN_S", "10.0")
        vad_target_max_s = vad_target_max_s if vad_target_max_s is not None else _get_env_float("VAD_TARGET_MAX_S", "20.0")
        vad_hard_max_s = vad_hard_max_s if vad_hard_max_s is not None else _get_env_float("VAD_HARD_MAX_S", "30.0")
        vad_overlap_s = vad_overlap_s if vad_overlap_s is not None else _get_env_float("VAD_OVERLAP_S", "1.0")
        vad_speech_pad_ms = vad_speech_pad_ms if vad_speech_pad_ms is not None else _get_env_int("VAD_SPEECH_PAD_MS", "250")
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
            with _MODEL_LOCK:
                asr_model, active_config = _active_model_or_503()
                _validate_request_for_config(active_config, diarization=diarization, timestamps=timestamps)
                effective_timestamps = _effective_timestamps(active_config, timestamps)
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
            with _MODEL_LOCK:
                asr_model, active_config = _active_model_or_503()
                _validate_request_for_config(active_config, diarization=diarization, timestamps=timestamps)
                effective_timestamps = _effective_timestamps(active_config, timestamps)
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
            turns = _run_sortformer(waveform, sr, Path(tmpdir))
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
