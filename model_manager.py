#!/usr/bin/env python3
"""ASR model lifecycle, switching, and GPU state."""

from __future__ import annotations

import gc
import threading
import time
from datetime import datetime, timezone
from typing import Any

import torch

from asr import ASRConfig, load_asr_backend
from chunk_transcribe import _patch_transcribe_dataloader_no_lhotse
from diarization import unload_diarizer
from model_registry import public_config, startup_model_key
from settings import AppSettings, SETTINGS


class ModelUnavailableError(RuntimeError):
    pass


class ModelSwitchError(RuntimeError):
    def __init__(self, detail: Any):
        self.detail = detail
        super().__init__(str(detail))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def cuda_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


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
        if not hasattr(asr_model, attr_name):
            continue
        joint = getattr(asr_model, attr_name)
        if not (hasattr(joint, "decoding") and hasattr(joint.decoding, "decoding")):
            continue

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


class ASRModelManager:
    def __init__(self, settings: AppSettings = SETTINGS):
        self.settings = settings
        self.lock = threading.RLock()
        self.model = None
        self.active_config: ASRConfig | None = None
        self.state: dict[str, Any] = {
            "state": "starting",
            "in_progress": False,
            "started_at": None,
            "completed_at": None,
            "last_error": None,
            "last_requested_model": startup_model_key(settings),
            "switch_count": 0,
        }

    def _cleanup_cuda(self) -> None:
        gc.collect()
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _load_model(self, config: ASRConfig, *, warmup: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            "Loading ASR model: "
            f"backend={config.backend!r}, key={config.model_key!r}, "
            f"model_name={config.model_name!r}"
        )
        asr_model = load_asr_backend(config)
        asr_model = asr_model.to(device)

        if config.backend == "nemo":
            _patch_transcribe_dataloader_no_lhotse(asr_model)
            if self.settings.disable_cuda_graphs:
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

    def _unload_unlocked(self, *, unload_diarization: bool) -> None:
        old_asr = self.model
        self.model = None
        if old_asr is not None:
            del old_asr

        if unload_diarization:
            unload_diarizer()

        self._cleanup_cuda()

    def require_active(self) -> tuple[Any, ASRConfig]:
        if self.model is None or self.active_config is None:
            raise ModelUnavailableError("ASR model is not loaded")
        return self.model, self.active_config

    def startup(self, config: ASRConfig) -> None:
        with self.lock:
            if self.model is not None:
                return

            self.state.update(
                {
                    "state": "loading",
                    "in_progress": True,
                    "started_at": utcnow(),
                    "completed_at": None,
                    "last_error": None,
                    "last_requested_model": config.model_key,
                }
            )
            try:
                self.model = self._load_model(config, warmup=True)
                self.active_config = config
                self.state.update(
                    {
                        "state": "ready",
                        "in_progress": False,
                        "completed_at": utcnow(),
                        "last_error": None,
                    }
                )
            except Exception as exc:
                self.state.update(
                    {
                        "state": "error",
                        "in_progress": False,
                        "completed_at": utcnow(),
                        "last_error": str(exc),
                    }
                )
                raise

    def switch(self, config: ASRConfig) -> dict[str, Any]:
        with self.lock:
            previous_config = self.active_config
            previous_public = public_config(previous_config)

            if previous_config == config and self.model is not None:
                return {
                    "ok": True,
                    "changed": False,
                    "active_model": public_config(self.active_config),
                    "cuda_mem": cuda_mem(),
                }

            self.state.update(
                {
                    "state": "loading",
                    "in_progress": True,
                    "started_at": utcnow(),
                    "completed_at": None,
                    "last_error": None,
                    "last_requested_model": config.model_key,
                }
            )

            self._unload_unlocked(unload_diarization=True)
            load_started = time.perf_counter()
            try:
                self.model = self._load_model(config, warmup=False)
                load_s = time.perf_counter() - load_started
                self.active_config = config
                self.state.update(
                    {
                        "state": "ready",
                        "in_progress": False,
                        "completed_at": utcnow(),
                        "last_error": None,
                        "switch_count": int(self.state.get("switch_count", 0)) + 1,
                    }
                )
                return {
                    "ok": True,
                    "changed": True,
                    "previous_model": previous_public,
                    "active_model": public_config(self.active_config),
                    "load_s": round(load_s, 3),
                    "cuda_mem": cuda_mem(),
                }
            except Exception as exc:
                new_error = str(exc)
                self.state["last_error"] = new_error

                if previous_config is not None:
                    try:
                        self.model = self._load_model(previous_config, warmup=False)
                        self.active_config = previous_config
                        self.state.update(
                            {
                                "state": "ready",
                                "in_progress": False,
                                "completed_at": utcnow(),
                                "last_error": new_error,
                            }
                        )
                        raise ModelSwitchError(
                            {
                                "message": "requested model failed to load; previous model was restored",
                                "error": new_error,
                                "active_model": public_config(self.active_config),
                            }
                        ) from exc
                    except ModelSwitchError:
                        raise
                    except Exception as restore_exc:
                        self.state.update(
                            {
                                "state": "error",
                                "in_progress": False,
                                "completed_at": utcnow(),
                                "last_error": (
                                    f"{new_error}; restore failed: {restore_exc}"
                                ),
                            }
                        )
                        raise ModelSwitchError(
                            {
                                "message": (
                                    "requested model failed to load and previous model "
                                    "restore failed"
                                ),
                                "error": new_error,
                                "restore_error": str(restore_exc),
                            }
                        ) from exc

                self.state.update(
                    {
                        "state": "error",
                        "in_progress": False,
                        "completed_at": utcnow(),
                        "last_error": new_error,
                    }
                )
                raise ModelSwitchError(new_error) from exc


MODEL_MANAGER = ASRModelManager()
