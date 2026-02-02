# Parakeet Progress

## Status

- Phase: Step 1 (minimal Parakeet setup)
- Environment: .venv created
- Last updated: 2026-01-25
## Latest investigation (2026-01-28)

- Symptom: hallucinated tokens like “Aaron/Aaron Powell/Aaron Ross” at the start/end of speaker sentences.
- Hypothesis: ASR is receiving a single long chunk (no VAD split), causing boundary artifacts.
- New debug tool: `chunk_only=true` on `/v1/audio/transcriptions` returns VAD chunk metadata without running ASR.
- Probe script: `test/asr_chunk_probe.py` (default output `test/tmp/asr_chunk_probe.json`).
- Result (earlier): `path/to/audio.wav` produced **one large chunk**, and the artifacts remain in that chunk’s ASR output.
- Result (2026-01-28): server-side trace run returned **13 uniform chunks** (~45s each with 0.5s overlap) with `vad_sample_rate=8000`, `duration=537.809s`. This indicates chunking is occurring when `VAD_ENERGY_ACTIVE_SKIP` triggers uniform slicing.
- Log evidence (user view, authoritative): trace IDs `trace-153382707` and `trace-153486325` show ingest + decode, then energy gate skip with ratio ~0.95 → 13 uniform intervals. A separate request shows energy gate enabled with ratio ~0.68 (no skip) and a full ASR path.
- Log evidence (latest, user view): `trace-153928262` shows VAD resample to 8 kHz, energy gate skip (ratio ~0.95), and 13 uniform chunks from `0.00–45.00` through `534.00–537.81`.
- Force VAD test (user view): `force_vad=on` produced speech-aligned chunks; trace `trace-154596386` shows first chunk `0.18–12.82`, last chunk `502.32–535.66`, and chunk duration checks matched expected (delta ~0.000).

## Problems observed

- VAD chunking can return a single interval for `MoreOrLessFull` in some runs, so ASR runs on a full-length window.
- Artifacts persist when ASR is fed the full audio as one chunk.
- The same file can also be uniformly chunked (13 chunks) depending on VAD/energy-gate behavior.
- Assistant-reported Docker logs can be stale if the container wasn’t rebuilt; the user’s own log view is authoritative and up to date.

## Changes made

- Added `chunk_only` parameter to ASR endpoint to return VAD chunk metadata without ASR.
- Added `test/asr_chunk_probe.py` to post audio to ASR and capture chunk boundaries.
- Added server-side `trace_audio` logging (ingest + decode) and validated with the 2026-01-28 trace run.

## Next hypotheses / possible fixes

- Investigate why VAD emits a single interval for `MoreOrLessFull` (energy gate skip? VAD config?).
- Force uniform chunking for continuous speech (`VAD_ENERGY_ACTIVE_SKIP` → uniform 45s windows).
- Reduce thresholds / merge gap to encourage splits or adjust `hard_max_s` logic.

## What works

- Python deps installed in `.venv`: `torch`, `torchaudio`, `nemo_toolkit[asr]`.
- Minimal transcription script added at `transcribe_parakeet.py` and executed.
- Transcription works on `path/to/audio.wav` with text output plus word/segment timestamps (after runtime patch to disable Lhotse).
- Containerized run succeeded; `docker compose up --build` produced text + timestamps and exited cleanly.

## What does not work / blockers

- Transcription run failed with a Lhotse DynamicCutSampler error (`TypeError: object.__init__() takes exactly one argument`).
- `transcribe_cfg` is not supported in this NeMo build (`TypeError: EncDecRNNTModel.transcribe() got an unexpected keyword argument 'transcribe_cfg'`).
- After updating `asr_model.cfg`, NeMo still used Lhotse in `_setup_transcribe_dataloader` (hardcoded `use_lhotse=True`).
- GPU is not available: `nvidia-smi` failed with `Failed to initialize NVML: Unknown Error`.

## Setup notes

- Created Python virtual environment at `.venv`.
- Installed `torch`, `torchaudio`, and `nemo_toolkit[asr]` per `docs/parakeet.md`.
- If network access is required for installs/downloads, request escalated permissions explicitly.
- GPU status check: `nvidia-smi` failed, so GPU drivers/CUDA appear unavailable in this environment.
- Plan change: move dev to a containerized environment that bundles CUDA + Python 3.11 (per `docs/parakeet.md`) to avoid host OS/toolchain issues.
- Added container artifacts: `Dockerfile`, `compose.yaml`, `.dockerignore`, and `requirements.txt`.
- Added `vad_chunk.py` and `docs/silero.md` for Silero VAD chunking of long WAVs.
- Silero VAD run failed in container due to tensor device mismatch (audio on CPU, model on CUDA); fix is to move audio to CUDA before `get_speech_timestamps`.
- Silero VAD run success: `vad_chunk.py /app/audio.wav` wrote 86 chunks to `Output/vad_chunks`.
- Added merge helpers + unit tests (`asr_merge.py`, `tests/test_asr_merge.py`); `python3 -m unittest discover -s tests` passes.
- Added `chunk_transcribe.py` to batch transcribe VAD chunks and merge timestamps into `Output/merged_words.json`.
- Container cache persisted under `Output/.cache` to avoid re-downloading Parakeet model between runs.
- Cache verified: container restored Parakeet model from `/app/.cache/huggingface/...`.
- Chunked ASR outputs generated: `Output/merged_words.json` and `Output/merged_text.txt` present despite logs not showing the final “Merged …” line.
- Output summary: `Output/merged_words.json` has 1554 words spanning ~0.08s to 535.16s.
- Created 30s clip `Output/MoreOrLess_30s.wav` and added `parakeet_clip_test.py` for clip-only timestamp inspection.
- Clip check: unchunked Parakeet timestamps for first ~30s are consistently ~0.16–0.32s later than merged VAD+ASR timestamps, consistent with VAD start detection and 0.2s padding.
- Added diarization alignment helpers + tests (`diarize_align.py`, `tests/test_diarize_align.py`); `python3 -m unittest discover -s tests` passes.
- Added diarization scripts (`diarize_full.py`, `merge_diarized.py`) and tests; compose now loads `.env` for `HF_TOKEN`.
- Diarization run failed in shared container due to pyannote 4.x vs NeMo torch pin mismatch; split into two containers (ASR and diarization) to avoid dependency conflicts.
- Merge failed: missing `Output/diarization_turns.json`; diarization step must run and produce the turns file before merging.
- Diarization container errors: torchcodec/FFmpeg and PyTorch `weights_only` safe-unpickle; mitigated by preloading audio via torchaudio and allowlisting `Specifications` in `diarize_full.py`.
- Diarization success: wrote 56 turns to `Output/diarization_turns.json` (std warning from pooling can be ignored).
- Merge success: wrote 81 diarized segments to `Output/diarized_segments.json`.
- WhisperX comparison (MoreOrLessFull): WhisperX segments=84 span=0.031–535.545s speakers={SPEAKER_00,SPEAKER_01}; ours segments=81 span=0.08–535.16s speakers={SPEAKER_00,SPEAKER_01,UNKNOWN}.
- WhisperX word-level comparison: WhisperX words=1514 vs ours=1554; first-300 mean abs start delta ~1.442s (median ~1.299s) using index-based pairing.
- WhisperX alignment-based comparison: 1393 matched words; mean abs start delta ~0.249s, median ~0.244s, mean signed delta ~-0.236s (ours slightly earlier).
- Added API scaffolding: `api.py`, `diarize_api.py`, `api_response.py`, and compose services `api` (port 8000) + `diarize_api` (port 8001).
- Enabled pip cache in Dockerfiles via `PIP_CACHE_DIR=/app/.cache/pip` and removed `--no-cache-dir` to speed rebuilds.
- API change: Parakeet model now loads once at startup with warmup and is reused per request (no per-request model init).
- API change: diarization URL is configurable via `DIARIZE_URL`; `api` now depends on `diarize_api`.
- Added in-memory VAD chunking path for API to avoid chunk file I/O; added tests for chunk merge helper.
- Silero VAD model now cached in-process to avoid reloading on every request.
- Added API usage guide: `docs/api.md`.
- Fixed API startup errors (stray patch text removed; missing `os` import added).
- Enabled pip cache in Dockerfiles via `PIP_CACHE_DIR=/app/.cache/pip` and removed `--no-cache-dir` to speed rebuilds.

## TODO

- [x] Load Parakeet once at API startup; add a warmup transcription.
- [x] Eliminate per-request model init; reuse a global model instance.
- [x] Merge VAD segments more aggressively (merge gaps <800ms and target 10–30s chunks, hard max 45s).
- Latest VAD run (post-merge) produced 3 chunks; max chunk still 476.5s (continuous speech with no gaps), so hard-cut with overlap is now enabled.
- VAD now hard-cuts continuous speech to ~45s windows with 0.5s overlap; latest run produced 13 chunks with expected overlaps.
- [x] Avoid disk I/O for chunks; pass in-memory tensors to NeMo if supported (API defaults to memory mode).
- [ ] Increase ASR batch size to reduce dataloader overhead.
- [ ] Enable mixed precision (`torch.autocast`) for inference.
- [ ] Add request-level GPU semaphore (one worker per GPU).
- [ ] Add timing logs for VAD, ASR, diarization, and merge.
- [ ] Add a fast mode that skips word timestamps when not needed.
- [x] Add API `timestamps` options (`segment`/`none`) and document diarization requirement.
- [ ] Re-benchmark against WhisperX after changes.
- [ ] Consider enabling TF32 for throughput (trade-off: slightly less deterministic vs faster matmul).
- [x] Add `/health` endpoints for ASR and diarize APIs.
- [ ] Investigate VAD latency and improve performance (decode vs VAD compute, CPU vs GPU, windowing).
- [ ] Test `VAD_DEVICE=cpu` vs `VAD_DEVICE=cuda` to compare VAD latency.
- Benchmarks (memory mode, `path/to/audio.wav`):
  - `timestamps=word`: 8.86s
  - `timestamps=segment`: 8.73s (~1.5% faster vs word)
  - `timestamps=none`: 7.62s (~14% faster vs word)
  - 16 kHz VAD, `VAD_DEVICE=cpu`, diarization=true: decode ~1.00s; VAD 5.95s; ASR 3.60s; diarization 10.04s; total 20.52s
  - 8 kHz VAD, `VAD_DEVICE=cuda`, diarization=true: decode 0.89s; VAD 7.40s; ASR 2.76s; diarization 9.88s; total 21.07s
- Energy gate test (1h audio, `VAD_ENERGY_GATE=1`, `VAD_DEVICE=cpu`, `VAD_SAMPLE_RATE=8000`): energy gate found 2 intervals; decode 0.28s; VAD 25.60s; ASR 18.86s; diarization 49.51s; total 95.09s
- Energy gate config used: `VAD_ENERGY_DB=-55`, `VAD_ENERGY_FRAME_MS=100`, `VAD_ENERGY_MIN_ACTIVE_MS=300`, `VAD_ENERGY_MERGE_GAP_MS=500`.
- Implemented: energy gate can skip Silero when active ratio >= `VAD_ENERGY_ACTIVE_SKIP` (default 0.7) and fall back to uniform chunking via `VAD_UNIFORM_CHUNK_S` + `VAD_UNIFORM_OVERLAP_S`.
- Updated: when skip threshold is hit, Silero is not run; uniform chunks are treated as speech directly.
- Energy gate skip test (1h audio, ratio=1.00): uniform intervals 80; decode 0.30s; VAD 0.48s; ASR 15.61s; diarization 50.72s; total 67.86s.
- Takeaway: energy-gate skip removes most VAD cost, diarization dominates overall latency.
- Added `DIARIZE_EMPTY_CACHE=1` option to run `gc.collect()` after diarization (no `torch.cuda.empty_cache()`).
- Added `DIARIZE_TF32=1` option to enable TF32 for diarization.
- TF32 test: no consistent diarization speedup observed (TF32=1 ~54.40s/53.11s vs TF32=0 ~51.96s/53.33s).
- Decision: keep `DIARIZE_TF32=0` by default.
- Fixed `NameError` in `run_vad_chunks_in_memory_from_waveform` by adding missing env reads for `VAD_ENERGY_ACTIVE_SKIP`/`VAD_UNIFORM_*`; validated with a 1s silence test inside the container.
- Added energy-gate option (`VAD_ENERGY_GATE=1`) to run a fast RMS screen before Silero.
- [x] Reduce short `UNKNOWN` speaker segments by lowering min overlap and merging sub-0.3s UNKNOWN into previous segment.
- Benchmarks (API, memory mode, `path/to/audio.wav`): `word` 8.86s, `segment` 8.73s (~1.5% faster), `none` 7.62s (~14% faster vs word).
- [ ] Investigate the delay time when loading silero and chunking for each request
## VRAM debug TODO

- [x] Add `cuda_mem()` logging (allocated vs reserved vs max) per request.
- [ ] Wrap Parakeet and pyannote inference in `torch.inference_mode()` and `autocast`.
- [ ] Add GPU concurrency guard (semaphore) with `GPU_CONCURRENCY=1` default.
- [ ] Add admin endpoint to call `torch.cuda.empty_cache()` + reset peak stats.
- [ ] Optional: auto-empty cache when reserved > threshold of total VRAM.
- [ ] Reduce peak memory by lowering batch size and/or max chunk length if needed.

## Run notes

- Attempted: `.venv/bin/python transcribe_parakeet.py path/to/audio.wav`
- Result: model downloaded successfully; transcription failed during Lhotse dataloader setup with `TypeError: object.__init__() takes exactly one argument`.
- Warnings: NeMo training/validation config warnings during model load; OneLogger telemetry disabled.
- Mitigation: updated `transcribe_parakeet.py` to pass `transcribe_cfg` with `use_lhotse=False` (plus `batch_size=1`, `num_workers=0`).
- New failure: `transcribe_cfg` unsupported; switching to disabling Lhotse via `asr_model.cfg` before calling `transcribe()` (no `transcribe_cfg` arg).
- Latest attempt (run by me): still hit Lhotse DynamicCutSampler error; NeMo's `_setup_transcribe_dataloader` hardcodes `use_lhotse=True`.
- Mitigation in progress: patch `_setup_transcribe_dataloader` at runtime to force `use_lhotse=False` and reuse the manifest path.
- Success: runtime patch applied; transcription completed with full text and timestamps (CPU mode; CUDA not available warning).
- Decision: move to containerized workflow (CUDA + Python 3.11) for stable GPU support and reproducible builds; host environment is not suitable for long-term GPU dev.
- Container build failed: `pip` used only the PyTorch index, so `nemo_toolkit==2.6.1` could not be resolved. Fix: add `--extra-index-url https://pypi.org/simple` in `Dockerfile`.
- Container run success: `docker compose up --build` produced timestamped transcription for `path/to/audio.wav` and exited with code 0.
- GPU verified in container: `torch 2.4.1+cu121`, `cuda available: True`, device `NVIDIA GeForce RTX 3090`.
