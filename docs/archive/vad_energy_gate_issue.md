# Energy Gate NameError Handoff

## Summary

The ASR API (`/v1/audio/transcriptions`) crashes with:

```
NameError: name 'energy_active_ratio_skip' is not defined
```

This happens inside `run_vad_chunks_in_memory_from_waveform` in `vad_chunk.py` during the energy‑gate logging path.

## Current Environment

`.env` contains:

```
VAD_DEVICE=cpu
VAD_SAMPLE_RATE=8000
VAD_ENERGY_GATE=1
VAD_ENERGY_DB=-55
VAD_ENERGY_FRAME_MS=100
VAD_ENERGY_MIN_ACTIVE_MS=300
VAD_ENERGY_MERGE_GAP_MS=500
VAD_ENERGY_ACTIVE_SKIP=0.7
VAD_UNIFORM_CHUNK_S=45
VAD_UNIFORM_OVERLAP_S=0.5
```

## Repro

1) Start API container with the above `.env`.
2) Call `POST /v1/audio/transcriptions` (diarization true, timestamps word).
3) API returns 500, stack trace shows NameError in `vad_chunk.py`.

## Error Trace (excerpt)

```
File "/app/vad_chunk.py", line 493, in run_vad_chunks_in_memory_from_waveform
  f"skip_ratio={energy_active_ratio_skip}, chunk_s={uniform_chunk_s}, "
NameError: name 'energy_active_ratio_skip' is not defined
```

## Attempted Fixes

1) Added energy‑gate config vars to `run_vad_chunks_from_waveform`.
2) Added energy‑gate config vars to `run_vad_chunks_in_memory_from_waveform`.
3) Added config logging for energy‑gate to help verify env and gate behavior.
4) Found duplicate env var lines in `run_vad_chunks_from_waveform` (removed).

The error persists in the in‑memory path, which suggests the container is still running code without the new variables or the variables are not in scope at runtime.

## Suspected Causes

- Container not recreated after code changes, so older `vad_chunk.py` is still running.
- Variables are defined in one function but not in the in‑memory path (or shadowed by a reload).
- Multiple copies of `vad_chunk.py` or a stale file inside the container image.

## Confirmed Root Cause

`run_vad_chunks_in_memory_from_waveform` references `energy_active_ratio_skip`,
`uniform_chunk_s`, and `uniform_overlap_s`, but does not define them. Those env reads
exist (and were duplicated) in `run_vad_chunks_from_waveform`, which is why the
in‑memory path throws `NameError`.

## Fix (code change)

In `vad_chunk.py`, inside `run_vad_chunks_in_memory_from_waveform`, add:

```python
energy_active_ratio_skip = float(os.environ.get("VAD_ENERGY_ACTIVE_SKIP", "0.7"))
uniform_chunk_s = float(os.environ.get("VAD_UNIFORM_CHUNK_S", "45"))
uniform_overlap_s = float(os.environ.get("VAD_UNIFORM_OVERLAP_S", "0.5"))
```

Place these right after `energy_merge_gap_ms = ...` and before the first use.

## Cleanup (recommended)

Remove duplicated env reads in `run_vad_chunks_from_waveform`:

```python
energy_active_ratio_skip = ...
uniform_chunk_s = ...
uniform_overlap_s = ...
# delete the duplicate triplet below
```

## Validation

Ran the following inside the container to confirm the NameError is gone:

```bash
docker compose exec -T api python3.11 - <<'PY'
import os, torch
from vad_chunk import run_vad_chunks_in_memory_from_waveform

# 1 second of silence
wave = torch.zeros(1, 8000)
chunks = run_vad_chunks_in_memory_from_waveform(
    waveform=wave,
    sample_rate=8000,
    threshold=0.5,
    min_speech_ms=250,
    min_silence_ms=200,
    merge_gap_ms=500,
    target_min_s=3.0,
    target_max_s=30.0,
    hard_max_s=45.0,
    overlap_s=0.5,
    speech_pad_ms=200,
)
print("ok, chunks:", len(chunks))
PY
```

## What to Check Next

1) Verify the running container has the latest `vad_chunk.py`:

```
docker compose exec api python3.11 - <<'PY'
import inspect, vad_chunk
print(inspect.getsource(vad_chunk.run_vad_chunks_in_memory_from_waveform))
PY
```

2) Confirm the env vars inside the container:

```
docker compose exec api env | grep VAD_
```

3) Ensure the container is recreated after code edits:

```
docker compose up -d --force-recreate api
```

4) If still failing, add a temporary `print(locals())` at the top of `run_vad_chunks_in_memory_from_waveform` to verify `energy_active_ratio_skip` is bound.

## Files to Include for Handoff

- `vad_chunk.py` (energy gate + VAD chunking logic)
- `api.py` (ASR endpoint, VAD invocation, decode path)
- `.env` (VAD env vars)
- `docs/api.md` (documents VAD controls)
- `docs/Parakeet_Progress.md` (recent benchmark notes)
