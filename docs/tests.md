# Tests

Automated unit tests live in `tests/`. Manual API/model diagnostics live in `scripts/probes/`.

Run the dependency-light core suite with:

```bash
python -m unittest \
  tests.test_api_response \
  tests.test_asr_merge \
  tests.test_diarize_align \
  tests.test_merge_diarized \
  tests.test_health_endpoints
```

The full unit suite includes tests that import PyTorch and ASR adapters and should be run in the project environment:

```bash
python -m unittest discover -s tests
```

## Current automated coverage

- `tests/test_api_response.py`: response shape, text override, and speaker extraction
- `tests/test_asr_backend.py`: backend aliases/adapters, Granite behaviour, audio preparation, and device validation
- `tests/test_asr_merge.py`: chunk metadata, timestamp offsets, ordering, and overlap deduplication
- `tests/test_asr_package.py`: compatibility of the legacy `asr_backend` import facade
- `tests/test_chunk_transcribe.py`: chunk hypothesis offsets and timestamp-free overlap merging
- `tests/test_diarization.py`: speaker normalization and diarization output parsing
- `tests/test_diarize_align.py`: speaker assignment and segment grouping
- `tests/test_health_endpoints.py`: presence of health routes and diarization health state
- `tests/test_merge_diarized.py`: end-to-end word/turn alignment helpers
- `tests/test_model_registry.py`: startup model resolution and capability rules
- `tests/test_model_manager.py`: model-switch failure and restoration invariants

## Manual probes

See `scripts/probes/README.md`. Probe output defaults to `tmp/probes/` and is ignored by Git.

## Integration checks still worth keeping

## 1) Silence handling (artifact check)
Goal: ensure silence or low‑energy sections do **not** hallucinate tokens (e.g., “Aaron”).
Setup:
- Use a file with long silence + low background music.
- Run with `force_vad=on` and `force_vad=off`.
Assertions:
- No words emitted within long silence regions (allow short padding tolerance).
- No repeated hallucinated tokens at chunk boundaries.

## 2) VAD mode parity (energy gate vs forced VAD)
Goal: ensure VAD chunking behavior is consistent and explainable across modes.
Setup:
- Run `chunk_only=true` with `force_vad=off` and `force_vad=on` on the same file.
Assertions:
- When `force_vad=on`, RMS gate is bypassed (no “energy gate skip” in logs).
- Chunk boundaries differ only as expected (speech‑aligned vs uniform).

## 3) Chunk duration correctness
Goal: chunk waveforms fed to ASR match expected durations from boundaries + padding.
Setup:
- Use `trace_audio=true` and capture `chunk_duration` logs.
Assertions:
- `abs(actual_s - expected_s) <= 0.01` for first/last chunks.

## 4) Timestamp monotonicity
Goal: merged word timestamps are strictly non‑decreasing and within audio duration.
Setup:
- Run with `timestamps=word` and merge results.
Assertions:
- `start >= 0`, `end >= start`, and `last_end <= duration`.

## 5) Overlap deduplication
Goal: overlapping chunks do not duplicate words in merged output.
Setup:
- Use a file with short speech at chunk boundaries and overlap enabled.
Assertions:
- No repeated words with identical text and overlapping time windows.

## 6) Diarization alignment sanity
Goal: speaker labels align to words without gaps or invalid labels.
Setup:
- Run with `diarization=true`, `timestamps=word`.
Assertions:
- All words have a `speaker` value.
- Speaker labels are in the returned `speakers` list.

## 7) Format decoding
Goal: MP3/WAV/FLAC decode paths are consistent.
Setup:
- Same audio in MP3/WAV/FLAC form.
Assertions:
- Duration within 0.01s across formats.
- Transcript similarity above a chosen threshold (simple word overlap).

## 8) Chunk-only API response shape
Goal: `chunk_only=true` returns a valid chunk list without ASR output.
Setup:
- Run chunk-only request.
Assertions:
- Response includes `chunks`, `vad_params`, and `duration`; text/segments/words omitted.

## 9) Energy gate skip threshold
Goal: verify `VAD_ENERGY_ACTIVE_SKIP` behavior.
Setup:
- High‑energy audio (music) and low‑energy audio (speech with silence).
Assertions:
- High‑energy audio triggers uniform chunks (skip).
- Low‑energy audio runs Silero VAD (no skip).

## 10) Request regression guard
Goal: ensure no new flags break existing clients.
Setup:
- Default request (no `force_vad`, no `chunk_only`).
Assertions:
- Same response schema and status as previous baseline.
