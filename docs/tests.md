# Tests (Critical)

This file captures the most important tests to implement for ASR + VAD + diarization.

## Current unit/behavior tests
These are the tests that already exist in `test/` and what they validate:
- `test/asr_debug_levers.py`: CLI harness that sweeps ASR API parameters (VAD, chunking, timestamps) and writes per-run JSON plus a `summary.json` with suspect-word and boundary diagnostics. Use it to compare boundary artifacts across settings.
- `test/test_api_response.py`: Verifies `api_response.build_response` output shape, text override, and speaker list extraction.
- `test/test_chunk_transcribe.py`: Validates `_merge_hypotheses` offsets and timestamp shifting for concatenated chunk outputs.
- `test/test_health_endpoints.py`: Static checks that `/health` route decorators exist in `api.py` and `diarize_api.py`.
- `test/test_merge_diarized.py`: Checks word-to-speaker assignment and segment grouping in `diarize_align` (including empty-turn handling).

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
