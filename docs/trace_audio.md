# Audio Trace (Server-Side)

Goal: trace how an MP3 moves through the ASR API and confirm each stage behaves as expected.

## Pipeline map (server-side)

1) Request ingest
   - Endpoint: `POST /v1/audio/transcriptions`
   - Input: multipart form + audio file (MP3/WAV/FLAC)
   - Expected: file bytes saved to a temp path with the original filename.

2) Decode to waveform
   - `torchaudio.load()` → waveform + sample rate
   - Expected: waveform tensor, correct duration, channel count.

3) Mixdown + resample
   - If stereo: average to mono
   - If not 16 kHz: resample to 16 kHz for ASR
   - Expected: mono tensor, 16 kHz, duration preserved.

4) VAD sample-rate branch
   - If `VAD_SAMPLE_RATE != 16000`, resample to VAD SR
   - Expected: separate VAD waveform with target SR.

5) VAD chunking
   - `run_vad_chunks_in_memory_from_waveform(...)`
   - Expected: list of speech chunks with `start_s`, `end_s`, and chunk waveforms.

6) ASR chunk input
   - Each chunk waveform fed to Parakeet
   - Expected: per-chunk timestamps starting at 0.0, then offset by `start_s`.

7) Merge + response
   - Merge words into global timeline
   - Optional diarization merge
   - Response assembled

## Tests (baby steps)

Each test should validate one stage in isolation. Start with the earliest stages and add logs only where needed.

### Test 1 — Request ingest
Goal: confirm the file bytes are written to disk and size matches upload.
How:
- Log file size (bytes) and original filename in the API.
Expected:
- Logged file size matches local file size.
Status: passed (2026-01-28)

### Test 2 — Decode integrity (MP3 → waveform)
Goal: confirm MP3 decodes correctly and duration matches expectations.
How:
- Log `sr`, `channels`, `num_samples`, `duration`.
Expected:
- Duration aligns with known file duration.
Status: passed (2026-01-28)

### Test 3 — Mixdown + resample
Goal: confirm mono conversion and 16 kHz resample are correct.
How:
- Log before/after `sr`, `channels`, `num_samples`, `duration`.
Expected:
- Channels = 1, SR = 16000, duration preserved.
Status: passed (2026-01-28)

### Test 4 — VAD SR branch
Goal: confirm VAD resample path when `VAD_SAMPLE_RATE != 16000`.
How:
- Set `VAD_SAMPLE_RATE=8000` and log VAD waveform stats.
Expected:
- VAD SR is 8000, duration preserved.
Status: passed (2026-01-28)

### Test 5 — VAD chunk output
Goal: confirm VAD returns multiple chunks (or explain why it doesn’t).
How:
- Log number of chunks, first/last chunk boundaries, and total covered time.
Expected:
- For conversational audio, multiple chunks or uniform chunks if energy-gate skip.
Status: passed (2026-01-28)

### Test 5b — Force VAD (ignore RMS gate)
Goal: confirm `force_vad=on` bypasses the RMS energy gate and runs Silero VAD directly.
How:
- Run the same request twice:
  1) `force_vad=off` (default)
  2) `force_vad=on`
- Compare chunk counts and boundaries.
Expected:
- With `force_vad=on`, logs should show *no* “energy gate skip” line and Silero should run on full audio.
- Chunk counts may differ (e.g., fewer or longer chunks if Silero returns a single interval).
Status: passed (2026-01-28)

### Test 6 — ASR input shape
Goal: confirm each chunk waveform shape and duration align with VAD boundaries.
How:
- Log per-chunk sample count + computed duration.
Expected:
- Chunk duration roughly equals `(end_s - start_s) + padding`.
Status: not run

### Test 7 — Merge correctness (sanity)
Goal: confirm global word timestamps are monotonic and within audio duration.
How:
- Validate first/last word timestamps, monotonic order.
Expected:
- No negative timestamps, last word <= duration.
Status: not run

## Proposed logging format (server-side)

Use a short `trace_id` per request and log:
- `ingest`: filename, bytes
- `decode`: sr, channels, num_samples, duration
- `resample`: sr_before, sr_after, channels_before/after
- `vad`: vad_sr, chunk_count, first/last chunk boundaries
- `asr`: batch_size, chunks_transcribed

## Notes

- We should only log debug data when a `trace_audio=true` flag is provided to avoid noisy production logs.
- If needed, dump a single chunk waveform to disk for deeper inspection (optional, not default).
