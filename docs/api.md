# API Guide

This repo exposes an OpenAI-style transcription endpoint with optional diarization.

## Services

- ASR API: `http://localhost:8000`
- Diarization runs in-process inside the ASR API via Sortformer when `diarization=true`.

## Build + run

```bash
docker compose build api
docker compose up api
```

## Current status

- ASR API loads Parakeet once at startup and logs per-request transcription time.
- Silero VAD is cached in-process to avoid reloading on each request.
- VAD chunking uses hard-cuts with overlap (30s max, 1.0s overlap) for continuous audio.

## Endpoint

`POST /v1/audio/transcriptions` (multipart/form-data)

Parameters (multipart/form-data):

| Parameter | Allowed values | Default | Notes |
| --- | --- | --- | --- |
| `file` | audio file | required | WAV/FLAC/MP3 tested. |
| `response_format` | `verbose_json` | `verbose_json` | Only supported format. |
| `diarization` | `true` or `false` | `false` | Requires `timestamps=word`. |
| `timestamps` | `word`, `segment`, `none` | `word` | `word` required for diarization. |
| `language` | `en` | `en` | Passed through to response. |
| `chunk_mode` | `memory`, `file` | `memory` | `file` writes chunks to disk. |
| `chunk_only` | `true` or `false` | `false` | Returns VAD chunks without ASR. |
| `trace_audio` | `true` or `false` | `false` | Enables request trace logs. |
| `force_vad` | `off`, `on` | `off` | `on` forces Silero VAD and ignores energy gate. |

Optional VAD overrides (per-request). If omitted, environment defaults apply:

| Parameter | Allowed values | Default |
| --- | --- | --- |
| `vad_sample_rate` | int | env `VAD_SAMPLE_RATE` (default `16000`) |
| `vad_threshold` | float | env `VAD_THRESHOLD` (default `0.38`) |
| `vad_min_speech_ms` | int | env `VAD_MIN_SPEECH_MS` (default `250`) |
| `vad_min_silence_ms` | int | env `VAD_MIN_SILENCE_MS` (default `300`) |
| `vad_merge_gap_ms` | int | env `VAD_MERGE_GAP_MS` (default `200`) |
| `vad_target_min_s` | float | env `VAD_TARGET_MIN_S` (default `10.0`) |
| `vad_target_max_s` | float | env `VAD_TARGET_MAX_S` (default `20.0`) |
| `vad_hard_max_s` | float | env `VAD_HARD_MAX_S` (default `30.0`) |
| `vad_overlap_s` | float | env `VAD_OVERLAP_S` (default `1.0`) |
| `vad_speech_pad_ms` | int | env `VAD_SPEECH_PAD_MS` (default `0`) |
| `vad_energy_gate` | `true` or `false` | env `VAD_ENERGY_GATE` (default `0`) |
| `vad_energy_db` | float | env `VAD_ENERGY_DB` (default `-35`) |
| `vad_energy_frame_ms` | int | env `VAD_ENERGY_FRAME_MS` (default `100`) |
| `vad_energy_min_active_ms` | int | env `VAD_ENERGY_MIN_ACTIVE_MS` (default `500`) |
| `vad_energy_merge_gap_ms` | int | env `VAD_ENERGY_MERGE_GAP_MS` (default `800`) |
| `vad_energy_active_skip` | float | env `VAD_ENERGY_ACTIVE_SKIP` (default `0.85`) |
| `vad_uniform_chunk_s` | float | env `VAD_UNIFORM_CHUNK_S` (default `30`) |
| `vad_uniform_overlap_s` | float | env `VAD_UNIFORM_OVERLAP_S` (default `1.0`) |

Note: The FastAPI `/docs` page now enumerates these options and defaults directly in the schema.

Example:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@path/to/audio.wav" \
  -F "response_format=verbose_json" \
  -F "diarization=true" \
  -F "timestamps=word" \
  -F "language=en" \
  -F "chunk_mode=memory"
```

Chunk-only example (debug VAD chunk boundaries without ASR):

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@path/to/audio.wav" \
  -F "response_format=verbose_json" \
  -F "timestamps=none" \
  -F "chunk_mode=memory" \
  -F "chunk_only=true"
```

## Response schema

Top-level:

```json
{
  "text": "Hello world.",
  "language": "en",
  "duration": 12.34,
  "words": [...],
  "segments": [...],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

`words[]` (canonical timeline):

```json
{
  "word": "Hello",
  "start": 0.00,
  "end": 0.42,
  "speaker": "SPEAKER_00"
}
```

Rules:

- sorted by `start`
- timestamps are global
- `speaker` optional but preferred

`segments[]` (derived, human-friendly):

```json
{
  "id": 3,
  "speaker": "SPEAKER_01",
  "start": 4.21,
  "end": 7.88,
  "text": "I think that's correct."
}
```

Rules:

- contiguous words
- same speaker
- gap threshold (default `0.6s`)

### Chunk-only response

If `chunk_only=true`, the endpoint skips ASR and returns VAD chunk metadata:

```json
{
  "ok": true,
  "chunk_only": true,
  "chunk_mode": "memory",
  "duration": 123.45,
  "sample_rate": 16000,
  "vad_sample_rate": 16000,
  "vad_params": {
    "threshold": 0.5,
    "min_speech_ms": 250,
    "min_silence_ms": 300,
    "merge_gap_ms": 800,
    "target_min_s": 10.0,
    "target_max_s": 30.0,
    "hard_max_s": 45.0,
    "overlap_s": 0.5,
    "speech_pad_ms": 200
  },
  "chunks": [
    {
      "index": 1,
      "start_s": 0.34,
      "end_s": 44.92,
      "duration_s": 44.58,
      "effective_start_s": 0.14,
      "effective_end_s": 45.12,
      "effective_duration_s": 44.98,
      "sample_rate": 16000,
      "num_samples": 720000
    }
  ]
}
```

## Health checks

- `GET /health` on the ASR API returns basic service state (model loaded + CUDA memory snapshot).

## Notes

- `.env` must contain `HF_TOKEN` for diarization. Use `.env.example` as a template.
- Current default VAD tuning: threshold 0.38, target_max 20s, hard_max 30s, overlap 1.0s, merge_gap 200ms, speech_pad 0ms.
- VAD chunking uses 0ms padding by default.
- Default `chunk_mode=memory` avoids writing WAVs to disk, reduces decode/IO overhead, and keeps GPU utilization higher.
- Use `chunk_mode=file` only if you need chunk WAVs persisted for debugging or if you suspect in-memory batching is unstable.
- `diarization=true` requires `timestamps=word` because speaker alignment uses word-level timestamps.
- In-memory chunking is the default for the API; file-based chunking is still available via `chunk_mode=file`.
- Short `UNKNOWN` speaker segments (<=0.6s or <=2 words) are merged into neighboring segments; if the same speaker appears on both sides, the segments are coalesced.
- Local Sortformer experiments live in `test/sortformer_dry.py` and `test/sortformer_align_dry.py`.
- `VAD_DEVICE` can force VAD to CPU or CUDA.
- `VAD_SAMPLE_RATE` lets you run VAD at 8 kHz while keeping ASR at 16 kHz. Timestamps remain in seconds.
- `VAD_ENERGY_GATE=1` enables a cheap RMS energy gate before Silero. It scans 100 ms frames on CPU and only runs Silero on candidate regions. Tunables: `VAD_ENERGY_DB` (default `-35`), `VAD_ENERGY_FRAME_MS` (default `100`), `VAD_ENERGY_MIN_ACTIVE_MS` (default `500`), `VAD_ENERGY_MERGE_GAP_MS` (default `800`).
- If the energy gate estimates active speech ratio >= `VAD_ENERGY_ACTIVE_SKIP` (default `0.85`), Silero is skipped and uniform chunks are created with `VAD_UNIFORM_CHUNK_S` (default `30`) and `VAD_UNIFORM_OVERLAP_S` (default `1.0`). These uniform settings are independent from Silero chunking defaults.
- When the skip threshold is hit, Silero is not run at all; uniform chunks are treated as speech directly.
- `DIARIZE_EMPTY_CACHE=1` runs `gc.collect()` after diarization and resets CUDA peak stats (model weights remain resident).
- `DIARIZE_TF32=1` enables TF32 in CUDA matmul/cudnn for diarization to improve throughput at the cost of strict determinism.
- Recommendation: keep `DIARIZE_TF32=0` for now; we did not see a consistent speedup.
- `DISABLE_CUDA_GRAPHS=1` disables NeMo RNNT CUDA graph decoding to mitigate intermittent CUDA illegal memory access errors.
- Energy gate fix validated via an in‑container 1s silence test using `run_vad_chunks_in_memory_from_waveform`.

## Recent benchmarks

Memory mode, `path/to/audio.wav`, diarization=false:
- `timestamps=word`: 8.86s
- `timestamps=segment`: 8.73s (~1.5% faster vs word)
- `timestamps=none`: 7.62s (~14% faster vs word)

Memory mode, `path/to/audio.wav`, diarization=true:
- 16 kHz VAD, `VAD_DEVICE=cpu`: decode ~1.00s; VAD 5.95s; ASR 3.60s; diarization 10.04s; total 20.52s
- 8 kHz VAD, `VAD_DEVICE=cuda`: decode 0.89s; VAD 7.40s; ASR 2.76s; diarization 9.88s; total 21.07s

Energy gate test (1h audio, `VAD_ENERGY_GATE=1`, `VAD_DEVICE=cpu`, `VAD_SAMPLE_RATE=8000`):
- energy gate found 2 intervals; decode 0.28s; VAD 25.60s; ASR 18.86s; diarization 49.51s; total 95.09s
- Current energy gate settings in `.env`: `VAD_ENERGY_DB=-35`, `VAD_ENERGY_FRAME_MS=100`, `VAD_ENERGY_MIN_ACTIVE_MS=500`, `VAD_ENERGY_MERGE_GAP_MS=800`.
- Proposed: run the cheap energy gate to estimate speech ratio; if active > ~70% of duration, skip Silero and chunk uniformly (30–45s with overlap) before ASR.
Energy gate skip test (1h audio, ratio=1.00, `VAD_ENERGY_ACTIVE_SKIP=0.85`):
- uniform intervals: 80; decode 0.30s; VAD 0.48s; ASR 15.61s; diarization 50.72s; total 67.86s
- Takeaway: energy-gate skip removes most VAD cost, but diarization dominates total time.

TF32 diarization comparison (same workload):
- `DIARIZE_TF32=1`: diarization 54.40s / 53.11s; total 70.44s / 69.09s
- `DIARIZE_TF32=0`: diarization 51.96s / 53.33s; total 68.26s / 69.26s
- Conclusion: no consistent speedup from TF32 in this run.
