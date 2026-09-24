# API Guide

This repo exposes an OpenAI-style transcription endpoint with optional diarization.

## Services

- ASR API: `http://localhost:$API_PORT` (`8000` by default)
- Diarization runs in-process inside the ASR API through NeMo `SortformerEncLabelModel` when `diarization=true`. The default model is `nvidia/Nemotron-3-Diarization`.

## Build + run

Set the API port and startup models in `.env`:

```env
API_PORT=8000
ASR_BACKEND=nemo
ASR_MODEL=parakeet-1.1b
DIARIZATION_MODEL=nvidia/Nemotron-3-Diarization
```

Then run:

```bash
docker compose up --build api
```

Compose publishes `API_PORT` and starts Uvicorn on the same container port. Shell variables can override `.env` for one run:

```bash
API_PORT=9000 ASR_MODEL=parakeet-0.6b docker compose up --build api
```

Without Docker:

```bash
python scripts/run_api.py
python scripts/run_api.py --port 9000
```

For a development bind mount:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build api
```

## Current status

- ASR API loads the selected ASR adapter once at startup and logs per-request transcription time.
- Silero VAD is cached in-process to avoid reloading on each request.
- VAD chunking uses hard-cuts with overlap (30s max, 1.0s overlap) for continuous audio.

## ASR model selection

Startup selection uses `ASR_BACKEND` and `ASR_MODEL`. For Granite:

```env
ASR_BACKEND=transformers-asr
ASR_MODEL=granite-speech-4.1-2b-nar
ASR_TRUST_REMOTE_CODE=1
ASR_RETURN_TIMESTAMPS=0
ASR_ATTENTION_IMPLEMENTATION=sdpa
```

List and switch curated models:

```bash
curl http://localhost:8000/v1/models

curl -X POST http://localhost:8000/v1/models/current \
  -H 'Content-Type: application/json' \
  -d '{"model":"granite-speech-4.1-2b-nar"}'
```

Granite Speech 4.1 2B NAR returns transcript text without word or segment
timestamps. Requests therefore use `timestamps=none`, and
`diarization=true` is rejected for this model.

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
| `force_vad` | `off`, `on` | `on` | `on` forces Silero VAD and ignores energy gate. |

Optional VAD overrides (per-request). If omitted, environment defaults apply:

| Parameter | Allowed values | Default |
| --- | --- | --- |
| `vad_sample_rate` | int | env `VAD_SAMPLE_RATE` (default `16000`) |
| `vad_threshold` | float | env `VAD_THRESHOLD` (default `0.30`) |
| `vad_min_speech_ms` | int | env `VAD_MIN_SPEECH_MS` (default `150`) |
| `vad_min_silence_ms` | int | env `VAD_MIN_SILENCE_MS` (default `220`) |
| `vad_merge_gap_ms` | int | env `VAD_MERGE_GAP_MS` (default `200`) |
| `vad_target_min_s` | float | env `VAD_TARGET_MIN_S` (default `10.0`) |
| `vad_target_max_s` | float | env `VAD_TARGET_MAX_S` (default `20.0`) |
| `vad_hard_max_s` | float | env `VAD_HARD_MAX_S` (default `30.0`) |
| `vad_overlap_s` | float | env `VAD_OVERLAP_S` (default `1.0`) |
| `vad_speech_pad_ms` | int | env `VAD_SPEECH_PAD_MS` (default `250`) |
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

- Set `DIARIZATION_MODEL` to override the default `nvidia/Nemotron-3-Diarization`. The previous `nvidia/diar_streaming_sortformer_4spk-v2.1` model remains usable as a fallback.
- Configure `HF_TOKEN` when the selected Hugging Face model requires authenticated access. Use `.env.example` as a template.
- Current API defaults: threshold 0.30, target_max 20s, hard_max 30s, overlap 1.0s, merge_gap 200ms, speech_pad 250ms.
- `force_vad=on` is the request default, so the RMS energy gate is bypassed unless `force_vad=off` is requested.
- Default `chunk_mode=memory` avoids writing WAVs to disk, reduces decode/IO overhead, and keeps GPU utilization higher.
- Use `chunk_mode=file` only if you need chunk WAVs persisted for debugging or if you suspect in-memory batching is unstable.
- `diarization=true` requires `timestamps=word` because speaker alignment uses word-level timestamps.
- In-memory chunking is the default for the API; file-based chunking is still available via `chunk_mode=file`.
- Short `UNKNOWN` speaker segments (<=0.6s or <=2 words) are merged into neighboring segments; if the same speaker appears on both sides, the segments are coalesced.
- Manual diarization probes live in `scripts/probes/diarization_dry.py` and `scripts/probes/diarization_align.py`; both honor `DIARIZATION_MODEL`.
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
