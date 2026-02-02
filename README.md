# Parakeet + Diarise

ASR + diarization service built around Parakeet transcription and Sortformer diarization. The API exposes an OpenAI-style `/v1/audio/transcriptions` endpoint with optional diarization.

## Quick start

```bash
docker compose build api
docker compose up api
```

API guide: `docs/api.md`

## API (what to send + what you get back)

### Endpoints

ASR API (main service, `api.py`):

- `POST /v1/audio/transcriptions` (multipart/form-data)
- `GET /health`

Standalone diarization service (optional, `diarize_api.py`):

- `POST /v1/diarize` (multipart/form-data)
- `GET /health`

### `POST /v1/audio/transcriptions`

Required form fields:

- `file` (WAV/FLAC/MP3)
- `response_format=verbose_json` (only supported value)

Optional form fields:

- `diarization=true|false` (default `false`)
- `timestamps=word|segment|none` (default `word`)
- `language=en` (default `en`)
- `chunk_mode=memory|file` (default `memory`)
- `chunk_only=true|false` (default `false`)
- `trace_audio=true|false` (default `false`)
- `force_vad=off|on` (default `off`)
- Per-request VAD overrides:
  - `vad_sample_rate`, `vad_threshold`, `vad_min_speech_ms`, `vad_min_silence_ms`
  - `vad_merge_gap_ms`, `vad_target_min_s`, `vad_target_max_s`, `vad_hard_max_s`
  - `vad_overlap_s`, `vad_speech_pad_ms`
  - `vad_energy_gate`, `vad_energy_db`, `vad_energy_frame_ms`, `vad_energy_min_active_ms`
  - `vad_energy_merge_gap_ms`, `vad_energy_active_skip`
  - `vad_uniform_chunk_s`, `vad_uniform_overlap_s`

Validation behavior:

- `response_format` must be `verbose_json`
- `timestamps` must be `word`, `segment`, or `none`
- `chunk_mode` must be `memory` or `file`
- `diarization=true` requires `timestamps=word`
- `force_vad` must be `off` or `on`

Response (top-level):

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

If `timestamps=word`, `words[]` is populated and `segments[]` is derived from words.
If `timestamps=segment`, `segments[]` is populated and `words[]` is empty.
If `timestamps=none`, both arrays are empty and `text` is a plain string transcript.

Chunk-only response (`chunk_only=true`) returns VAD metadata only and skips ASR:

```json
{
  "ok": true,
  "chunk_only": true,
  "chunk_mode": "memory",
  "duration": 123.45,
  "sample_rate": 16000,
  "vad_sample_rate": 16000,
  "vad_params": { "...": "..." },
  "chunks": [ { "...": "..." } ]
}
```

### `POST /v1/diarize` (optional standalone service)

Required form fields:

- `file` (WAV/FLAC/MP3)

Response is a list of speaker turns:

```json
[
  { "start": 0.0, "end": 1.23, "speaker": "SPEAKER_00" }
]
```

Full request/response examples and schemas live in `docs/api.md`.

## Environment variables

Use `.env.example` as the template. Common settings:

- `HF_TOKEN` (required for diarization downloads)
- `DIARIZE_URL` (external diarize service URL if used)
- `DIARIZE_EMPTY_CACHE` (1 to `gc.collect()` + reset CUDA stats after diarize)
- `DIARIZE_TF32` (1 to enable TF32 in CUDA matmul/cudnn)
- `VAD_DEVICE` (`cpu` or `cuda`)
- `VAD_SAMPLE_RATE`
- `VAD_THRESHOLD`
- `VAD_MIN_SPEECH_MS`
- `VAD_MIN_SILENCE_MS`
- `VAD_MERGE_GAP_MS`
- `VAD_TARGET_MIN_S`
- `VAD_TARGET_MAX_S`
- `VAD_HARD_MAX_S`
- `VAD_OVERLAP_S`
- `VAD_SPEECH_PAD_MS`
- `VAD_ENERGY_GATE`
- `VAD_ENERGY_DB`
- `VAD_ENERGY_FRAME_MS`
- `VAD_ENERGY_MIN_ACTIVE_MS`
- `VAD_ENERGY_MERGE_GAP_MS`
- `VAD_ENERGY_ACTIVE_SKIP`
- `VAD_UNIFORM_CHUNK_S`
- `VAD_UNIFORM_OVERLAP_S`

See `docs/api.md` for detailed VAD/energy-gate behavior and defaults.

## Docs

- `docs/api.md` - endpoint, parameters, examples, response schema
- `docs/parakeet.md` - Parakeet notes and container guidance
- `docs/pyannote.md` - legacy pyannote notes
- `docs/sortformer.md` - Sortformer diarization notes
- `docs/trace_audio.md` - chunk tracing and diagnostics
- `docs/tests.md` - test plan + existing tests
- `docs/Parakeet_Progress.md` - progress log
- `docs/sortformer_progress.md` - diarization progress log
- `docs/VAD_energy_gate_issue.md` - VAD energy gate investigation
- `docs/brief.md` - project brief

## Tests

Unit/behavior tests live in `test/`. See `docs/tests.md` for what each test does and the planned test matrix.

## Repo layout

- `api.py` - ASR API service
- `diarize_api.py` - diarization API wiring
- `diarize_align.py` / `merge_diarized.py` - word/segment alignment helpers
- `vad_chunk.py` - VAD chunking logic
- `chunk_transcribe.py` / `asr_merge.py` - ASR chunking and merge utilities
- `Dockerfile`, `compose.yaml` - container setup

Ignored runtime artifacts (expected): `.venv/`, `Output/`, `__pycache__/`.
