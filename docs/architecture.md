# Architecture

The service is split by responsibility so model-specific code does not leak into the HTTP layer.

## Request path

```text
FastAPI request
    |
    v
api.py
    |
    +--> vad_chunk.py
    |
    +--> model_manager.py --> asr/
    |
    +--> diarization.py
    |
    +--> diarize_align.py
    |
    v
api_response.py
```

## Modules

### `api.py`

Owns HTTP routes, request validation at the protocol boundary, VAD orchestration, and response assembly. It does not load or own ASR models.

### `settings.py`

Loads `.env` and environment variables once and exposes typed application settings.

### `asr/`

Contains the ASR adapters and the canonical model registry.

- `registry.py`: model IDs, repository names, capabilities, and aliases
- `types.py`: shared ASR configuration and hypothesis types
- `loader.py`: backend dispatch
- backend modules: implementation-specific adapters

`asr_backend.py` remains only as a compatibility facade for older imports.

### `model_registry.py`

Turns canonical ASR model specs into runtime configuration. It also handles timestamp capability checks and diarization eligibility.

It does not duplicate model repository names; those live in `asr/registry.py`.

### `model_manager.py`

Owns the active ASR model, its lock, switching, failed-switch restoration, CUDA cleanup, and model state reported by the API.

### `diarization.py`

Owns the diarization model, inference lock, audio preparation, output parsing, and speaker-label normalization.

Both the main API and standalone diarization API call this module.

### `chunk_transcribe.py` and `asr_merge.py`

Own chunk transcription and timeline merging.

### `vad_chunk.py`

Owns speech detection and chunk generation.

## State ownership

There are two long-lived model owners:

- `MODEL_MANAGER` owns ASR state.
- `diarization.py` owns diarization state.

The API reads those owners; it does not maintain duplicate model globals.
