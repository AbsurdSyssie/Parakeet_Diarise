# Parakeet + Diarise

A GPU speech-to-text API with speaker diarization.

It exposes an OpenAI-style transcription endpoint, lets you switch ASR models at runtime, and can attach speaker labels to word-level transcripts.

The default diarization model is [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), loaded through NeMo's `SortformerEncLabelModel`.

## Features

- OpenAI-style `POST /v1/audio/transcriptions` endpoint
- Runtime-selectable ASR models
- Nemotron 3 speaker diarization
- Word, segment, or text-only timestamps
- Speaker attribution from ASR word timestamps
- Silero VAD with configurable chunking
- In-memory or file-backed chunk processing
- Docker setup for NVIDIA GPUs

Supported ASR backends include NVIDIA Parakeet and Nemotron ASR, Whisper, faster-whisper, Cohere Transcribe, and IBM Granite Speech.

## Quick start

Create your local environment file:

```bash
cp .env.example .env
```

Add a Hugging Face token if the model you use requires one:

```env
HF_TOKEN=...
```

Build and start the API:

```bash
docker compose build api
docker compose up api
```

The API listens on `http://localhost:8000`.

For live source mounts during development:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build api
```

The normal `compose.yaml` runs the code baked into the image; the development override mounts the checkout into `/app`.

## Transcribe audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json" \
  -F "timestamps=word" \
  -F "diarization=true"
```

A diarized response looks like this:

```json
{
  "text": "Hello world.",
  "language": "en",
  "duration": 12.34,
  "words": [
    {
      "word": "Hello",
      "start": 0.0,
      "end": 0.42,
      "speaker": "SPEAKER_00"
    }
  ],
  "segments": [
    {
      "id": 0,
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 1.2,
      "text": "Hello world."
    }
  ],
  "speakers": ["SPEAKER_00"]
}
```

Diarization requires `timestamps=word` because speaker turns are aligned to ASR word timestamps.

For the full request schema, VAD controls, and response formats, see [docs/api.md](docs/api.md).

## ASR models

The API loads one ASR model at a time. List the available models:

```bash
curl http://localhost:8000/v1/models
```

Show the active model:

```bash
curl http://localhost:8000/v1/models/current
```

Switch models without restarting the service:

```bash
curl -X POST http://localhost:8000/v1/models/current \
  -H "Content-Type: application/json" \
  -d '{"model":"parakeet-1.1b"}'
```

The curated model registry currently includes:

| Model | Backend | Timestamps | Diarization |
| --- | --- | --- | --- |
| Parakeet TDT 0.6B v3 | NeMo | word + segment | yes |
| Parakeet TDT 1.1B | NeMo | word + segment | yes |
| Nemotron 3.5 ASR Streaming 0.6B | NeMo | word + segment | yes |
| Medical Whisper Large v3 | Transformers | word + segment | yes |
| faster-whisper Large v3 | CTranslate2 | word + segment | yes |
| Cohere Transcribe | Transformers | text only | no |
| Granite Speech 4.1 2B NAR | Transformers | text only | no |

Text-only models cannot use diarization because the alignment stage needs word timestamps.

## Diarization

The main API runs diarization in-process.

By default it loads:

```env
DIARIZATION_MODEL=nvidia/Nemotron-3-Diarization
```

To use the previous four-speaker Sortformer checkpoint instead:

```env
DIARIZATION_MODEL=nvidia/diar_streaming_sortformer_4spk-v2.1
```

The application converts input audio to mono 16 kHz, runs the diarizer over the full recording, then assigns its speaker turns to ASR words.

See [docs/sortformer.md](docs/sortformer.md) for model and inference details.

## Configuration

Start from `.env.example`.

The main settings are:

| Variable | Purpose |
| --- | --- |
| `ASR_BACKEND` | ASR backend used at startup |
| `ASR_MODEL` | ASR model or registry key used at startup |
| `DIARIZATION_MODEL` | NeMo diarization checkpoint |
| `HF_TOKEN` | Hugging Face authentication |
| `VAD_DEVICE` | Run VAD on `cpu` or `cuda` |
| `VAD_SAMPLE_RATE` | Sample rate used by VAD |
| `DISABLE_CUDA_GRAPHS` | Disable NeMo RNNT CUDA graph decoding |

VAD thresholds, chunk sizes, overlap, padding, and energy-gate settings are documented in [docs/api.md](docs/api.md).

## Standalone diarization API

`diarize_api.py` can run diarization as a separate service.

It exposes:

- `POST /v1/diarize`
- `GET /health`

A diarization response is a list of speaker turns:

```json
[
  {
    "start": 0.0,
    "end": 1.23,
    "speaker": "SPEAKER_00"
  }
]
```

## Command-line transcription

The original command-line entry point remains available:

```bash
python transcribe_parakeet.py audio.wav
```

You can select another backend and model:

```bash
python transcribe_parakeet.py audio.wav \
  --backend transformers-asr \
  --model granite-speech-4.1-2b-nar \
  --timestamps none
```

## Tests

Automated tests live in `tests/`. Manual diagnostics live in `scripts/probes/`.

See [docs/tests.md](docs/tests.md) for the current test matrix and manual checks.

## Repository layout

| Path | Purpose |
| --- | --- |
| `api.py` | FastAPI routes and transcription orchestration |
| `settings.py` | Environment configuration |
| `model_registry.py` | Runtime model config and capability checks |
| `model_manager.py` | ASR model lifecycle, switching, and GPU state |
| `asr/` | ASR adapters, canonical model registry, shared types, and backend dispatch |
| `asr_backend.py` | Compatibility facade for the previous ASR import path |
| `chunk_transcribe.py` | Chunk transcription |
| `vad_chunk.py` | VAD and chunk generation |
| `diarize_align.py` | Speaker-to-word alignment |
| `diarize_api.py` | Optional standalone diarization API |
| `scripts/` | CLI utilities and manual probes |
| `tests/` | Automated unit tests |
| `docs/` | Current API, architecture, model, and test documentation |
| `docs/archive/` | Historical implementation notes and progress logs |
| `legacy/` | Unsupported executable experiments kept for reference |
| `Dockerfile` | CUDA/Python runtime |
| `compose.yaml` | Built-image runtime service |
| `compose.dev.yaml` | Development bind-mount override |

## Further reading

- [API guide](docs/api.md)
- [Diarization](docs/sortformer.md)
- [Parakeet notes](docs/parakeet.md)
- [Granite Speech](docs/granite.md)
- [Audio tracing](docs/trace_audio.md)
- [Tests](docs/tests.md)
- [Architecture](docs/architecture.md)

Historical implementation notes live under `docs/archive/`; unsupported executable experiments live under `legacy/`.
