# Parakeet ASR

This project supports NVIDIA Parakeet through the NeMo ASR backend. This page describes how Parakeet is used **in this repository**; installation, container, and API details live in the main README and API guide.

## Supported models

The canonical model registry currently exposes two Parakeet aliases:

| Alias | Model | Notes |
| --- | --- | --- |
| `parakeet-0.6b` | `nvidia/parakeet-tdt-0.6b-v3` | 0.6B multilingual TDT model |
| `parakeet-1.1b` | `nvidia/parakeet-tdt-1.1b` | 1.1B English TDT model |

`parakeet-0.6b-v3` supports 25 European languages and native word timestamps. NVIDIA describes the 1.1B checkpoint as an English model.

The repository treats both Parakeet entries as supporting word timestamps, segment timestamps, and diarization.

## Select a Parakeet model

For Docker or the API server, set the startup model in `.env`:

```env
ASR_BACKEND=nemo
ASR_MODEL=parakeet-1.1b
```

or:

```env
ASR_BACKEND=nemo
ASR_MODEL=parakeet-0.6b
```

Then start the service:

```bash
docker compose up --build api
```

You can override the model for one Compose run:

```bash
ASR_MODEL=parakeet-0.6b docker compose up --build api
```

You can also switch models while the API is running:

```bash
curl -X POST "http://localhost:${API_PORT:-8000}/v1/models/current" \
  -H "Content-Type: application/json" \
  -d '{"model":"parakeet-1.1b"}'
```

The service keeps one ASR model loaded at a time.

## Command-line transcription

`transcribe_parakeet.py` is now a generic ASR CLI despite its historical filename.

Use Parakeet explicitly with:

```bash
python transcribe_parakeet.py audio.wav \
  --backend nemo \
  --model parakeet-1.1b \
  --timestamps word
```

For the 0.6B model:

```bash
python transcribe_parakeet.py audio.wav \
  --backend nemo \
  --model parakeet-0.6b \
  --timestamps word
```

The CLI loads the model directly; it does not call the HTTP API and therefore does not use `API_PORT`.

## Timestamps and diarization

Parakeet is used for the ASR timeline. When `timestamps=word`, the API merges chunk-level word timestamps into one global timeline.

With:

```text
diarization=true
timestamps=word
```

the flow is:

```text
audio
  -> VAD/chunking
  -> Parakeet ASR words
  -> Nemotron 3 diarization turns
  -> speaker-to-word alignment
  -> OpenAI-style response
```

Parakeet does **not** provide the speaker labels. Speaker turns come from the diarization model configured by `DIARIZATION_MODEL`.

See [sortformer.md](sortformer.md) for diarization details.

## Audio handling in this project

Clients do not need to pre-convert every file to a 16 kHz mono WAV before using the API.

The service:

- decodes the uploaded audio;
- downmixes multi-channel audio to mono;
- resamples the ASR waveform to 16 kHz;
- optionally applies VAD and chunking;
- sends waveform chunks to the active ASR adapter.

WAV, FLAC, and MP3 are the formats exercised by the API path. FFmpeg and libsndfile are installed in the Docker image.

For direct model experiments outside the project pipeline, follow the input requirements on NVIDIA's model card.

## NeMo integration

Parakeet models are loaded through:

```python
nemo.collections.asr.models.ASRModel.from_pretrained(...)
```

The current code path is:

```text
asr/registry.py
    -> resolves parakeet alias

asr/nemo.py
    -> loads the NeMo checkpoint

model_manager.py
    -> owns model lifecycle and GPU state

chunk_transcribe.py
    -> performs chunk transcription and timestamp merging
```

The project also applies its no-Lhotse transcription dataloader patch before NeMo inference. Keep that behaviour in the application code rather than copying old setup snippets from archived notes.

## Runtime environment

The supported environment is defined by the repository, not by this document:

- `Dockerfile` defines the CUDA base image and Python version;
- `requirements.txt` pins PyTorch, torchaudio, and the NeMo source revision;
- `.env.example` defines runtime configuration;
- `compose.yaml` defines the GPU service.

At present the Docker image uses Python 3.12. Older notes recommending Python 3.11 are historical and have been moved to `docs/archive/`.

## Model notes

### Parakeet 0.6B v3

NVIDIA's current model card describes `nvidia/parakeet-tdt-0.6b-v3` as a 600M-parameter multilingual ASR model covering 25 European languages, with automatic language detection, punctuation/capitalization, and word timestamps.

Model card: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>

### Parakeet 1.1B

NVIDIA describes `nvidia/parakeet-tdt-1.1b` as an approximately 1.1B-parameter English FastConformer TDT model.

Model card: <https://huggingface.co/nvidia/parakeet-tdt-1.1b>

## Related documentation

- [API guide](api.md)
- [Architecture](architecture.md)
- [Speaker diarization](sortformer.md)
- [Tests](tests.md)

Historical Parakeet setup and debugging notes are preserved under `docs/archive/`.
