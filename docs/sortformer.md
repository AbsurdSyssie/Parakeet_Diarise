# Speaker diarization

The API uses NVIDIA NeMo's `SortformerEncLabelModel` for speaker diarization.

## Default model

The default checkpoint is:

```text
nvidia/Nemotron-3-Diarization
```

Nemotron 3 Diarization supports streaming and offline-style inference and up to 8 speakers. The application keeps the existing high-latency/offline-style streaming settings so the rest of the ASR + word-alignment pipeline does not need to change.

Override the model with:

```bash
DIARIZATION_MODEL=nvidia/Nemotron-3-Diarization
```

To roll back to the previous checkpoint:

```bash
DIARIZATION_MODEL=nvidia/diar_streaming_sortformer_4spk-v2.1
```

## Runtime requirements

NVIDIA's Nemotron 3 Diarization instructions require Python 3.12 or later and NeMo ASR support. The repository Docker image uses Python 3.12.

The shared runtime in `diarization.py` loads the selected model lazily on the first diarization request. Both `api.py` and `diarize_api.py` use that runtime, so model loading, audio preparation, speaker normalization, and output parsing have one implementation.

## Inference configuration

The application uses the model's existing `diarize(...)` interface:

```python
from nemo.collections.asr.models import SortformerEncLabelModel

model = SortformerEncLabelModel.from_pretrained(
    os.getenv("DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization")
)
model.eval()

model.sortformer_modules.chunk_len = 340
model.sortformer_modules.chunk_right_context = 40
model.sortformer_modules.fifo_len = 40
model.sortformer_modules.spkcache_update_period = 300
model._check_streaming_parameters()

predicted_segments = model.diarize(
    audio=["/path/to/mono-16khz.wav"],
    batch_size=1,
)
```

`diarization.py` converts audio to mono 16 kHz before inference. Speaker turns are normalized to `SPEAKER_00`, `SPEAKER_01`, and so on, then aligned to ASR word timestamps by `diarize_align.py`.

## Environment variables

- `DIARIZATION_MODEL`: Hugging Face/NeMo model ID used for diarization.
- `DIARIZE_TF32=1`: enable TF32 for the standalone diarization service.
- `DIARIZE_EMPTY_CACHE=1`: run garbage collection/reset CUDA peak stats after standalone diarization.
- `HF_TOKEN`: Hugging Face token when model access requires authentication.

## Dry runs

```bash
python scripts/probes/diarization_dry.py --audio path/to/audio.wav
python scripts/probes/diarization_align.py --audio path/to/audio.wav
```

Both scripts honor `DIARIZATION_MODEL`.
