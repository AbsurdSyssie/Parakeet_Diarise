# pyannote (speaker-diarization-community-1) – Installation and Usage (GPU)

This document provides **developer-focused** instructions for installing and running **pyannote** diarization locally (Python) with optional **GPU** acceleration.

Pipeline target:

* `pyannote/speaker-diarization-community-1`

---

## 1) What this pipeline does

`community-1` is a pretrained **speaker diarization** pipeline: it outputs **who spoke when** as speaker-labeled time segments.

Audio expectations (handled automatically, but recommended to comply upfront):

* **Mono** audio
* **16 kHz** sample rate

If the input is stereo/multi-channel, it is downmixed to mono; if the sample rate differs, it is resampled to 16 kHz when loading.

---

## 2) Prerequisites

### Hugging Face access

This pipeline is hosted on the Hugging Face Hub and typically requires:

* Accepting the model/pipeline user conditions on the model page
* A Hugging Face access token (environment variable recommended)

Environment variable:

* `HF_TOKEN` (your Hugging Face token)

### System packages (recommended)

`ffmpeg` is recommended for audio decoding support.

```bash
apt-get update && apt-get install -y ffmpeg
```

---

## 3) Installation

### Python environment

* Python 3.10+ recommended
* PyTorch installed (CPU or CUDA build)

Install pyannote:

```bash
pip install -U pyannote.audio
```

Notes:

* GPU acceleration depends on using a CUDA-enabled PyTorch install.

---

## 4) Quick GPU sanity check

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
PY
```

---

## 5) Load the pipeline (local inference)

```python
import os
import torch
from pyannote.audio import Pipeline

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN env var is required for gated/private models.")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    use_auth_token=HF_TOKEN,
)

# Move pipeline to GPU if available
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
```

---

## 6) Run diarization

### Basic diarization

```python
diari = pipeline("/path/to/audio.wav")
```

`diari` is a `pyannote.core.Annotation` with speaker-labeled time segments.

### Constrain speaker count (optional)

```python
diari = pipeline(
    "/path/to/audio.wav",
    min_speakers=1,
    max_speakers=4,
)
```

---

## 7) Convert output to a JSON-friendly segment list

```python
segments = []
for turn, _, speaker in diari.itertracks(yield_label=True):
    segments.append({
        "speaker": speaker,
        "start": float(turn.start),
        "end": float(turn.end),
    })

# Optional: sort by time
segments.sort(key=lambda s: (s["start"], s["end"]))
```

This produces a minimal structure suitable for merging with ASR timestamps.

---

## 8) Exclusive diarization (no overlaps)

### Cloud API concept

pyannote’s hosted API documentation describes an **exclusive diarization** output (diarization without overlapping speech) returned under `exclusiveDiarization`.

### Local pipeline note

The local Python pipeline returns an `Annotation` which can include overlaps in challenging audio (overlapping speech). If your downstream merge logic benefits from non-overlapping speaker turns, you can post-process diarization into an “exclusive” representation by applying an overlap-resolution policy (e.g., selecting the highest-confidence speaker per frame, or converting to non-overlapping turns). The exact method depends on the pipeline version and what internal scores are exposed.

---

## 9) Performance notes (GPU)

* Load the pipeline **once** at process startup (FastAPI startup hook) and reuse for requests.
* Limit concurrency on a single GPU to avoid OOM.
* For long audio, consider async processing and/or chunking (diarization quality generally benefits from longer context, so chunking should be designed carefully).

---

## 10) Common failure modes

* ❌ Missing/invalid `HF_TOKEN` or user conditions not accepted
* ❌ CPU-only PyTorch installed (pipeline runs on CPU)
* ❌ Missing `ffmpeg` leading to decode failures for non-WAV formats

---

## 11) Minimal sanity test

```python
assert segments, "Expected at least one diarization segment"
assert all(s["end"] > s["start"] for s in segments)
```

---

## 12) Primary references (URLs)

* Community-1 model card: [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
* pyannote.audio PyPI page (ffmpeg/token notes): [https://pypi.org/project/pyannote-audio/](https://pypi.org/project/pyannote-audio/)
* pyannote.audio repository: [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
* pyannote hosted API diarization reference (exclusive diarization field): [https://docs.pyannote.ai/api-reference/diarize](https://docs.pyannote.ai/api-reference/diarize)
