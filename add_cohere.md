## Goal

Add a third backend:

```env
ASR_BACKEND=transformers-asr
ASR_MODEL=CohereLabs/cohere-transcribe-03-2026
```

This should coexist with:

```env
ASR_BACKEND=nemo      # Parakeet
ASR_BACKEND=whisper   # Whisper-specific models
```

The key design point: **do not reuse `WhisperASRBackend` for Cohere**. The current Whisper backend passes Whisper-only generation options like `return_token_timestamps`, chunk windowing, and task/language generation kwargs. 

---

## Current state to account for

### `asr_backend.py`

Current backend handling:

* NeMo aliases for Parakeet
* Whisper aliases for Medical Whisper
* `resolve_asr_model()` only accepts `nemo`, `whisper`, `transformers`, and `transformers-whisper`
* `load_asr_backend()` routes `transformers` to the Whisper adapter, which is too broad  

### `api.py`

`api.py` already reads:

```python
ASR_BACKEND = os.getenv("ASR_BACKEND", "nemo").strip().lower()
MODEL_KEY = os.getenv("ASR_MODEL", DEFAULT_MODEL_KEY).strip()
MODEL_NAME = resolve_asr_model(ASR_BACKEND, MODEL_KEY)
```

and passes those values into `ASRConfig`. 

So most of the API wiring is already present.

### Dependencies

`requirements.txt` already has `transformers`, `accelerate`, and `safetensors`, so the generic Transformers backend likely does not require a new core dependency. 

---

# Implementation plan

## 1. Extend `ASRConfig`

Add generic Transformers options:

```python
@dataclass
class ASRConfig:
    backend: str
    model_key: str
    model_name: str
    hf_token: str | None = None
    language: str = "en"
    task: str = "transcribe"
    trust_remote_code: bool = False
    return_timestamps: bool = False
    chunk_length_s: int | None = None
    stride_length_s: int | tuple[int, int] | None = None
```

Purpose:

* `trust_remote_code`: for HF models that require custom code
* `return_timestamps`: generic ASR timestamp toggle
* `chunk_length_s` / `stride_length_s`: optional generic pipeline chunking, not forced by default

---

## 2. Add model aliases

In `asr_backend.py`:

```python
TRANSFORMERS_ASR_MODEL_ALIASES = {
    "cohere-transcribe-03-2026": "CohereLabs/cohere-transcribe-03-2026",
}
```

Keep alias sets separate:

```python
NEMO_MODEL_ALIASES = {...}
WHISPER_MODEL_ALIASES = {...}
TRANSFORMERS_ASR_MODEL_ALIASES = {...}
```

Do **not** put Cohere in `WHISPER_MODEL_ALIASES`.

---

## 3. Fix backend naming

Right now, `ASR_BACKEND=transformers` is treated as Whisper. That is risky. Change this:

```python
if normalized_backend in {"whisper", "transformers", "transformers-whisper"}:
```

to:

```python
if normalized_backend in {"whisper", "transformers-whisper"}:
```

Then add:

```python
if normalized_backend in {"transformers-asr", "hf-asr"}:
    if not normalized_key:
        return TRANSFORMERS_ASR_MODEL_ALIASES["cohere-transcribe-03-2026"]
    return TRANSFORMERS_ASR_MODEL_ALIASES.get(normalized_key, normalized_key)
```

This prevents generic HF models from accidentally using Whisper-only generation parameters.

---

## 4. Add `TransformersASRBackend`

Create a new class beside `WhisperASRBackend`:

```python
class TransformersASRBackend:
    """Generic Hugging Face Transformers ASR adapter."""

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        trust_remote_code: bool = False,
        return_timestamps: bool = False,
        chunk_length_s: int | None = None,
        stride_length_s: int | tuple[int, int] | None = None,
    ) -> None:
        from transformers import pipeline

        self.model_name = model_name
        self.return_timestamps = return_timestamps
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipe_kwargs = {
            "task": "automatic-speech-recognition",
            "model": model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if hf_token:
            pipe_kwargs["token"] = hf_token

        self.pipe = pipeline(**pipe_kwargs)
```

### Why separate class

The Whisper class currently forces:

```python
generate_kwargs["return_timestamps"] = True
generate_kwargs["return_token_timestamps"] = True
```

and adds Whisper-specific `chunk_length_s=30` / `stride_length_s=(0, 2)` when timestamps are requested. 

A generic model may reject those args.

---

## 5. Implement generic `.transcribe()`

It must expose the same shape expected by existing chunk code:

```python
outputs = asr_model.transcribe(..., return_hypotheses=True)
```

Existing code downstream expects objects with:

```python
hyp.text
hyp.timestamp
```

So generic output should return `SimpleHypothesis`.

Implementation:

```python
def transcribe(
    self,
    inputs: Iterable[Any],
    timestamps: bool = False,
    verbose: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    return_hypotheses: bool = True,
    **_: Any,
) -> list[SimpleHypothesis] | list[str]:
    prepared = [self._prepare_input(item) for item in inputs]

    pipe_call_kwargs = {
        "batch_size": batch_size,
    }

    should_return_timestamps = timestamps and self.return_timestamps
    if should_return_timestamps:
        pipe_call_kwargs["return_timestamps"] = True

    if self.chunk_length_s:
        pipe_call_kwargs["chunk_length_s"] = self.chunk_length_s
    if self.stride_length_s:
        pipe_call_kwargs["stride_length_s"] = self.stride_length_s

    outputs = self.pipe(prepared, **pipe_call_kwargs)
    if isinstance(outputs, dict):
        outputs = [outputs]

    hypotheses = [
        self._to_hypothesis(output, timestamps=should_return_timestamps)
        for output in outputs
    ]

    if return_hypotheses:
        return hypotheses
    return [hyp.text for hyp in hypotheses]
```

---

## 6. Implement generic timestamp parsing

Transformers ASR pipelines commonly return:

```python
{
    "text": "...",
    "chunks": [
        {"text": "...", "timestamp": (start, end)}
    ]
}
```

But not every model supports it. So parsing must be tolerant.

```python
def _to_hypothesis(self, output: dict[str, Any], timestamps: bool) -> SimpleHypothesis:
    text = str(output.get("text") or "").strip()

    if not timestamps:
        return SimpleHypothesis(text=text, timestamp={"word": [], "segment": []})

    words = []
    segments = []

    for chunk in output.get("chunks") or []:
        chunk_text = str(chunk.get("text") or "").strip()
        ts = chunk.get("timestamp") or (None, None)

        if not chunk_text:
            continue
        if not isinstance(ts, (list, tuple)) or len(ts) < 2:
            continue

        start, end = ts[0], ts[1]
        if start is None or end is None:
            continue

        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue

        if end_f <= start_f:
            continue

        words.append({"word": chunk_text, "start": start_f, "end": end_f})
        segments.append({"segment": chunk_text, "start": start_f, "end": end_f})

    return SimpleHypothesis(text=text, timestamp={"word": words, "segment": segments})
```

For Cohere, start with `ASR_RETURN_TIMESTAMPS=0`; add timestamp support only after confirming the pipeline emits `chunks`.

---

## 7. Wire into `load_asr_backend()`

Add:

```python
if backend in {"transformers-asr", "hf-asr"}:
    return TransformersASRBackend(
        model_name=config.model_name,
        hf_token=config.hf_token,
        trust_remote_code=config.trust_remote_code,
        return_timestamps=config.return_timestamps,
        chunk_length_s=config.chunk_length_s,
        stride_length_s=config.stride_length_s,
    )
```

Also update error text:

```python
raise ValueError(
    "ASR_BACKEND must be 'nemo', 'whisper', or 'transformers-asr' "
    f"(got {config.backend!r})"
)
```

---

## 8. Update `api.py` environment parsing

Add:

```python
ASR_TRUST_REMOTE_CODE = _get_env_bool("ASR_TRUST_REMOTE_CODE", "0")
ASR_RETURN_TIMESTAMPS = _get_env_bool("ASR_RETURN_TIMESTAMPS", "0")
ASR_CHUNK_LENGTH_S = os.getenv("ASR_CHUNK_LENGTH_S")
ASR_STRIDE_LENGTH_S = os.getenv("ASR_STRIDE_LENGTH_S")
```

Parse optional ints:

```python
def _get_optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)
```

Then:

```python
ASR_CHUNK_LENGTH_S = _get_optional_env_int("ASR_CHUNK_LENGTH_S")
ASR_STRIDE_LENGTH_S = _get_optional_env_int("ASR_STRIDE_LENGTH_S")
```

Pass into config:

```python
config = ASRConfig(
    backend=ASR_BACKEND,
    model_key=MODEL_KEY,
    model_name=MODEL_NAME,
    hf_token=HF_TOKEN,
    language=ASR_LANGUAGE,
    task=ASR_TASK,
    trust_remote_code=ASR_TRUST_REMOTE_CODE,
    return_timestamps=ASR_RETURN_TIMESTAMPS,
    chunk_length_s=ASR_CHUNK_LENGTH_S,
    stride_length_s=ASR_STRIDE_LENGTH_S,
)
```

---

## 9. Fix default model selection

Current `api.py` defaults to Medical Whisper for anything other than NeMo:

```python
DEFAULT_MODEL_KEY = "parakeet-0.6b" if ASR_BACKEND == "nemo" else "medical-whisper-large-v3"
```

That should become:

```python
if ASR_BACKEND == "nemo":
    DEFAULT_MODEL_KEY = "parakeet-0.6b"
elif ASR_BACKEND in {"whisper", "transformers-whisper"}:
    DEFAULT_MODEL_KEY = "medical-whisper-large-v3"
elif ASR_BACKEND in {"transformers-asr", "hf-asr"}:
    DEFAULT_MODEL_KEY = "cohere-transcribe-03-2026"
else:
    DEFAULT_MODEL_KEY = "parakeet-0.6b"
```

---

## 10. Update `/health`

Add:

```python
"asr_trust_remote_code": ASR_TRUST_REMOTE_CODE,
"asr_return_timestamps": ASR_RETURN_TIMESTAMPS,
"asr_chunk_length_s": ASR_CHUNK_LENGTH_S,
"asr_stride_length_s": ASR_STRIDE_LENGTH_S,
```

This makes it obvious which path is running.

---

## 11. Handle `timestamps=word` for generic ASR

Current API defaults:

```python
timestamps: Literal["word", "segment", "none"] = Form("word", ...)
```

For `transformers-asr`, word timestamps may not be supported. So add a guard:

```python
effective_timestamps = timestamps

if ASR_BACKEND in {"transformers-asr", "hf-asr"} and timestamps != "none":
    if not ASR_RETURN_TIMESTAMPS:
        effective_timestamps = "none"
```

Then use `effective_timestamps` in transcription and merge logic:

```python
result = transcribe_chunks_in_memory_mode(
    ...
    timestamps=effective_timestamps,
)
```

and:

```python
words = result["words"] if effective_timestamps == "word" else []
```

This prevents false diarization support.

Also reject diarization if generic backend timestamps are disabled:

```python
if diarization and ASR_BACKEND in {"transformers-asr", "hf-asr"} and not ASR_RETURN_TIMESTAMPS:
    raise HTTPException(
        status_code=400,
        detail="diarization requires ASR_RETURN_TIMESTAMPS=1 for transformers-asr",
    )
```

---

## 12. Update `.env.example`

Add:

```env
# Generic Hugging Face Transformers ASR, e.g. Cohere
# ASR_BACKEND=transformers-asr
# ASR_MODEL=cohere-transcribe-03-2026
# ASR_MODEL=CohereLabs/cohere-transcribe-03-2026
# ASR_RETURN_TIMESTAMPS=0
# ASR_TRUST_REMOTE_CODE=0
# ASR_CHUNK_LENGTH_S=
# ASR_STRIDE_LENGTH_S=
```

Recommended Cohere config:

```env
ASR_BACKEND=transformers-asr
ASR_MODEL=CohereLabs/cohere-transcribe-03-2026
ASR_RETURN_TIMESTAMPS=0
ASR_TRUST_REMOTE_CODE=0
HF_TOKEN=
```

If model loading says remote code is required:

```env
ASR_TRUST_REMOTE_CODE=1
```

Only enable that when necessary.

---

## 13. Optional: make `dotenv` optional

Since your old container crashed on missing `dotenv`, make this safe:

```python
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False
```

Then the app can still start in old containers that bind-mount updated source but do not have the rebuilt dependency set.

---

## 14. Testing sequence

### A. Parakeet still works

```env
ASR_BACKEND=nemo
ASR_MODEL=parakeet-1.1b
```

Expected `/health`:

```json
{
  "asr_backend": "nemo",
  "asr_model_name": "nvidia/parakeet-tdt-1.1b"
}
```

### B. Medical Whisper still works

```env
ASR_BACKEND=whisper
ASR_MODEL=medical-whisper-large-v3
ASR_LANGUAGE=en
ASR_TASK=transcribe
```

Expected:

```json
{
  "asr_backend": "whisper",
  "asr_model_name": "Na0s/Medical-Whisper-Large-v3"
}
```

### C. Cohere text-only works first

```env
ASR_BACKEND=transformers-asr
ASR_MODEL=CohereLabs/cohere-transcribe-03-2026
ASR_RETURN_TIMESTAMPS=0
```

Request with:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F timestamps=none
```

Expected:

* `text` populated
* `segments` empty
* no diarization

### D. Cohere timestamp experiment

Only after text-only works:

```env
ASR_RETURN_TIMESTAMPS=1
ASR_CHUNK_LENGTH_S=30
```

Then test:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F timestamps=segment
```

If the model does not emit `chunks`, keep it text-only.

---

# Final expected env modes

## Parakeet

```env
ASR_BACKEND=nemo
ASR_MODEL=parakeet-1.1b
```

## Medical Whisper

```env
ASR_BACKEND=whisper
ASR_MODEL=medical-whisper-large-v3
ASR_LANGUAGE=en
ASR_TASK=transcribe
```

## Cohere

```env
ASR_BACKEND=transformers-asr
ASR_MODEL=CohereLabs/cohere-transcribe-03-2026
ASR_RETURN_TIMESTAMPS=0
ASR_TRUST_REMOTE_CODE=0
```

That is the clean design: **three backends, no cross-contamination of Whisper-specific settings into generic Transformers ASR.**
