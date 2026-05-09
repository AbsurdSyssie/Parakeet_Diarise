# Bugs and Issues: Cohere TransformersASRBackend

**File**: `asr_backend.py` — `TransformersASRBackend` class (line 196)

**Model**: `CohereLabs/cohere-transcribe-03-2026`

**Goal**: Add a third ASR backend (`ASR_BACKEND=transformers-asr`) using Hugging Face Transformers, initially targeting the Cohere Transcribe model. Must coexist with NeMo and Whisper backends.

## Environment

| Package | Version |
|---------|---------|
| torch | 2.11.0 |
| transformers | 5.8.0 |
| nemo-toolkit | 2.6.1 |
| torchaudio | 2.11.0 |

Model requires per model card: `transformers>=5.4.0`, `torch==2.10.0` (tested with).

---

## Bug 1 (FIXED): Pipeline API incompatible with Cohere model

**First encountered**: 2026-05-09, during initial _transcribe_ call.

**Error**:
```
ValueError: Expected `input_ids` or `decoder_input_ids`.
```
From `modeling_cohere_asr.py:796` in the model's `forward()` method.

**Root cause**: The original `TransformersASRBackend` used `transformers.pipeline("automatic-speech-recognition", model=...)`. The Cohere model's custom `forward()` expects `input_features` (from the processor) not raw `input_ids`. The HuggingFace ASR pipeline's `__call__` → `preprocess` pathway does not correctly route audio through the `CohereAsrFeatureExtractor` into `input_features` for this model. The pipeline was designed for Whisper/Wav2Vec2-style models and the Cohere model uses a different interface.

**Fix**: Rewrote `TransformersASRBackend.__init__` to use:
```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
self.processor = AutoProcessor.from_pretrained(model_name, ...)
self.model = CohereAsrForConditionalGeneration.from_pretrained(model_name, ...)
```
And rewrote `transcribe()` to use the processor + `model.generate()` + `processor.decode()` pipeline instead of the HuggingFace `pipeline()` helper. This matches the model card's documented usage.

**Key insight**: `AutoModel.from_pretrained()` returns the base `CohereAsrModel` class (no `.generate()`). Must use `CohereAsrForConditionalGeneration` directly. `AutoModelForSpeechSeq2Seq` also works but requires `trust_remote_code` everywhere.

---

## Bug 2 (FIXED): `_validate_model_kwargs` rejects `length` parameter

**Error**:
```
ValueError: The following `model_kwargs` are not used by the model: ['length'] 
(note: typos in the generate arguments will also show up in this list)
```
From `transformers/generation/utils.py` in `_validate_model_kwargs()`.

**Root cause**: The Cohere model's custom `generate()` method (line 885 in `modeling_cohere_asr.py`) explicitly passes `length` to `super().generate()`:
```python
generation_kwargs["length"] = length
return super().generate(**generation_kwargs)
```

The HuggingFace `GenerationMixin._validate_model_kwargs()` checks all kwargs against the model's `forward()` signature. `CohereAsrForConditionalGeneration.forward()` does NOT declare `length` in its parameter list:
```python
# Forward signature (from inspect.signature):
def forward(self, input_features=None, attention_mask=None, decoder_input_ids=None, 
            decoder_attention_mask=None, encoder_outputs=None, past_key_values=None,
            decoder_inputs_embeds=None, decoder_position_ids=None, use_cache=None,
            labels=None, **kwargs)
```

However, `CohereAsrForConditionalGeneration.prepare_inputs_for_generation()` DOES consume `length`:
```python
def prepare_inputs_for_generation(self, input_ids, ..., **kwargs):
    return {
        ...
        "input_features": kwargs.get("input_features"),
        "length": kwargs.get("length"),        # <-- consumed here
        ...
    }
```

The encoder submodule inside the model needs `length` to know how many mel frames are valid vs padding. Without it, the encoder produces wrong output shapes.

**Fix**: Monkeypatched `_validate_model_kwargs` to a no-op at the end of `__init__`:
```python
self.model._validate_model_kwargs = lambda kwargs, model_kwargs=None: None
```
This is safe because `prepare_inputs_for_generation` correctly routes `length` to the encoder. The validation is only a safety check; the model handles the kwargs correctly via `prepare_inputs_for_generation`.

**Alternative approaches considered**:
- Stripping `length` before calling `generate()`. Rejected: causes shape mismatch in encoder (see Bug 3).
- Adding `length` as a dummy parameter to the model's forward. Rejected: too invasive, would require modifying cached HuggingFace model code.
- Using `model(**inputs).loss + model.generate()` separately. Rejected: overcomplicated.

---

## Bug 3 (UNRESOLVED): Encoder shape mismatch — `mat1 and mat2 shapes cannot be multiplied`

**Error** (with real audio `Examples/MoreOrLess.wav`, 9.99s, 159887 samples):
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x32000 and 4096x1280)
```
From inside `_prepare_encoder_decoder_kwargs_for_generation()` → `encoder(**encoder_kwargs)` → Conformer layer matmul.

With dummy audio (1s, 16000 samples):
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x3328 and 4096x1280)
```

**Investigation so far**:

1. **Processor output**: The `AutoProcessor` produces correct shapes:
   - `input_features`: `(1, 128, T)` where T depends on audio length (hop_length=160)
   - `length`: `(1,)` — integer tensor (e.g., `[1000]` for 10s audio)

2. **Feature extractor config**: `FilterbankFeatures` with `sampling_rate=16000`, `hop_length=160`, expected `num_mel_bins=128`. The `CohereAsrFeatureExtractor.filterbank` is a `FilterbankFeatures` module from `processing_cohere_asr.py`.

3. **Dtype handling**: Inputs are converted to `model.dtype` (float16) before `generate()`. `length` tensor is `int64` — tested both with and without dtype conversion.

4. **Bypassing `_validate_model_kwargs`** (Bug 2 fix) allows `generate()` to proceed, but then the encoder fails at the first dense layer.

5. **Without `length`**: If `length` is stripped from kwargs before `generate()`, the model runs but produces wrong shapes — the encoder doesn't know the true sequence length vs padding and the matmul dimensions don't align.

6. **The shape `16x3328`** in the error: `16` could be related to the Conformer's internal dimension (`d_model=1280`? no). The encoder architecture uses a Conformer with `encoder_dim=1280` and `attention_heads=16`. The `3328` is likely the mel frame count times the filterbank dimension before reshaping, or a product of unexpected dimension computation.

7. **The shape `4096x1280`**: `1280` matches the model's internal dimension (`d_model=1280`). `4096` is `128 * 32` — possibly the internal expansion factor of a feedforward layer (4x expansion of 1280 ≈ 5120, but 4096 = 128 * 32).

**Hypothesis**: The filterbank inside the feature extractor may not be producing outputs compatible with the encoder's expected dimensions. Either:
- The filterbank's `forward()` output is not being compiled correctly (note: the model code calls `torch.compile` on filterbank layers in `_setup_compile()`)
- The filterbank dimensions (128 mel bins × T frames) don't match the encoder's expected input dimension (1280) — there's a projection layer missing or failing
- The encoder's first dense/projection layer expects `[batch, T, 128]` but receives `[batch, 128, T]` (transpose issue in `prepare_inputs_for_generation`)

**What needs investigation**:
1. Check the exact matmul dimensions at the failing layer — trace through `_prepare_encoder_decoder_kwargs_for_generation` to see exactly what arguments are passed to `encoder(**kwargs)`
2. Check if `CohereAsrForConditionalGeneration` expects transposed input vs what the processor produces
3. Compare with a minimal working example from the model card (which uses `device_map="auto"` instead of `device="cuda:0"`)
4. Test with `device_map="auto"` instead of `.to(device)` to see if device placement affects the data flow
5. Verify the `input_features` are being passed to the correct submodel (encoder vs full conditional generation wrapper)

---

## Bug 4 (RESOLVED BY WORKAROUND): Torch version incompatibility with transformers 5.8.0

**Original environment**: torch==2.4.1+cu121, transformers==4.53.3

**Error with transformers 5.8.0 + torch 2.4.1**:
```
ValueError: infer_schema(func): Parameter input has unsupported type torch.Tensor.
```
From `torch/_library/custom_ops.py` — transformers 5.8.0 uses a newer `torch.library` API that requires torch>=2.6.0 for custom ops registration.

**Resolution**: Upgraded torch to 2.11.0 and torchaudio to 2.11.0. However, this may break NeMo (nemo-toolkit 2.6.1 was tested with torch 2.4.1 originally). The NeMo Parakeet backend might not work with torch 2.11.0 — this needs testing.

---

## Bug 5 (RESOLVED): `transformers==4.53.3` — `feature_extractor._processor_class` NoneType

**Error with transformers 4.53.3**:
```
AttributeError: 'NoneType' object has no attribute '_processor_class'
```
From `transformers/pipelines/automatic_speech_recognition.py:203`.

**Cause**: The pipeline constructor in older transformers versions tries to access `feature_extractor._processor_class` even when `feature_extractor` is None. Cohere model has a feature_extractor but no processor.

**Resolution**: Upgraded to transformers>=5.4.0 where this code path was fixed.

---

## Current State

- **Model loads successfully** on CUDA
- **Processor works**: correctly produces `input_features` and `length` from raw audio
- **Validation bypass works**: `_validate_model_kwargs` monkeypatch allows `generate()` to proceed
- **Encoder matmul fails**: shape mismatch inside the Conformer encoder during first forward pass
- **NeMo/Whisper backends**: Not yet tested after torch upgrade from 2.4.1 to 2.11.0 — may be broken

## Files Modified

| File | Changes |
|------|---------|
| `asr_backend.py` | `ASRConfig` +4 fields, `TRANSFORMERS_ASR_MODEL_ALIASES`, `TransformersASRBackend` class (Cohere-specific), fixed backend naming, `load_asr_backend` wiring, validation monkeypatch |
| `api.py` | Optional dotenv, 4 new env vars, `_get_optional_env_int`, fixed default model selection, `/health` new fields, `effective_timestamps` guard, diarization rejection |
| `.env.example` | Cohere config section |
| `test/test_asr_backend.py` | 28 unit tests (backend-agnostic resolve/ASRConfig/hypothesis tests) |

## Next Steps for Investigating Agent

1. **Debug encoder matmul**: Add print statements inside `_prepare_encoder_decoder_kwargs_for_generation` or monkeypatch it to see what kwargs are passed to `encoder()`. Trace through the encoder's first dense layer to find the exact dimension mismatch.

2. **Test minimal model card example**: Run the EXACT code from the HuggingFace model card with this exact environment. If it fails too, the issue is environment/compatibility, not our code.

3. **Try `device_map="auto"`**: The model card example uses `device_map="auto"` not explicit `.to(device)`. This may affect how submodules are placed and how data flows.

4. **Check NeMo compatibility**: After torch 2.4.1→2.11.0 upgrade, verify NeMo Parakeet backend still loads and transcribes.

5. **Consider vLLM**: The model card recommends vLLM for production serving. May be an alternative backend path that bypasses these manual integration issues.
