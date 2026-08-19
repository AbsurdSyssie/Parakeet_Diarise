# IBM Granite Speech

Supported model:

```text
ibm-granite/granite-speech-4.1-2b-nar
```

The adapter uses the model repository's custom `AutoModel`, `AutoProcessor`,
and `model.transcribe()` implementation. It feeds 16 kHz mono waveforms into
the processor, decodes `output.preds`, and presents the result through the
same `SimpleHypothesis` interface as the other ASR adapters.

## Capabilities

- Batched transcription
- WAV, MP3, FLAC, and in-memory waveform input
- BF16 inference on CUDA; float32 on CPU
- SDPA by default
- Optional FlashAttention 2
- Transcript text only

The NAR checkpoint does not expose word or segment timestamps. Consequently:

- `timestamps=word` and `timestamps=segment` are reduced to `none`
- `diarization=true` is rejected
- overlapping VAD chunk text is merged using suffix/prefix token matching

## Command line

The existing command remains backward compatible:

```bash
python transcribe_parakeet.py audio.wav
```

Select Granite through the extended arguments:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python transcribe_parakeet.py Examples/output_10s.mp3 \
  --backend transformers-asr \
  --model granite-speech-4.1-2b-nar \
  --timestamps none \
  --device auto \
  --attention-implementation sdpa
```

The alias and full repository name are both accepted.

## API startup configuration

```env
ASR_BACKEND=transformers-asr
ASR_MODEL=granite-speech-4.1-2b-nar
ASR_TRUST_REMOTE_CODE=1
ASR_RETURN_TIMESTAMPS=0
ASR_ATTENTION_IMPLEMENTATION=sdpa
```

The curated registry automatically enables remote code for Granite. Explicitly
setting `ASR_TRUST_REMOTE_CODE=1` documents the security decision in deployment
configuration.

## Runtime switching

```bash
curl -X POST http://localhost:8000/v1/models/current \
  -H 'Content-Type: application/json' \
  -d '{"model":"granite-speech-4.1-2b-nar"}'
```

Model switching unloads the previous ASR and diarization models before loading
Granite.

## Dependencies and acceleration

The pinned Torch, torchaudio, and Transformers versions in `requirements.txt`
satisfy the checkpoint requirements. The default `sdpa` path does not need an
additional package.

For FlashAttention 2, install `flash-attn` in a CUDA development image and use
`--attention-implementation flash_attention_2`. The adapter reports a clear
error if it is selected without CUDA or without the package installed.

The model checkpoint is approximately 4.5 GB. Persist `HF_HOME` as the Compose
configuration already does to avoid repeated downloads.

### CUDA 13 NVRTC library path

The PyTorch 2.11 CUDA 13 wheel stores NVRTC under the virtual environment
rather than a system linker directory. Export it before starting the API:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Without this path, model warm-up can fail with:

```text
nvrtc: error: failed to open libnvrtc-builtins.so.13.0
```

## Verification

```bash
.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v

.venv/bin/python transcribe_parakeet.py Examples/output_10s.mp3 \
  --backend transformers-asr \
  --model granite-speech-4.1-2b-nar \
  --timestamps none \
  --attention-implementation sdpa
```
