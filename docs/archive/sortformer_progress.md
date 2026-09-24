# Sortformer Progress

## Status

- Phase: local experimentation (repo `.venv`)
- Target model: `nvidia/diar_streaming_sortformer_4spk-v2.1`
- Last updated: 2026-02-02

## What we set up

- Installed NeMo + deps into local `.venv`:
  - `torch`, `torchaudio`, `Cython`, `packaging`, `nemo_toolkit[asr]` (via pip)
- Added a local dry-run script for Sortformer diarization:
  - `test/sortformer_dry.py`
- Added a local alignment script to combine:
  - Parakeet ASR (via running API on port 8000)
  - Sortformer diarization (local)
  - Alignment using `diarize_align.py`
  - Script: `test/sortformer_align_dry.py`
  - Uses `/dev/shm` temp WAV when available (NeMo accepts file paths only)
- Moved Sortformer diarization into the ASR API container (single-container workflow).

## Successful runs

### Sortformer dry run (WAV)

Command:
```
.venv/bin/python test/sortformer_dry.py --audio path/to/audio.wav
```

Result:
- Model downloaded from HF on first run
- Output written to `test/tmp/sortformer_segments.json`
- 55 diarization turns produced

### ASR + Sortformer alignment (WAV)

Command:
```
.venv/bin/python test/sortformer_align_dry.py --audio path/to/audio.wav
```

Result:
- ASR words + Sortformer turns aligned successfully
- Output written to:
  - `test/tmp/sortformer_align/asr_words.json`
  - `test/tmp/sortformer_align/sortformer_turns.json`
  - `test/tmp/sortformer_align/aligned_words.json`
  - `test/tmp/sortformer_align/aligned_segments.json`
- UNKNOWN speakers eliminated after fixing turn ordering (see below)

### ASR + Sortformer alignment (MP3)

Command:
```
.venv/bin/python test/sortformer_align_dry.py --audio "path/to/audio.mp3"
```

Result:
- MP3 downmixed to mono and resampled to 16 kHz in-memory
- Temporary WAV written to `/dev/shm` (or `test/tmp` fallback)
- ASR + Sortformer alignment completed successfully

## Issues encountered

### 1) `diarize()` rejected `sample_rate` for numpy input

Error:
```
TypeError: SortformerEncLabelModel.diarize() got an unexpected keyword argument 'sample_rate'
```

Fix:
- Pass audio file path(s) instead of numpy arrays.
- Remove `sample_rate` argument.

### 2) `torchcodec` import error when loading audio

Error:
```
ImportError: TorchCodec is required for load_with_torchcodec.
```

Fix:
- Added `soundfile` fallback in `test/sortformer_dry.py` for audio loading.

### 3) Permission error writing to `Output/`

Error:
```
PermissionError: [Errno 13] Permission denied: 'Output/sortformer_segments.json'
```

Fix:
- Added fallback output path to `test/tmp/sortformer_segments.json`.

### 4) UNKNOWN speakers in aligned segments

Symptom:
- Many `UNKNOWN` speakers in `aligned_words.json` and `aligned_segments.json`.

Root cause:
- Sortformer turns were not time-sorted; a late turn (`start=90.88`) appeared after a much later turn (`start=534.96`).
- The two-pointer alignment skipped valid turns.

Fix:
- Sort turns by `(start, end)` in `run_sortformer()`.
- Result: `UNKNOWN` speakers dropped to zero for WAV run.

### 5) MP3 diarization failed due to stereo shape

Command:
```
.venv/bin/python test/sortformer_align_dry.py --audio "path/to/audio.mp3"
```

Error:
```
TypeError: Input shape mismatch occured for input_signal in module AudioToMelSpectrogramPreprocessor :
Input shape expected = (batch, time)
Input shape found : torch.Size([1, 11911680, 2])
```

Cause:
- Sortformer pipeline expects mono `(batch, time)`.
- MP3 is stereo (2 channels), so NeMo’s internal preprocessor receives `(batch, time, channels)`.

Fix:
- Added mono downmix + 16 kHz resample before diarization.
- NeMo `diarize()` in this version accepts file paths only, so we write a temporary WAV (RAM-backed in `/dev/shm` when possible) and delete it after.

## Current scripts

- `test/sortformer_dry.py`
  - Runs Sortformer on a local file and writes turns.
- `test/sortformer_align_dry.py`
  - Calls ASR API on port 8000 (words)
  - Runs Sortformer locally (turns)
  - Aligns speakers with `diarize_align.py`
  - Writes aligned words and segments
  - Downmixes/resamples to mono 16 kHz and uses a temp WAV for Sortformer input

## In-container workflow (current)

- `api.py` loads Parakeet once at startup and loads Sortformer lazily on first diarization request.
- `diarization=true` now runs Sortformer in-process (no external diarization service).
- Sortformer uses a mono 16 kHz WAV written to a temp directory for inference (NeMo `diarize()` requires a file path).

## How the local flow works (today)

1) ASR (Parakeet API on port 8000) produces `words[]`.
2) Sortformer runs locally on a mono 16 kHz audio stream and outputs diarization turns.
3) Turns are normalized and sorted by `(start, end)`.
4) `assign_speakers()` labels each word by max overlap, with a small `max_gap_s` fallback.
5) `group_words_into_segments()` merges contiguous words into diarized segments.

## Example output (aligned segments)

```json
{
  "speaker": "SPEAKER_00",
  "start": 96.2,
  "end": 100.28,
  "text": "Alright, and you and you said, sorry, has he this is he left the house? He's left the house"
}
```

## Next steps

- Decide whether to keep the `/dev/shm` temp WAV approach or move to a NeMo version that accepts numpy + sample_rate.
- Optionally add CLI flags:
  - `--asr-url` (override API URL)
  - `--max-gap-s` (alignment tolerance)
  - `--force-mono` (downmix in script)
