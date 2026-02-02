# Silero VAD (chunking long WAVs)

This step uses Silero VAD to split long audio into speech-only chunks before ASR.

## Usage (container)

```bash
docker compose run --rm api python3.11 vad_chunk.py /app/audio.wav
```

Chunks are written to `Output/vad_chunks` by default. Filenames include segment start/end times.

## Notes

- The script downloads the Silero VAD model via `torch.hub` on first run.
- Defaults are tuned for conversational audio; adjust `--threshold`, `--min-speech-ms`, `--min-silence-ms`, `--merge-gap-ms`, `--target-min-s`, `--target-max-s`, `--hard-max-s`, `--overlap-s`, and `--speech-pad-ms` as needed.
- For continuous speech/music with few silences, chunks are hard-cut at `--hard-max-s` with `--overlap-s` overlap (default ~45s windows, 0.5s overlap).
- Default `--speech-pad-ms` is 200ms to reduce word clipping at chunk boundaries.

## Chunked ASR merge

Transcribe VAD chunks and merge word timestamps into the full-file timeline:

```bash
docker compose run --rm api python3.11 chunk_transcribe.py \
  --chunk-dir Output/vad_chunks \
  --out-json Output/merged_words.json \
  --out-text Output/merged_text.txt
```

The merged word list is timestamped relative to the original audio.

Note: model downloads are cached under `Output/.cache` (mounted into the container) so reruns do not re-download the Parakeet model.

## Implementation 

Goal

Feed Silero VAD speech chunks into Parakeet, then reconstruct a single global transcript where all timestamps are relative to the original full audio.

What to implement
1) Define the chunk metadata (must exist for every chunk)

For each Silero VAD chunk, you need:

chunk_id

orig_start_s / orig_end_s (seconds in the original file)

pad_left_s / pad_right_s (how much padding was added)

audio_path (or in-memory waveform)

effective_start_s = orig_start_s - pad_left_s (clamped to 0)

effective_end_s = orig_end_s + pad_right_s

This is the key: Parakeet timestamps are relative to the chunk audio you feed it, starting at 0.0.
So you must add effective_start_s back to every timestamp in the output.

2) Audio format contract (required)

Before Parakeet:

mono

16 kHz

float32 waveform in [-1, 1] (or standard WAV that loads to that)

Do this once and then extract chunks from the normalized waveform.

3) Chunking rules (avoid boundary errors)

Use these defaults:

pad_left_s = pad_right_s = 0.2 seconds

merge gaps < 0.35s into the same chunk (speech separated by tiny pauses shouldn’t split)

if a chunk becomes too long, split only at silence boundaries

Reason: padding prevents cutting words; merging prevents excessive micro-chunks.

How to line up timestamps
4) Offset Parakeet timestamps into original timeline

When you transcribe a chunk, Parakeet returns word timestamps like:

word.start and word.end in seconds relative to chunk start

Convert to global times by:

word.start_global = word.start_local + effective_start_s

word.end_global = word.end_local + effective_start_s

Do the same for:

segment timestamps (timestamp["segment"])

char timestamps if you use them

5) Handle overlap / duplicates caused by padding

Because you padded, adjacent chunks may overlap in time.

You must de-duplicate words in overlaps. Use a simple deterministic policy:

Policy

Maintain last_emitted_time = end of last emitted word

For each new word (in time order), if:

word.start_global < last_emitted_time - 0.02 (20 ms tolerance)

AND the word text matches a recently emitted word (optional but recommended)
→ drop it

Otherwise emit it and update last_emitted_time.

If you want even simpler:

drop any word whose end_global <= last_emitted_time

This is usually sufficient with small padding.

6) Output structure

Build one unified list:

words[] sorted by start_global

optionally convert words to segments[] by speaker later

For now, ensure the final words:

are monotonically increasing in time

do not contain duplicates at boundaries

still cover all speech

Implementation checklist for the dev
A) Data flow

Load full audio → normalize mono/16kHz

Run Silero VAD → get speech intervals (samples or seconds)

Merge close intervals + pad them

Extract chunk waveforms + store metadata (effective_start_s)

For each chunk:

run Parakeet transcribe(..., timestamps=True)

offset timestamps by effective_start_s

Merge all chunk word lists in time order

Deduplicate in overlaps

Produce final transcript text by joining words (or use Parakeet text as a hint, but words are canonical)

B) Tests to run

silence-only file → no chunks → no ASR calls

one short phrase with leading/trailing silence → timestamps match real position

two phrases separated by 1s silence → global timestamps jump correctly

forced overlap (padding) → no duplicated boundary words

C) Common pitfalls

forgetting padding offsets and using orig_start_s instead of effective_start_s

extracting chunks from unnormalized audio (48kHz stereo) → timestamp drift and worse ASR

splitting too aggressively → Parakeet capitalizes/punctuates strangely at each chunk

not handling overlaps → repeated words in final transcript


## With pyannote

How to combine ASR (chunked) with pyannote (full-file)
1) Canonical timeline

Everything must live on the same timebase:

seconds

relative to start of the original audio file

from the same normalized waveform (mono, 16 kHz)

You already do this for ASR by offsetting chunk timestamps with effective_start_s.

pyannote output is already in that timebase if you diarize the full file.

2) Run diarization once

pyannote gives you speaker turns:

(start_s, end_s, speaker_label)

Convert to a sorted list:

turns = [{"start": s, "end": e, "speaker": spk}, ...]
turns.sort(key=lambda t: (t["start"], t["end"]))

3) Assign speaker to each ASR word (or ASR segment)

For each word with [ws, we], pick the speaker turn that overlaps most.

Overlap between word and turn:
overlap = max(0, min(we, turn.end) - max(ws, turn.start))

Assignment policy (simple and robust):

choose the turn with max overlap

if max overlap < small threshold (e.g. 0.03s), label as UNKNOWN

This is O(N*M) if naïve; implement it with two pointers since both lists are sorted.

Two-pointer outline:

iterate words in time order

advance diarization turn index while turn.end < word.start

check current turn and next few turns that might overlap

4) Build final “diarized transcript segments”

Once each word has a speaker, create segments by grouping contiguous words where:

speaker stays the same

gap between words is small (e.g. < 0.6s)

Segment fields:

start = first word start

end = last word end

speaker

text = joined words

This yields your “usual JSON schema” style segments.

Where VAD chunks fit into this

VAD chunking is used only for:

deciding what audio to send to Parakeet

saving GPU time on ASR

It does not need to be used for diarization if you choose Option A.

You can optionally still use VAD to skip diarization when:

no speech detected at all

but this is not a big win unless you have many empty recordings

Practical caveats you should tell your dev
Overlap speech

pyannote can produce overlapping turns in difficult audio.
If you use “exclusive” diarization (non-overlapping), assignment is simpler.
If not, the max-overlap policy still works.

Boundary padding duplicates

Your ASR merge step must deduplicate overlap words before diarization alignment, otherwise you’ll assign speakers twice and get repeated text.

Speaker label stability

pyannote speaker labels are only stable within the file:
SPEAKER_00 does not mean the same person across different uploads.

Minimal implementation plan (what to do next)

Keep your existing VAD → chunk ASR pipeline and ensure you output:

global word list with start, end, word

Add pyannote diarization on the full file and output:

sorted turn list (start, end, speaker)

Implement assign_speaker(words, turns) using overlap-max and two pointers.

Implement group_words_into_segments(words_with_speaker).
