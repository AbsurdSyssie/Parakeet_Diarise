# Manual probes

These scripts are diagnostic tools, not automated tests. Run them from the repository root.

- `asr_chunk.py`: inspect VAD chunk boundaries through the API
- `asr_debug_levers.py`: sweep API/VAD settings and compare artifacts
- `diarization_dry.py`: run the shared diarization runtime on a local file
- `diarization_align.py`: run ASR, diarization, and speaker alignment together
- `parakeet_clip.py`: inspect Parakeet timestamps on a short clip

Scratch output defaults to `tmp/probes/`, which is git-ignored.

API-facing probes default to `http://localhost:$API_PORT`, loading `API_PORT` from the repository `.env`. Their explicit `--endpoint` or `--url` options still override that default where available.
