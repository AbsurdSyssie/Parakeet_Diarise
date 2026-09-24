# Historical API benchmarks

This file preserves benchmark and tuning notes that were previously embedded in the API reference. Values reflect the environment and code state at the time they were recorded; they are not current performance guarantees.

## Recent benchmarks

Memory mode, `path/to/audio.wav`, diarization=false:
- `timestamps=word`: 8.86s
- `timestamps=segment`: 8.73s (~1.5% faster vs word)
- `timestamps=none`: 7.62s (~14% faster vs word)

Memory mode, `path/to/audio.wav`, diarization=true:
- 16 kHz VAD, `VAD_DEVICE=cpu`: decode ~1.00s; VAD 5.95s; ASR 3.60s; diarization 10.04s; total 20.52s
- 8 kHz VAD, `VAD_DEVICE=cuda`: decode 0.89s; VAD 7.40s; ASR 2.76s; diarization 9.88s; total 21.07s

Energy gate test (1h audio, `VAD_ENERGY_GATE=1`, `VAD_DEVICE=cpu`, `VAD_SAMPLE_RATE=8000`):
- energy gate found 2 intervals; decode 0.28s; VAD 25.60s; ASR 18.86s; diarization 49.51s; total 95.09s
- Current energy gate settings in `.env`: `VAD_ENERGY_DB=-35`, `VAD_ENERGY_FRAME_MS=100`, `VAD_ENERGY_MIN_ACTIVE_MS=500`, `VAD_ENERGY_MERGE_GAP_MS=800`.
- Proposed: run the cheap energy gate to estimate speech ratio; if active > ~70% of duration, skip Silero and chunk uniformly (30–45s with overlap) before ASR.
Energy gate skip test (1h audio, ratio=1.00, `VAD_ENERGY_ACTIVE_SKIP=0.85`):
- uniform intervals: 80; decode 0.30s; VAD 0.48s; ASR 15.61s; diarization 50.72s; total 67.86s
- Takeaway: energy-gate skip removes most VAD cost, but diarization dominates total time.

TF32 diarization comparison (same workload):
- `DIARIZE_TF32=1`: diarization 54.40s / 53.11s; total 70.44s / 69.09s
- `DIARIZE_TF32=0`: diarization 51.96s / 53.33s; total 68.26s / 69.26s
- Conclusion: no consistent speedup from TF32 in this run.

