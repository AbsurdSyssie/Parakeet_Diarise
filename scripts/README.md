# Scripts

Utilities that are useful from the command line but are not part of the API runtime.

## Utilities

- `run_api.py`: launch the FastAPI service; defaults to `API_PORT` from `.env`, with `--port` as a CLI override
- `merge_diarized.py`: merge word timestamps with diarization turns from JSON files

## Manual probes

Diagnostic and benchmarking helpers live in `probes/`. See `probes/README.md`.

Run scripts from the repository root so their documented relative paths resolve consistently.
