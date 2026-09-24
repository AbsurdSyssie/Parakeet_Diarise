#!/usr/bin/env python3
"""Run the FastAPI service using .env defaults with CLI overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from settings import SETTINGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the transcription API.")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SETTINGS.api_port,
        help="Bind port (default: API_PORT from .env, or 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the server when source files change.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
