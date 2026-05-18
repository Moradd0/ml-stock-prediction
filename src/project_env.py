"""Load ``.env`` and runtime defaults for any project entry point."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PLACEHOLDER_VALUES = frozenset(
    {"", "dummy", "your_key", "your_finnhub_key_here", "your_key_here"}
)


def load_project_env() -> None:
    """
    Apply project-root ``.env`` and safe defaults for FinBERT / threading.

    Call once at process start (``scripts/run.py``, CLI, training, etc.).
    Shell ``export FINNHUB_API_KEY=...`` still wins over ``.env`` when set to a real key.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    path = ROOT / ".env"
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        current = os.environ.get(key)
        if current is None or current in _PLACEHOLDER_VALUES:
            if value not in _PLACEHOLDER_VALUES:
                os.environ[key] = value


def finnhub_key_configured() -> bool:
    key = os.environ.get("FINNHUB_API_KEY", "")
    return bool(key) and key not in _PLACEHOLDER_VALUES


def require_finnhub_key() -> str:
    key = os.environ.get("FINNHUB_API_KEY", "")
    if key and key not in _PLACEHOLDER_VALUES:
        return key
    raise SystemExit(
        "FINNHUB_API_KEY is not set.\n"
        "  One-time setup:\n"
        "    cp .env.example .env\n"
        "    # edit .env and paste your key from https://finnhub.io/dashboard\n"
        "  Or run:  python scripts/run.py setup\n"
        "  No need to export in the shell after .env is saved."
    )
