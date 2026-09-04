"""Minimal .env loader — no third-party dependency, no secret ever logged.

Why this exists: the safest place for an API key is a file you create in a text
editor and never type into a shell. Typing `export`/`$env:` into a terminal
writes the key into that shell's history file in plaintext, where it stays.

Precedence is deliberate: a real environment variable ALWAYS wins over `.env`.
That way a CI runner, a Docker `env_file`, or a lab machine's system environment
cannot be silently overridden by a stale file someone left in a checkout.

`.env` is gitignored. `.env.example` is committed as the template and must never
contain a real value.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set once the first load has run, so importing several config modules does not
#: re-read the file.
_loaded = False

#: Keys we are willing to take from a .env file. An allowlist rather than
#: "anything in the file" keeps a stray line from redefining PATH or similar.
ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "OPENAI_API_KEY",
        "PLATO_MODEL_CHEAP",
        "PLATO_MODEL_STRONG",
        "PLATO_LLM_BASE_URL",
        "PLATO_HARDWARE",
        "PLATO_FAKE_SUCCESS_RATE",
        "PLATO_SAVE_DIR",
    }
)


def project_root() -> Path:
    """The repository root (this file lives in <root>/config/)."""
    return Path(__file__).resolve().parents[1]


def parse_env_text(text: str) -> dict[str, str]:
    """Parse .env content into a mapping. Pure function, easy to test.

    Supports ``KEY=value``, ``export KEY=value``, ``#`` comments, blank lines,
    and single- or double-quoted values. Deliberately does not support
    interpolation or multi-line values -- an API key needs none of it, and every
    extra feature is another way to get the parse subtly wrong.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(path: str | Path | None = None, *, override: bool = False) -> list[str]:
    """Load `.env` into ``os.environ``. Returns the NAMES that were set.

    Never returns or logs a value. Real environment variables win unless
    ``override=True``, which exists only for tests.
    """
    global _loaded
    env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.is_file():
        _loaded = True
        return []

    # utf-8-sig, not utf-8: Notepad and several Windows editors write a UTF-8
    # BOM, which would otherwise become part of the first key's NAME
    # ("﻿OPENAI_API_KEY") and the key would be silently dropped.
    applied: list[str] = []
    for key, value in parse_env_text(env_path.read_text(encoding="utf-8-sig")).items():
        if key not in ALLOWED_KEYS:
            continue
        if not override and os.environ.get(key):
            continue  # a real environment variable always wins
        os.environ[key] = value
        applied.append(key)

    _loaded = True
    return applied


def ensure_loaded() -> None:
    """Load `.env` once per process. Called by :mod:`config.models` on import."""
    if not _loaded:
        load_env()
