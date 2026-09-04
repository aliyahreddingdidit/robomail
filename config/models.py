"""Model tier selection, env-var driven.

Upstream PLATO hardcodes model='gpt-4o' in overall_planner.py,
scene_comprehension.py and step_planner.py. That model's API shutdown date is
confirmed as 2026-10-23, so nothing here may default to it.

Two tiers, per CLAUDE.md:
  * CHEAP  -> Scene Understanding, Goal Extraction   (vision, low reasoning load)
  * STRONG -> Planner, Step Planner, Verification    (reasoning / arbitration)

Both are overridable by environment variable so no model id is baked into code.
"""

from __future__ import annotations

import os
import warnings

from config import env as _env

# Load .env once, if present. Real environment variables always win over it, so
# Docker's env_file, CI secrets and a lab machine's system environment are never
# overridden by a stale checkout. See config/env.py.
_env.ensure_loaded()

#: Models known to be retired or with an announced shutdown. Selecting one of
#: these emits a warning rather than failing, so an operator can still pin an
#: old model deliberately for a reproduction run.
RETIRED_MODELS: dict[str, str] = {
    "gpt-4o": "API shutdown 2026-10-23",
    "gpt-4o-2024-05-13": "API shutdown 2026-10-23",
    "gpt-4o-2024-08-06": "API shutdown 2026-10-23",
    "gpt-4o-2024-11-20": "API shutdown 2026-10-23",
}

# Defaults chosen 2026-09; both support vision and strict JSON-schema structured
# output. Override with PLATO_MODEL_CHEAP / PLATO_MODEL_STRONG -- see README.
DEFAULT_CHEAP_MODEL = "gpt-4.1-mini"
DEFAULT_STRONG_MODEL = "gpt-4.1"

ENV_CHEAP = "PLATO_MODEL_CHEAP"
ENV_STRONG = "PLATO_MODEL_STRONG"
ENV_API_KEY = "OPENAI_API_KEY"
ENV_BASE_URL = "PLATO_LLM_BASE_URL"

def _resolve(env_var: str, default: str) -> str:
    model = os.environ.get(env_var, "").strip() or default
    if model in RETIRED_MODELS:
        warnings.warn(
            f"{env_var}={model} is retired ({RETIRED_MODELS[model]}). "
            f"Set {env_var} to a current model.",
            RuntimeWarning,
            stacklevel=3,
        )
    return model

def cheap_model() -> str:
    """Model for Scene Understanding and Goal Extraction."""
    return _resolve(ENV_CHEAP, DEFAULT_CHEAP_MODEL)

def strong_model() -> str:
    """Model for the High-Level Planner, Step Planner and Verification reasoning."""
    return _resolve(ENV_STRONG, DEFAULT_STRONG_MODEL)

def base_url() -> str | None:
    """Optional OpenAI-compatible base URL (Azure, vLLM, OpenRouter, ...)."""
    return os.environ.get(ENV_BASE_URL, "").strip() or None

def api_key() -> str | None:
    """The API key, or None if unset. Never hardcoded, never logged."""
    return os.environ.get(ENV_API_KEY, "").strip() or None

def describe() -> dict:
    """Non-secret snapshot of model configuration, safe to write into trial logs."""
    return {
        "cheap_model": cheap_model(),
        "strong_model": strong_model(),
        "base_url": base_url(),
        "api_key_present": api_key() is not None,
    }
