"""Tests for .env loading and the guards that keep a key out of git.

A leaked key cannot be un-leaked -- git history is permanent and the only real
remedy is rotating the key. So the protections are tested like any other
behaviour, not left to a convention someone has to remember.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import env, models

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_the_ordinary_forms():
    parsed = env.parse_env_text(
        "\n".join(
            [
                "# a comment",
                "",
                "OPENAI_API_KEY=sk-plain",
                'PLATO_MODEL_CHEAP="quoted-double"',
                "PLATO_MODEL_STRONG='quoted-single'",
                "export PLATO_HARDWARE=mock",
                "   PLATO_FAKE_SUCCESS_RATE = 0.8   ",
            ]
        )
    )
    assert parsed["OPENAI_API_KEY"] == "sk-plain"
    assert parsed["PLATO_MODEL_CHEAP"] == "quoted-double"
    assert parsed["PLATO_MODEL_STRONG"] == "quoted-single"
    assert parsed["PLATO_HARDWARE"] == "mock"
    assert parsed["PLATO_FAKE_SUCCESS_RATE"] == "0.8"


def test_ignores_comments_blanks_and_malformed_lines():
    parsed = env.parse_env_text("# only a comment\n\nNO_EQUALS_SIGN\n   \n")
    assert parsed == {}


def test_a_key_containing_equals_signs_survives():
    """Base64-ish keys contain '='; partition on the FIRST '=' only."""
    parsed = env.parse_env_text("OPENAI_API_KEY=sk-abc==def=")
    assert parsed["OPENAI_API_KEY"] == "sk-abc==def="


# --------------------------------------------------------------------------
# Loading precedence
# --------------------------------------------------------------------------


def test_loads_allowed_keys_from_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")

    applied = env.load_env(path, override=True)
    assert applied == ["OPENAI_API_KEY"]
    assert models.api_key() == "sk-from-file"


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """Docker env_file, CI secrets and lab machine env must not be overridden."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-real-environment")
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=sk-from-stale-checkout\n", encoding="utf-8")

    applied = env.load_env(path)
    assert applied == []
    assert models.api_key() == "sk-from-real-environment"


def test_unlisted_keys_are_not_imported(tmp_path, monkeypatch):
    """A stray line must not be able to redefine PATH or anything else."""
    monkeypatch.setenv("PATH", "/original/path")
    path = tmp_path / ".env"
    path.write_text("PATH=/malicious\nSOME_OTHER_VAR=x\n", encoding="utf-8")

    applied = env.load_env(path, override=True)
    assert applied == []
    import os

    assert os.environ["PATH"] == "/original/path"
    assert "SOME_OTHER_VAR" not in os.environ


def test_a_missing_env_file_is_not_an_error(tmp_path):
    assert env.load_env(tmp_path / "does_not_exist") == []


def test_load_env_returns_names_never_values(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=sk-secret-value\n", encoding="utf-8")

    applied = env.load_env(path, override=True)
    assert "sk-secret-value" not in json.dumps(applied)


# --------------------------------------------------------------------------
# Git guards
# --------------------------------------------------------------------------


def test_gitignore_excludes_dotenv():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in ignore]
    assert ".env" in stripped, ".gitignore must exclude .env"
    assert "!.env.example" in stripped, ".env.example must stay committed as the template"


def test_the_example_file_holds_no_real_key():
    """A template with a real value in it is the classic way keys leak."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("OPENAI_API_KEY"):
            _, _, value = line.partition("=")
            assert value.strip() == "", ".env.example must ship with an empty key"
    assert "sk-" not in text


def test_no_real_env_file_is_tracked_anywhere_in_the_repo():
    """Catch a .env that got copied somewhere unexpected (e.g. into tests/)."""
    strays = [
        p for p in ROOT.rglob(".env")
        if ".git" not in p.parts and "PLATO" not in p.parts
    ]
    for stray in strays:
        # Existing on disk is fine -- it is gitignored. Being inside a package
        # directory that gets shipped is not.
        assert stray.parent == ROOT, f"unexpected .env location: {stray}"


def test_the_logger_records_key_presence_not_the_key(monkeypatch, tmp_path):
    from agents.logging_agent import TrialLogger

    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-never-appear-in-a-log")
    payload = TrialLogger("secret_check", log_dir=tmp_path).finalise()

    assert payload["model_config"]["api_key_present"] is True
    assert "sk-must-never-appear-in-a-log" not in json.dumps(payload)
    written = (tmp_path / "secret_check.jsonl").read_text(encoding="utf-8")
    assert "sk-must-never-appear-in-a-log" not in written


def test_the_missing_key_error_names_the_variable_not_a_value(monkeypatch):
    from agents.llm_client import MissingAPIKeyError, get_client

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        get_client("TestAgent")
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert "sk-" not in str(excinfo.value)
