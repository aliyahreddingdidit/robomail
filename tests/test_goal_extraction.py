"""Tests for the Goal Extraction Agent (CLAUDE.md item 2).

The load-bearing property is the one the architecture depends on: this agent
produces ONE goal and never a plan. Decomposition belongs to the High-Level
Planner, and the guard rails here are what stop the two collapsing together.

The vision call itself needs an API key. The tests that need one are skipped
with an explicit reason rather than silently passing -- see
``test_extract_goal_against_real_booklet_pages``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agents import goal_extraction
from agents.goal_extraction import (
    ExtractedGoal,
    GoalValidationError,
    extract_goal,
    normalise_goal,
    validate_goal,
)
from agents.llm_client import MissingAPIKeyError
from config import models
from tests.fake_llm import FakeLLM, goal as canned_goal

FIXTURES = Path(__file__).parent / "fixtures"
BOOKLET_PAGES = [FIXTURES / "real_booklet_page_12.png", FIXTURES / "real_booklet_page_13.png"]


# --------------------------------------------------------------------------
# The single-goal contract
# --------------------------------------------------------------------------

GOOD_GOALS = [
    "turn the cabbage indicator solution basic",
    "produce a fizzing reaction in the cup",
    "make the powder expand into snow",
    "show that the sand does not get wet",
    "turn the indicator red using the acidic solution",
]

PLANS_NOT_GOALS = [
    "scoop baking soda, add it to the indicator, then stir",
    "1. add water 2. add citric acid 3. add baking soda",
    "Step 1: pour the indicator into the cup",
    "add 10 mL of warm water and 1 big scoop of citric acid to a clear cup, then mix well",
    "pour the red cup in; afterwards pour the blue cup in",
    "identify the beaker and the cups; pour the indicator",
    "pour the red and blue cups into the empty cup and stir until purple",
]


@pytest.mark.parametrize("text", GOOD_GOALS)
def test_valid_goals_are_accepted(text):
    assert validate_goal(text) == text


@pytest.mark.parametrize("text", PLANS_NOT_GOALS)
def test_plans_are_rejected_not_quietly_accepted(text):
    with pytest.raises(GoalValidationError):
        validate_goal(text)


def test_rejection_message_names_whose_job_decomposition_is():
    with pytest.raises(GoalValidationError, match="High-Level Planner"):
        validate_goal("scoop the powder, then stir the cup")


def test_empty_goal_is_rejected():
    with pytest.raises(GoalValidationError, match="empty"):
        validate_goal("   ")


def test_overlong_goal_is_rejected_as_a_plan():
    long_goal = " ".join(["turn"] * (goal_extraction.MAX_GOAL_WORDS + 1))
    with pytest.raises(GoalValidationError, match="become a plan"):
        validate_goal(long_goal)


def test_multiline_output_is_rejected():
    with pytest.raises(GoalValidationError):
        validate_goal("turn the indicator basic\nadd more baking soda")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('  "turn the indicator basic"  ', "turn the indicator basic"),
        ("Goal: turn the indicator basic.", "turn the indicator basic"),
        ("The goal is turn the indicator basic", "turn the indicator basic"),
        ("turn   the  indicator   basic", "turn the indicator basic"),
    ],
)
def test_normalisation_strips_model_boilerplate(raw, expected):
    assert normalise_goal(raw) == expected
    assert validate_goal(raw) == expected


# --------------------------------------------------------------------------
# The agent itself, through a transport double
# --------------------------------------------------------------------------


def test_extract_goal_returns_only_a_goal():
    """The agent's output surface must not grow steps or an object inventory."""
    llm = FakeLLM({"extracted_goal": canned_goal("turn the cabbage indicator solution basic")})
    result = extract_goal(BOOKLET_PAGES[0], client=llm)

    assert isinstance(result, ExtractedGoal)
    assert result.goal == "turn the cabbage indicator solution basic"
    assert set(result.as_dict()) == {"goal", "page_topic", "reads_page_clearly", "source_image"}


def test_extract_goal_sends_the_page_image_not_a_text_transcription():
    """The booklet page reaches the model as an image, as if from the robot's camera."""
    llm = FakeLLM({"extracted_goal": canned_goal("make the powder expand into snow")})
    extract_goal(BOOKLET_PAGES[0], client=llm)

    call = llm.calls_for("extracted_goal")[0]
    assert "camera" in call.user_text.lower()
    # The system prompt must forbid decomposition, or the guard rails are the
    # only thing standing between this agent and the Planner's job.
    assert "do not list steps" in call.system_prompt.lower()


def test_extract_goal_uses_the_cheap_model_tier():
    llm = FakeLLM({"extracted_goal": canned_goal("produce a fizzing reaction")})
    extract_goal(BOOKLET_PAGES[0], client=llm)
    assert llm.calls_for("extracted_goal")[0].model == models.cheap_model()
    assert llm.calls_for("extracted_goal")[0].model != "gpt-4o"


def test_extract_goal_rejects_a_model_that_returns_a_plan():
    """If the model drifts into decomposition, the agent fails loudly."""
    llm = FakeLLM({"extracted_goal": canned_goal("scoop the soda, add it, then stir")})
    with pytest.raises(GoalValidationError):
        extract_goal(BOOKLET_PAGES[0], client=llm)


def test_extract_goal_propagates_the_clarity_flag():
    llm = FakeLLM({"extracted_goal": canned_goal("turn the indicator basic", clear=False)})
    assert extract_goal(BOOKLET_PAGES[0], client=llm).reads_page_clearly is False


def test_missing_image_is_reported_clearly():
    llm = FakeLLM({"extracted_goal": canned_goal("turn the indicator basic")})
    with pytest.raises(FileNotFoundError):
        extract_goal(FIXTURES / "no_such_page.png", client=llm)


def test_no_api_key_raises_a_named_actionable_error(monkeypatch):
    monkeypatch.delenv(models.ENV_API_KEY, raising=False)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        extract_goal(BOOKLET_PAGES[0])
    assert "GoalExtractionAgent" in str(excinfo.value)
    assert models.ENV_API_KEY in str(excinfo.value)


# --------------------------------------------------------------------------
# Live check against the real booklet photographs
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get(models.ENV_API_KEY),
    reason=(
        f"needs {models.ENV_API_KEY}: this is the only check that the agent reads a REAL "
        f"booklet photograph correctly. Without a key it is recorded as unverified in "
        f"docs/progress.md rather than silently passing."
    ),
)
@pytest.mark.parametrize("page", BOOKLET_PAGES, ids=lambda p: p.name)
def test_extract_goal_against_real_booklet_pages(page):
    """Given a booklet page photo, does the agent produce one reasonable goal?"""
    result = extract_goal(page)
    assert result.goal
    validate_goal(result.goal)  # still a goal, not a plan
    assert len(result.goal.split()) <= goal_extraction.MAX_GOAL_WORDS
    # Both available pages are the Magic Beaker pH-indicator spread.
    assert any(
        word in result.goal.lower()
        for word in ("indicator", "colour", "color", "acid", "base", "basic", "purple",
                     "blue", "red", "fizz", "ph")
    ), f"goal does not mention anything on the page: {result.goal!r}"
