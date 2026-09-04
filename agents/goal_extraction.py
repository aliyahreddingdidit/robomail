"""Goal Extraction Agent (CLAUDE.md architecture item 2). NEW.

Input:  a photo of ONE page of the physical instruction booklet. This stands in
        for the robot's own camera reading the manual -- it is treated exactly
        that way, never as a human-typed text field.
Output: ONE short natural-language goal prompt describing the *outcome* the page
        is asking for. For example: "turn the cabbage indicator solution basic".

Deliberately NOT a task-spec parser. It does not emit steps, sub-tasks, object
inventories, reagent quantities or an ordering. Decomposing the goal is the
High-Level Planner's job (item 3); collapsing the two defeats the point of the
architecture and was an abandoned earlier design that needed a human review gate.

The guard rails below are load-bearing, not cosmetic: :func:`validate_goal`
rejects anything that has started to look like a plan, so the failure mode is a
loud rejection rather than a quiet scope creep into the Planner's territory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agents import llm_client
from config import models

AGENT_NAME = "GoalExtractionAgent"

#: A goal is one clause. Anything longer has almost certainly become a plan.
MAX_GOAL_WORDS = 18
MAX_GOAL_CHARS = 140

#: Markers that indicate the model drifted into decomposition.
_STEP_MARKERS = (
    re.compile(r"\bstep\s*\d", re.I),
    re.compile(r"^\s*\d+\s*[.)]", re.M),      # "1." / "2)" enumerations
    re.compile(r"\bthen\b", re.I),            # sequencing word: always a plan
    re.compile(r"\bafter(wards| that)\b", re.I),
    re.compile(r"\bfollowed by\b", re.I),
    re.compile(r"\bnext,", re.I),
    re.compile(r"[;•]"),                      # semicolons / bullets
)

#: Verbs that name a manipulation *procedure* rather than an outcome. A goal may
#: contain at most one; two or more means the model has written a recipe.
_PROCEDURE_VERBS = (
    "scoop", "pour", "stir", "mix", "dispense", "pipette", "grasp", "pick up",
    "place", "add", "dissolve", "fill", "transfer", "measure", "shake", "sprinkle",
)
MAX_PROCEDURE_VERBS = 1

_SYSTEM_PROMPT = """You are the Goal Extraction Agent of an autonomous robotic chemist.

You are looking at a frame from the robot's own camera, pointed at one page of a
printed chemistry-kit instruction booklet. Nobody has transcribed it for you.
Read the page yourself.

Your ONLY job is to state, in one short natural-language clause, the OUTCOME
that this page is asking for. Think: "what is the observable end state that would
mean this page's experiment worked?"

Rules -- these are strict:
  * Output exactly ONE goal clause, under 18 words.
  * Describe an OUTCOME or END STATE, not a procedure.
  * Do NOT list steps, sub-tasks, actions, reagent quantities, or an ordering.
  * Do NOT enumerate the objects or equipment involved.
  * Do NOT use "first", "then", "next", "after that", numbered lists, or semicolons.
  * A separate planner agent will decompose your goal and will read the scene
    itself. It does not need, and must not be given, your version of the recipe.
  * If the page shows several related demonstrations, state the single overarching
    outcome that the page is building towards, not each one.

Good goals:
  "turn the cabbage indicator solution basic"
  "produce a fizzing reaction in the cup"
  "make the powder expand into snow"
  "show that the sand stays dry in water"

Bad goals (these are plans, and will be rejected):
  "scoop baking soda, add it to the indicator, then stir"
  "add 10 mL of warm water and 1 big scoop of citric acid to a clear cup"
  "identify the beaker, the cups and the scoop, then pour"

Also report, separately from the goal:
  page_topic          -- a two-to-five word label for what the page is about
  reads_page_clearly  -- false if the photo is too blurry/cropped to read confidently
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "One short outcome clause, under 18 words. No steps.",
        },
        "page_topic": {"type": "string"},
        "reads_page_clearly": {"type": "boolean"},
    },
    "required": ["goal", "page_topic", "reads_page_clearly"],
    "additionalProperties": False,
}


class GoalValidationError(ValueError):
    """The extracted goal violated the single-goal contract."""


@dataclass
class ExtractedGoal:
    """The Goal Extraction Agent's entire output surface."""

    goal: str
    page_topic: str
    reads_page_clearly: bool
    source_image: str

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "page_topic": self.page_topic,
            "reads_page_clearly": self.reads_page_clearly,
            "source_image": self.source_image,
        }


def normalise_goal(raw: str) -> str:
    """Trim a model-supplied goal to its canonical single-clause form."""
    text = " ".join(str(raw).split())
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"^(the\s+)?goal\s*(is|:)\s*", "", text, flags=re.I)
    return text.rstrip(".").strip()


def validate_goal(goal: str) -> str:
    """Enforce the single-goal contract. Raises :class:`GoalValidationError`.

    This is what keeps item 2 from quietly becoming item 3.
    """
    text = normalise_goal(goal)
    if not text:
        raise GoalValidationError("goal is empty")
    if len(text) > MAX_GOAL_CHARS:
        raise GoalValidationError(
            f"goal is {len(text)} chars (max {MAX_GOAL_CHARS}); it has become a plan, not a goal"
        )
    words = text.split()
    if len(words) > MAX_GOAL_WORDS:
        raise GoalValidationError(
            f"goal is {len(words)} words (max {MAX_GOAL_WORDS}); it has become a plan, not a goal"
        )
    if "\n" in goal.strip():
        raise GoalValidationError("goal spans multiple lines; expected a single clause")
    for marker in _STEP_MARKERS:
        if marker.search(text):
            raise GoalValidationError(
                f"goal contains step-decomposition marker {marker.pattern!r}; "
                "decomposition is the High-Level Planner's job, not this agent's"
            )
    lowered = text.lower()
    found = [v for v in _PROCEDURE_VERBS if re.search(rf"\b{re.escape(v)}\b", lowered)]
    if len(found) > MAX_PROCEDURE_VERBS:
        raise GoalValidationError(
            f"goal chains procedure verbs {found}; that is a recipe, not an outcome. "
            "Decomposition is the High-Level Planner's job, not this agent's"
        )
    return text


def extract_goal(
    booklet_page_image: str | Path,
    *,
    model: str | None = None,
    client=None,
) -> ExtractedGoal:
    """Read one booklet page image and return a single goal string.

    Raises :class:`agents.llm_client.MissingAPIKeyError` if no key is configured,
    and :class:`GoalValidationError` if the model produced a plan instead of a goal.
    """
    path = Path(booklet_page_image)
    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or models.cheap_model(),
        system_prompt=_SYSTEM_PROMPT,
        user_content=[
            llm_client.text_block(
                "This is a frame from your camera showing one page of the chemistry-kit "
                "booklet. State the single outcome this page is asking for."
            ),
            llm_client.image_block(path, detail="high"),
        ],
        schema=_SCHEMA,
        schema_name="extracted_goal",
        client=client,
    )
    goal = validate_goal(response.get("goal", ""))
    return ExtractedGoal(
        goal=goal,
        page_topic=str(response.get("page_topic", "")).strip(),
        reads_page_clearly=bool(response.get("reads_page_clearly", True)),
        source_image=str(path),
    )
