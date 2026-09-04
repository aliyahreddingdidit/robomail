"""Fixed low-level action vocabulary (CLAUDE.md architecture item 5).

Defined ONCE here and imported everywhere else. The Step Planner is constrained
to select only from this enum via structured output / JSON schema -- it cannot
invent an action name. If a sub-task genuinely cannot be expressed with these
actions, that is an explicit, reported planner failure (`InfeasibleStep`), never
a made-up action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Action(str, Enum):
    """The complete set of manipulation actions available to the robot."""

    PICKUP = "PICKUP"
    POUR = "POUR"
    SCOOP = "SCOOP"
    PIPETTE_DISPENSE = "PIPETTE_DISPENSE"
    STIR = "STIR"
    PLACE = "PLACE"


#: Canonical ordered list of action names, used to build JSON-schema enums.
ACTION_NAMES: list[str] = [a.value for a in Action]

#: Human-readable semantics, injected into agent system prompts so the model
#: picks between actions on meaning rather than on the name alone.
ACTION_SEMANTICS: dict[Action, str] = {
    Action.PICKUP: (
        "Grasp an object (tool or container) and lift it. Afterwards that object "
        "is held by the single gripper and no other object can be held."
    ),
    Action.POUR: (
        "Tip a currently-held container so its liquid contents flow into a target "
        "container. Requires the source container to already be held."
    ),
    Action.SCOOP: (
        "Use a held scoop/spoon to transfer a measured quantity of powder from a "
        "source container. Requires a scoop to already be held."
    ),
    Action.PIPETTE_DISPENSE: (
        "Use a held pipette to draw and release a small volume of liquid, for "
        "drop-scale additions. Requires a pipette to already be held."
    ),
    Action.STIR: (
        "Agitate the contents of a container with a held implement to dissolve or "
        "mix, without transferring anything between containers."
    ),
    Action.PLACE: (
        "Set a currently-held object down at a target location and release it, "
        "freeing the gripper."
    ),
}

#: Actions that require the gripper to already be holding something.
REQUIRES_HELD_OBJECT: frozenset[Action] = frozenset(
    {Action.POUR, Action.SCOOP, Action.PIPETTE_DISPENSE, Action.STIR, Action.PLACE}
)


class InfeasibleStep(Exception):
    """Raised when a sub-task cannot be expressed in the fixed vocabulary.

    This is a *reported* planning failure that the orchestrator logs and feeds
    back into replanning. It is deliberately not a fallback to free-text actions.
    """

    def __init__(self, subtask: str, reason: str) -> None:
        self.subtask = subtask
        self.reason = reason
        super().__init__(f"Cannot express sub-task {subtask!r} in the fixed action vocabulary: {reason}")


def parse_action(raw: str) -> Action:
    """Coerce a model-supplied string to an :class:`Action`.

    Structured output should already guarantee membership; this is the belt-and-
    braces check for providers whose schema enforcement is advisory.
    """
    if isinstance(raw, Action):
        return raw
    key = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return Action(key)
    except ValueError as exc:
        raise InfeasibleStep(
            subtask="<unknown>",
            reason=f"model returned action {raw!r}, which is not one of {ACTION_NAMES}",
        ) from exc


def vocabulary_prompt_block() -> str:
    """Render the vocabulary for injection into an LLM system prompt."""
    lines = [
        "You may ONLY use the following actions. This list is exhaustive and closed.",
        "You must not invent, rename, compose or hyphenate action names.",
        "",
    ]
    for action in Action:
        lines.append(f"  {action.value}: {ACTION_SEMANTICS[action]}")
    lines += [
        "",
        "If a sub-task cannot be accomplished with the actions above, do NOT approximate",
        "it with a different action and do NOT invent a name. Instead set",
        '"feasible": false and explain what capability is missing in "infeasible_reason".',
    ]
    return "\n".join(lines)


def action_enum_schema() -> dict:
    """JSON-schema fragment restricting a field to the fixed vocabulary."""
    return {"type": "string", "enum": list(ACTION_NAMES)}


@dataclass
class MotionPrimitive:
    """One low-level command produced by the Step Planner.

    Mirrors PLATO's `step_planner.py` output grammar (Go-to / Grasp / Tilt).
    """

    kind: str  # "GOTO" | "GRASP" | "TILT"
    location: str | None = None
    delta_cm: tuple[float, float, float] | None = None
    gripper: int | None = None  # 1 = close, 0 = open
    tilt_deg: tuple[float, float, float] | None = None
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "location": self.location,
            "delta_cm": list(self.delta_cm) if self.delta_cm else None,
            "gripper": self.gripper,
            "tilt_deg": list(self.tilt_deg) if self.tilt_deg else None,
            "explanation": self.explanation,
        }


@dataclass
class PlannedStep:
    """An enum-constrained action plus the motion primitives that realise it."""

    action: Action
    target_object: str | None
    tool: str | None
    location: str | None
    primitives: list[MotionPrimitive] = field(default_factory=list)
    rationale: str = ""

    def as_dict(self) -> dict:
        return {
            "action": self.action.value,
            "target_object": self.target_object,
            "tool": self.tool,
            "location": self.location,
            "primitives": [p.as_dict() for p in self.primitives],
            "rationale": self.rationale,
        }


def validate_sequence(actions: Sequence[Action]) -> list[str]:
    """Cheap structural sanity check over an action sequence.

    Returns a list of human-readable warnings (empty means clean). This encodes
    only the single-gripper physical truths from the robot profile; it does not
    second-guess the planner's chemistry.
    """
    warnings: list[str] = []
    holding = False
    for i, action in enumerate(actions, start=1):
        if action is Action.PICKUP:
            if holding:
                warnings.append(
                    f"step {i}: PICKUP while the single gripper is already holding something"
                )
            holding = True
        elif action is Action.PLACE:
            if not holding:
                warnings.append(f"step {i}: PLACE with an empty gripper")
            holding = False
        elif action in REQUIRES_HELD_OBJECT and not holding:
            warnings.append(
                f"step {i}: {action.value} requires a held tool/container but the gripper is empty"
            )
    return warnings
