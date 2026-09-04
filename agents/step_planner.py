"""Low-Level Action Generation (CLAUDE.md architecture item 5).

Adapted from PLATO/PLATO/step_planner.py. Stays LLM-driven: the model reasons out
the Go-to / Grasp / Tilt sequence for each action, exactly as upstream does. This
is deliberate and load-bearing -- the paper's claim is that an LLM-driven stack
degrades less under scene variation than a scripted baseline, so there are no
fixed or hand-authored trajectories anywhere in this module.

Changes from upstream:
  * The high-level action label is constrained to the fixed enum in
    :mod:`agents.action_vocabulary` via structured output, instead of free text.
  * Structured JSON replaces upstream's "split on 'steps list:'" text scraping.
  * The robot capability profile (item 6) is injected into the system prompt.
  * ``model='gpt-4o'`` replaced by the env-driven strong tier.
  * An action the vocabulary cannot express raises :class:`InfeasibleStep`, a
    reported failure that feeds replanning, rather than an invented action name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agents import llm_client
from agents.action_vocabulary import (
    Action,
    InfeasibleStep,
    MotionPrimitive,
    PlannedStep,
    action_enum_schema,
    parse_action,
    vocabulary_prompt_block,
)
from agents.affordance import Affordance
from config import models
from config.robot_profile import DEFAULT_PROFILE, RobotProfile

AGENT_NAME = "StepPlannerAgent"

PRIMITIVE_KINDS = ("GOTO", "GRASP", "TILT")

_SYSTEM_PROMPT_TEMPLATE = """You are the Low-Level Action Generation Agent of an autonomous robotic chemist.

You convert ONE high-level action into the low-level commands a parallel-plate
gripper can execute. Reason the motion out yourself; there is no library of
pre-recorded trajectories and you must not ask for one.

You will be given:
  Action        -- the action to realise, from a fixed vocabulary.
  Location      -- the semantic workspace position to work at (the object centroid).
  Positioning   -- where to be relative to that location (Behind, Above, ...).
  Description   -- [object LxWxH cm, tool length cm]; either array may be empty.
  Object        -- what the arm interacts with; not currently held.
  Tool          -- what the gripper currently holds; "none" if empty.
  Previous Steps- what you emitted for the immediately preceding action, for context.

{vocabulary_block}

Output a sequence of primitives, each exactly one of:
  GOTO  -- move the end effector to `location` plus (deltaX, deltaY, deltaZ) in cm.
  GRASP -- gripper 1 = close, 0 = open. Only for grasping or releasing.
  TILT  -- (ThetaX, ThetaY, ThetaZ) in degrees, RELATIVE to the current pose.

Robot coordinate system:
  The gripper (and any tool it holds) faces +X by default, its base mount behind
  it in -X. The line joining the gripper plates is the Y axis. Up is +Z.
  Tilts follow the right-hand rule and are relative, starting from (0, 0, 0).
    +ThetaY tilt up, -ThetaY tilt down, +ThetaZ rotate left, -ThetaZ rotate right,
    +ThetaX twist clockwise, -ThetaX twist anti-clockwise.
  A TILT primitive must have exactly one non-zero component.

Guidelines:
  * A trajectory is expressed as a sequence of GOTO poses, not a special command.
  * Account for tool length when positioning: if the gripper holds a 15 cm scoop
    and must reach an object, offset by the tool length plus half the object's
    relevant dimension.
  * When releasing something into a container, do not go to the container centroid
    (that is inside it). Release from about 10 cm above.
  * A pour is a GOTO above the target followed by a TILT about the correct axis,
    then a TILT back to level.
  * Assume every upstream step already succeeded.
  * Give a one-clause explanation for each primitive.

{robot_block}

{positions_block}

If the requested action cannot be realised with GOTO / GRASP / TILT given this
gripper, set "feasible" to false and say what is missing. Do not invent primitives.
"""

_PRIMITIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(PRIMITIVE_KINDS)},
        "location": {"type": "string"},
        "delta_x_cm": {"type": "number"},
        "delta_y_cm": {"type": "number"},
        "delta_z_cm": {"type": "number"},
        "gripper": {"type": "integer", "enum": [0, 1]},
        "theta_x_deg": {"type": "number"},
        "theta_y_deg": {"type": "number"},
        "theta_z_deg": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": [
        "kind",
        "location",
        "delta_x_cm",
        "delta_y_cm",
        "delta_z_cm",
        "gripper",
        "theta_x_deg",
        "theta_y_deg",
        "theta_z_deg",
        "explanation",
    ],
    "additionalProperties": False,
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "infeasible_reason": {"type": "string"},
        "action": action_enum_schema(),
        "rationale": {"type": "string"},
        "primitives": {"type": "array", "items": _PRIMITIVE_SCHEMA},
    },
    "required": ["feasible", "infeasible_reason", "action", "rationale", "primitives"],
    "additionalProperties": False,
}


@dataclass
class ObjectGeometry:
    """Object/tool dimensions in cm, in PLATO's Description format."""

    object_lwh_cm: tuple[float, float, float] | None = None
    tool_length_cm: float | None = None

    def as_description(self) -> str:
        obj = list(self.object_lwh_cm) if self.object_lwh_cm else []
        tool = [self.tool_length_cm] if self.tool_length_cm is not None else []
        return f"[{obj}, {tool}]"


#: Nominal dimensions for the chemistry kit, used only to give the Step Planner
#: the same geometric context PLATO's Description field carried. These are
#: measurements of the labware, NOT trajectories -- the motion is still reasoned
#: out by the model on every call.
KIT_GEOMETRY: dict[str, ObjectGeometry] = {
    "beaker": ObjectGeometry(object_lwh_cm=(7.0, 7.0, 10.0)),
    "clear cup": ObjectGeometry(object_lwh_cm=(8.0, 8.0, 9.0)),
    "paper cup": ObjectGeometry(object_lwh_cm=(8.0, 8.0, 9.0)),
    "pipette": ObjectGeometry(tool_length_cm=15.0),
    "measuring scoop": ObjectGeometry(tool_length_cm=12.0),
    "small scoop": ObjectGeometry(tool_length_cm=10.0),
    "big scoop": ObjectGeometry(tool_length_cm=13.0),
    "stirring rod": ObjectGeometry(tool_length_cm=18.0),
}


def geometry_for(object_name: str | None, tool_name: str | None) -> ObjectGeometry:
    """Best-effort geometry lookup by fuzzy name match; empty when unknown."""
    geom = ObjectGeometry()
    if object_name:
        for key, value in KIT_GEOMETRY.items():
            if key in object_name.lower() and value.object_lwh_cm:
                geom.object_lwh_cm = value.object_lwh_cm
                break
    if tool_name:
        for key, value in KIT_GEOMETRY.items():
            if key in tool_name.lower() and value.tool_length_cm is not None:
                geom.tool_length_cm = value.tool_length_cm
                break
    return geom


def _to_primitive(raw: dict) -> MotionPrimitive:
    kind = str(raw.get("kind", "")).upper()
    if kind not in PRIMITIVE_KINDS:
        raise InfeasibleStep("<primitive>", f"model emitted unknown primitive kind {kind!r}")
    explanation = str(raw.get("explanation", "")).strip()
    if kind == "GOTO":
        return MotionPrimitive(
            kind="GOTO",
            location=str(raw.get("location", "")).strip() or None,
            delta_cm=(
                float(raw.get("delta_x_cm", 0.0)),
                float(raw.get("delta_y_cm", 0.0)),
                float(raw.get("delta_z_cm", 0.0)),
            ),
            explanation=explanation,
        )
    if kind == "GRASP":
        return MotionPrimitive(
            kind="GRASP", gripper=int(raw.get("gripper", 1)), explanation=explanation
        )
    return MotionPrimitive(
        kind="TILT",
        tilt_deg=(
            float(raw.get("theta_x_deg", 0.0)),
            float(raw.get("theta_y_deg", 0.0)),
            float(raw.get("theta_z_deg", 0.0)),
        ),
        explanation=explanation,
    )


def plan_step(
    affordance: Affordance,
    *,
    positioning: str = "Behind",
    previous_steps: Sequence[PlannedStep] = (),
    profile: RobotProfile = DEFAULT_PROFILE,
    model: str | None = None,
    client=None,
) -> PlannedStep:
    """Turn one affordance decision into an enum-labelled motion sequence."""
    geometry = geometry_for(affordance.target_object, affordance.tool)
    prev = (
        "; ".join(
            f"{s.action.value} -> {len(s.primitives)} primitives" for s in previous_steps[-2:]
        )
        or "none"
    )
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        vocabulary_block=vocabulary_prompt_block(),
        robot_block=profile.as_prompt_block(),
        positions_block=profile.positions_prompt_block(),
    )
    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or models.strong_model(),
        system_prompt=system_prompt,
        user_content=[
            llm_client.text_block(
                f"Action: {affordance.action.value}\n"
                f"Location: {affordance.location}\n"
                f"Positioning: {positioning}\n"
                f"Description: {geometry.as_description()}\n"
                f"Object: {affordance.target_object or 'none'}\n"
                f"Tool: {affordance.tool or 'none'}\n"
                f"Previous Steps: {prev}\n\n"
                f"Sub-task this realises: {affordance.subtask.description}"
            )
        ],
        schema=_SCHEMA,
        schema_name="low_level_step",
        client=client,
    )

    if not response.get("feasible", True):
        raise InfeasibleStep(
            affordance.subtask.description,
            str(response.get("infeasible_reason", "")).strip()
            or "step planner reported the action is not realisable with GOTO/GRASP/TILT",
        )

    action = parse_action(response.get("action", affordance.action.value))
    if action is not affordance.action:
        # The Affordance Agent already committed to an action; the Step Planner
        # silently substituting a different one would hide a planning
        # disagreement. Report it instead.
        raise InfeasibleStep(
            affordance.subtask.description,
            f"step planner returned {action.value} but the affordance agent chose "
            f"{affordance.action.value}; refusing to silently substitute",
        )

    primitives = [_to_primitive(p) for p in response.get("primitives", [])]
    if not primitives:
        raise InfeasibleStep(
            affordance.subtask.description,
            "step planner returned an empty motion sequence",
        )

    return PlannedStep(
        action=action,
        target_object=affordance.target_object,
        tool=affordance.tool,
        location=affordance.location,
        primitives=primitives,
        rationale=str(response.get("rationale", "")).strip(),
    )


__all__ = [
    "AGENT_NAME",
    "Action",
    "KIT_GEOMETRY",
    "ObjectGeometry",
    "PlannedStep",
    "geometry_for",
    "plan_step",
]
