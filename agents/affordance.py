"""Affordance / Tool-Use Agent (CLAUDE.md architecture item 4).

Maps one high-level sub-task to (a) an action drawn from the fixed vocabulary,
(b) the correct tool for it, and (c) the target object/container -- then asks a
grasp provider for a grasp on that tool.

Grasping itself is PLATO's existing module, reused rather than reimplemented.
Upstream exposes it as ``do_grasp`` in ``grasping/grasping/os_tog/notebooks``,
which is a git submodule that is not populated in a plain clone and which needs
the real camera and network weights. So this file defines a narrow
:class:`GraspProvider` interface with two implementations:

  * :class:`PlatoGraspProvider` -- delegates to PLATO's ``do_grasp`` when the
    submodule and hardware are actually available.
  * :class:`MockGraspProvider`  -- returns a nominal grasp for the mock cell, so
    the whole pipeline runs headless (item 11).

The handle flag produced by the Scene Understanding Agent is the same signal
PLATO's grasping module consumes, so the two agree on what "graspable by the
handle" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agents import llm_client
from agents.action_vocabulary import (
    ACTION_NAMES,
    Action,
    InfeasibleStep,
    action_enum_schema,
    parse_action,
    vocabulary_prompt_block,
)
from agents.high_level_planner import SubTask
from agents.scene_understanding import Scene
from config import models
from config.robot_profile import DEFAULT_PROFILE, RobotProfile

AGENT_NAME = "AffordanceAgent"

_SYSTEM_PROMPT_TEMPLATE = """You are the Affordance / Tool-Use Agent of an autonomous robotic chemist.

You are given ONE sub-task from the high-level plan and the objects the Scene
Understanding Agent grounded in the workspace. Decide:

  action        -- which single action from the fixed vocabulary realises this
                   sub-task.
  tool          -- the object the gripper must be holding to perform it, or "none"
                   if the action needs an empty gripper (PICKUP always does).
  target_object -- the object or container the action acts upon, or "none".
  location      -- the workspace position to work at, named exactly as listed.
  needs_pickup_first -- true if `tool` is not "none" and the gripper would have to
                   pick that tool up before this action can happen.

{vocabulary_block}

Affordance guidance:
  * Powders are transferred with a scoop (SCOOP), never poured from their tub.
  * Drop-scale liquid additions use the pipette (PIPETTE_DISPENSE).
  * Bulk liquid transfer between containers is POUR, and the source container
    must be held first.
  * Dissolving or mixing is STIR, with a held implement.
  * Containers are not relocated unless the sub-task requires it; prefer moving
    contents into a stationary container.
  * An object with a handle is grasped by the handle; one without is grasped by
    the body. The scene description tells you which is which.

{robot_block}

{positions_block}

If this sub-task cannot be realised by exactly one action from the vocabulary,
set "feasible" to false and say what capability is missing. Do NOT approximate it
with a different action and do NOT invent an action name.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "infeasible_reason": {"type": "string"},
        "action": action_enum_schema(),
        "tool": {"type": "string"},
        "target_object": {"type": "string"},
        "location": {"type": "string"},
        "needs_pickup_first": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "feasible",
        "infeasible_reason",
        "action",
        "tool",
        "target_object",
        "location",
        "needs_pickup_first",
        "rationale",
    ],
    "additionalProperties": False,
}


@dataclass
class Grasp:
    """A grasp on a tool, in the robot base frame."""

    object_name: str
    position_cm: tuple[float, float, float]
    approach: str
    width_cm: float
    confidence: float
    source: str  # "plato_os_tog" | "mock"

    def as_dict(self) -> dict:
        return {
            "object_name": self.object_name,
            "position_cm": list(self.position_cm),
            "approach": self.approach,
            "width_cm": self.width_cm,
            "confidence": self.confidence,
            "source": self.source,
        }


class GraspProvider(Protocol):
    """Narrow interface over PLATO's grasping module."""

    def grasp_for(self, object_name: str, has_handle: bool) -> Grasp: ...


class MockGraspProvider:
    """Nominal grasps for the mock cell (item 11). No perception, no hardware."""

    def grasp_for(self, object_name: str, has_handle: bool) -> Grasp:
        return Grasp(
            object_name=object_name,
            position_cm=(0.0, 0.0, 0.0),
            approach="handle" if has_handle else "body",
            width_cm=2.0 if has_handle else 6.0,
            confidence=1.0,
            source="mock",
        )


class PlatoGraspProvider:
    """Delegates to PLATO's existing ``do_grasp`` (grasping/os_tog submodule).

    Only usable when the submodule is initialised and the real cell is attached.
    Import is deferred so that a plain clone -- where the submodule is empty --
    still imports this module fine.
    """

    def __init__(self, save_path: str, gripper_cam, arm) -> None:
        self.save_path = save_path
        self.gripper_cam = gripper_cam
        self.arm = arm

    def grasp_for(self, object_name: str, has_handle: bool) -> Grasp:
        try:
            from grasping import do_grasp  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - requires submodule + hardware
            raise RuntimeError(
                "PLATO's grasping module is unavailable. Initialise the submodule with "
                "'git submodule update --init --recursive' inside PLATO/, or run with "
                "PLATO_HARDWARE=mock to use MockGraspProvider."
            ) from exc
        result = do_grasp(
            self.save_path,
            self.gripper_cam,
            query_tool=object_name,
            query_task="pickup",
            fa=self.arm,
        )
        position = tuple(float(v) for v in (result or (0.0, 0.0, 0.0)))[:3]
        return Grasp(
            object_name=object_name,
            position_cm=position,  # type: ignore[arg-type]
            approach="handle" if has_handle else "body",
            width_cm=2.0 if has_handle else 6.0,
            confidence=1.0,
            source="plato_os_tog",
        )


@dataclass
class Affordance:
    """The Affordance Agent's decision for one sub-task."""

    subtask: SubTask
    action: Action
    tool: str | None
    target_object: str | None
    location: str
    needs_pickup_first: bool
    rationale: str
    grasp: Grasp | None = None

    def as_dict(self) -> dict:
        return {
            "subtask_index": self.subtask.index,
            "subtask": self.subtask.description,
            "action": self.action.value,
            "tool": self.tool,
            "target_object": self.target_object,
            "location": self.location,
            "needs_pickup_first": self.needs_pickup_first,
            "rationale": self.rationale,
            "grasp": self.grasp.as_dict() if self.grasp else None,
        }


def _none_if_blank(value: str) -> str | None:
    text = str(value or "").strip()
    return None if text.lower() in ("", "none", "null", "n/a") else text


def resolve_affordance(
    subtask: SubTask,
    scene: Scene,
    *,
    grasp_provider: GraspProvider | None = None,
    profile: RobotProfile = DEFAULT_PROFILE,
    model: str | None = None,
    client=None,
) -> Affordance:
    """Map one sub-task to a vocabulary action, tool, target and grasp.

    Raises :class:`InfeasibleStep` if the sub-task cannot be expressed with the
    fixed vocabulary -- an explicit, loggable planner failure (item 5).
    """
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
                f"Sub-task {subtask.index}: {subtask.description}\n"
                f"Why this step exists: {subtask.rationale}\n\n"
                f"{scene.as_prompt_block()}\n\n"
                f"Choose the action, tool, target and location."
            )
        ],
        schema=_SCHEMA,
        schema_name="affordance_decision",
        client=client,
    )

    if not response.get("feasible", True):
        raise InfeasibleStep(
            subtask.description,
            str(response.get("infeasible_reason", "")).strip()
            or f"affordance agent reported no action in {ACTION_NAMES} realises this sub-task",
        )

    try:
        action = parse_action(response.get("action", ""))
    except InfeasibleStep as exc:
        raise InfeasibleStep(subtask.description, exc.reason) from exc

    tool = _none_if_blank(response.get("tool", ""))
    target = _none_if_blank(response.get("target_object", ""))
    affordance = Affordance(
        subtask=subtask,
        action=action,
        tool=tool,
        target_object=target,
        location=str(response.get("location", "")).strip() or "Home Pose",
        needs_pickup_first=bool(response.get("needs_pickup_first", False)),
        rationale=str(response.get("rationale", "")).strip(),
    )

    grasp_target = tool or target
    if grasp_provider is not None and grasp_target:
        has_handle = next(
            (o.has_handle for o in scene.objects if o.name.lower() == grasp_target.lower()),
            False,
        )
        affordance.grasp = grasp_provider.grasp_for(grasp_target, has_handle)
    return affordance
