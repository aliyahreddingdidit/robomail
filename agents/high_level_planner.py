"""High-Level Planner Agent (CLAUDE.md architecture item 3).

Adapted from PLATO/PLATO/overall_planner.py. Changes from upstream:

  * Takes the single goal string from the Goal Extraction Agent (item 2) and
    decomposes it ITSELF. Nothing upstream pre-decides the plan -- this is the
    actual planning step and the paper's core claim lives here.
  * The robot capability profile (item 6) is injected into the system prompt, so
    physically impossible instructions ("pour both cups at the same time") are
    reasoned around autonomously at plan time. No human review gate.
  * Every profile-driven adjustment is reported as a structured
    ``CapabilityWorkaround`` so the Logging Agent can record it (item 8).
  * Structured JSON output replaces upstream's "split on 'overall plan:'" text
    scraping, which broke whenever the model reformatted its answer.
  * ``model='gpt-4o'`` replaced by the env-driven strong tier.
  * Adds :func:`replan` for closed-loop correction (item 9): the planner is given
    the failed sub-task and the Verification Agent's reason string and produces a
    corrective sub-plan itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from agents import llm_client
from agents.action_vocabulary import vocabulary_prompt_block
from agents.scene_understanding import Scene
from config import models
from config.robot_profile import DEFAULT_PROFILE, CapabilityWorkaround, RobotProfile

AGENT_NAME = "HighLevelPlannerAgent"

_SYSTEM_PROMPT_TEMPLATE = """You are the High-Level Planner Agent of an autonomous robotic chemist.

You are given ONE goal describing an outcome, plus a description of the objects
the Scene Understanding Agent grounded in the workspace. Decompose the goal into
an ordered list of sub-tasks that a single robot arm can carry out to achieve it.

You are the planning step. Nobody upstream has decided the plan for you, and
nobody downstream will second-guess your chemistry. Reason about which reagents
and which order actually produce the requested outcome.

{robot_block}

{positions_block}

Each sub-task must be a short imperative clause naming what to do and to what,
for example "scoop one measure of sodium bicarbonate into the indicator cup".
A downstream Affordance Agent picks the tool and a downstream Step Planner emits
the motion; you do not emit coordinates, joint angles or gripper commands.

The downstream executor is restricted to a fixed action vocabulary. Write
sub-tasks that are achievable with it:
{vocabulary_block}

Do NOT add a sub-task for taking a photo, observing, checking or verifying. The
orchestrator captures the verification frames itself, and there is no camera
action in the vocabulary. Every sub-task you emit must be a physical
manipulation. Name the container to be observed in "observation_target" instead.

Plan the gripper explicitly: the arm starts empty, so a sub-task that uses a tool
must be preceded by a sub-task that picks that tool up, and the tool must be put
back down before a different one is used.

Autonomy rules -- these are strict:
  * Never plan a step that asks a human to do, check, approve or confirm anything.
  * Never plan two things happening simultaneously. If the source material calls
    for simultaneity, serialise it and record a capability workaround.
  * If the goal is a demonstration that something does NOT change (for example
    showing a material stays dry), the correct plan still performs the action and
    observes; absence of change is then the SUCCESS condition, not a failure.
    Set "expected_observation" accordingly so the Verification Agent is not
    told to hunt for a change that should not occur.
  * If the goal cannot be achieved with the objects present, set "feasible" to
    false and explain; do not invent objects.

For each sub-task also give:
  rationale            -- one clause on why this step is needed.
  target_container     -- the container whose state this step changes, or "none".

For the plan as a whole give:
  observation_target   -- the container whose state change proves the goal was met.
  expected_observation -- what the Verification Agent should look for at the end,
                          phrased as an observable visual outcome.
  verification_modality-- one of: color, motion, volume. Pick "color" for
                          indicator colour change, "motion" for effervescence or
                          bubbling, "volume" for expansion or level change.
  capability_workarounds - one entry for EVERY time the robot capability profile
                          above changed your plan relative to a literal reading of
                          the goal or the source material. Empty list if none.
"""

_REPLAN_PROMPT_TEMPLATE = """You are the High-Level Planner Agent of an autonomous robotic chemist,
performing closed-loop correction.

A plan you produced was executed and the Chemistry Outcome Verification Agent
reported FAILURE. Produce a corrective sub-plan.

{robot_block}

{positions_block}

{vocabulary_block}

You will be given: the goal, the workspace objects, the plan as executed, which
sub-tasks succeeded, which sub-task failed, the verification agent's verdict and
its reason string, and how many corrective attempts have already been made.

Rules:
  * Diagnose first: say in "diagnosis" why you think the outcome did not occur.
  * Then emit a corrective plan that starts from the CURRENT state of the bench.
    Steps that already succeeded have already happened -- do not blindly repeat
    them unless repeating is genuinely part of the correction (e.g. adding a
    further measure of reagent is a legitimate correction and the intended
    headline result of this system).
  * Never ask a human for help, approval or diagnosis.
  * If you conclude the goal is genuinely unreachable with the objects present,
    set "feasible" to false and explain. That is an honest reported failure and
    is preferable to a plan you do not believe in.
"""


def _subtask_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "rationale": {"type": "string"},
            "target_container": {"type": "string"},
        },
        "required": ["description", "rationale", "target_container"],
        "additionalProperties": False,
    }


def _workaround_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "constraint": {"type": "string"},
            "literal_instruction": {"type": "string"},
            "adopted_approach": {"type": "string"},
            "outcome_differs": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": [
            "constraint",
            "literal_instruction",
            "adopted_approach",
            "outcome_differs",
            "note",
        ],
        "additionalProperties": False,
    }


_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "infeasible_reason": {"type": "string"},
        "subtasks": {"type": "array", "items": _subtask_schema()},
        "observation_target": {"type": "string"},
        "expected_observation": {"type": "string"},
        "verification_modality": {"type": "string", "enum": ["color", "motion", "volume"]},
        "capability_workarounds": {"type": "array", "items": _workaround_schema()},
    },
    "required": [
        "feasible",
        "infeasible_reason",
        "subtasks",
        "observation_target",
        "expected_observation",
        "verification_modality",
        "capability_workarounds",
    ],
    "additionalProperties": False,
}

_REPLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "feasible": {"type": "boolean"},
        "infeasible_reason": {"type": "string"},
        "subtasks": {"type": "array", "items": _subtask_schema()},
        "observation_target": {"type": "string"},
        "expected_observation": {"type": "string"},
        "verification_modality": {"type": "string", "enum": ["color", "motion", "volume"]},
        "capability_workarounds": {"type": "array", "items": _workaround_schema()},
    },
    "required": [
        "diagnosis",
        "feasible",
        "infeasible_reason",
        "subtasks",
        "observation_target",
        "expected_observation",
        "verification_modality",
        "capability_workarounds",
    ],
    "additionalProperties": False,
}


@dataclass
class SubTask:
    """One ordered step of the high-level plan."""

    index: int
    description: str
    rationale: str = ""
    target_container: str = "none"

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "rationale": self.rationale,
            "target_container": self.target_container,
        }


@dataclass
class Plan:
    """The High-Level Planner Agent's output."""

    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    observation_target: str = ""
    expected_observation: str = ""
    verification_modality: str = "color"
    capability_workarounds: list[CapabilityWorkaround] = field(default_factory=list)
    feasible: bool = True
    infeasible_reason: str = ""
    diagnosis: str = ""
    is_correction: bool = False

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "subtasks": [s.as_dict() for s in self.subtasks],
            "observation_target": self.observation_target,
            "expected_observation": self.expected_observation,
            "verification_modality": self.verification_modality,
            "capability_workarounds": [w.as_dict() for w in self.capability_workarounds],
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "diagnosis": self.diagnosis,
            "is_correction": self.is_correction,
        }

    def as_prompt_block(self) -> str:
        if not self.subtasks:
            return "(empty plan)"
        return "\n".join(f"  {s.index}. {s.description}" for s in self.subtasks)


def _system_prompt(template: str, profile: RobotProfile) -> str:
    return template.format(
        robot_block=profile.as_prompt_block(),
        positions_block=profile.positions_prompt_block(),
        vocabulary_block=vocabulary_prompt_block(),
    )


def _parse_plan(goal: str, response: dict, *, is_correction: bool) -> Plan:
    subtasks = [
        SubTask(
            index=i,
            description=str(s["description"]).strip(),
            rationale=str(s.get("rationale", "")).strip(),
            target_container=str(s.get("target_container", "none")).strip(),
        )
        for i, s in enumerate(response.get("subtasks", []), start=1)
    ]
    workarounds = [
        CapabilityWorkaround(
            constraint=str(w.get("constraint", "")),
            literal_instruction=str(w.get("literal_instruction", "")),
            adopted_approach=str(w.get("adopted_approach", "")),
            outcome_differs=bool(w.get("outcome_differs", False)),
            note=str(w.get("note", "")),
        )
        for w in response.get("capability_workarounds", [])
    ]
    return Plan(
        goal=goal,
        subtasks=subtasks,
        observation_target=str(response.get("observation_target", "")).strip(),
        expected_observation=str(response.get("expected_observation", "")).strip(),
        verification_modality=str(response.get("verification_modality", "color")).strip().lower(),
        capability_workarounds=workarounds,
        feasible=bool(response.get("feasible", True)),
        infeasible_reason=str(response.get("infeasible_reason", "")).strip(),
        diagnosis=str(response.get("diagnosis", "")).strip(),
        is_correction=is_correction,
    )


def plan_goal(
    goal: str,
    scene: Scene,
    *,
    profile: RobotProfile = DEFAULT_PROFILE,
    model: str | None = None,
    client=None,
) -> Plan:
    """Decompose ``goal`` into an ordered sub-task plan, grounded in ``scene``."""
    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or models.strong_model(),
        system_prompt=_system_prompt(_SYSTEM_PROMPT_TEMPLATE, profile),
        user_content=[
            llm_client.text_block(
                f"Goal: {goal}\n\n{scene.as_prompt_block()}\n\n"
                "Decompose this goal into an ordered sub-task plan."
            )
        ],
        schema=_PLAN_SCHEMA,
        schema_name="subtask_plan",
        client=client,
    )
    return _parse_plan(goal, response, is_correction=False)


def replan(
    goal: str,
    scene: Scene,
    failed_plan: Plan,
    *,
    failed_subtask: SubTask | None,
    completed_subtasks: Sequence[SubTask],
    verification_reason: str,
    attempt: int,
    max_attempts: int,
    profile: RobotProfile = DEFAULT_PROFILE,
    model: str | None = None,
    client=None,
) -> Plan:
    """Generate a corrective sub-plan after a Verification Agent failure (item 9)."""
    completed = (
        "\n".join(f"  - {s.description}" for s in completed_subtasks) or "  (none)"
    )
    failed_text = failed_subtask.description if failed_subtask else "(whole-plan outcome check)"
    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or models.strong_model(),
        system_prompt=_system_prompt(_REPLAN_PROMPT_TEMPLATE, profile),
        user_content=[
            llm_client.text_block(
                f"Goal: {goal}\n\n"
                f"{scene.as_prompt_block()}\n\n"
                f"Plan as executed:\n{failed_plan.as_prompt_block()}\n\n"
                f"Sub-tasks that succeeded:\n{completed}\n\n"
                f"Sub-task that failed verification: {failed_text}\n\n"
                f"Expected observation: {failed_plan.expected_observation}\n"
                f"Verification Agent verdict: FAILURE\n"
                f"Verification Agent reason: {verification_reason}\n\n"
                f"This is corrective attempt {attempt} of at most {max_attempts}.\n"
                "Diagnose and produce a corrective sub-plan."
            )
        ],
        schema=_REPLAN_SCHEMA,
        schema_name="corrective_plan",
        client=client,
    )
    return _parse_plan(goal, response, is_correction=True)
