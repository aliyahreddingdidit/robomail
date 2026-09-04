"""A TEST-ONLY transport double for the LLM provider.

########################################################################
##  This is NOT part of the pipeline. It is a test fixture.           ##
##                                                                    ##
##  It mocks the *provider transport* (the openai client object), in  ##
##  exactly the way MockFrankaArm mocks the arm. It does not replace  ##
##  the planner with a script: production code never imports it, and  ##
##  `Orchestrator(llm_client_override=...)` is only ever set by tests.##
##                                                                    ##
##  What this buys: the closed-loop machinery -- ordering, the retry   ##
##  cap, the verification feedback path, the logging schema -- is     ##
##  verifiable on a machine with no API key. What it does NOT buy:    ##
##  any evidence that a real model plans these tasks well. That needs ##
##  a key and is recorded as unverified in docs/progress.md.          ##
########################################################################

Canned responses are keyed by the agent's JSON schema name, so a test states
what each agent returns without caring about call order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list[_Choice]


@dataclass
class RecordedCall:
    """One captured request, so tests can assert on what agents actually sent."""

    model: str
    schema_name: str
    system_prompt: str
    user_text: str


class FakeLLM:
    """Quacks like ``openai.OpenAI`` for :func:`llm_client.structured_completion`.

    ``responses`` maps a schema name to either a dict (returned every time) or a
    callable ``(call_index, request) -> dict`` for responses that must change
    between calls -- which is how a replanning test makes the second plan differ
    from the first.
    """

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[RecordedCall] = []
        self._counts: dict[str, int] = {}
        self.chat = _Chat(self)

    def _respond(self, request: dict) -> _Completion:
        fmt = request.get("response_format", {})
        schema_name = fmt.get("json_schema", {}).get("name", "")
        messages = request.get("messages", [])
        system_prompt = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else []
        user_text = " ".join(
            block.get("text", "") for block in user if isinstance(block, dict)
        ) if isinstance(user, list) else str(user)

        self.calls.append(
            RecordedCall(
                model=request.get("model", ""),
                schema_name=schema_name,
                system_prompt=system_prompt,
                user_text=user_text,
            )
        )
        index = self._counts.get(schema_name, 0)
        self._counts[schema_name] = index + 1

        if schema_name not in self._responses:
            raise AssertionError(
                f"FakeLLM has no canned response for schema {schema_name!r}; "
                f"known: {sorted(self._responses)}"
            )
        payload = self._responses[schema_name]
        if isinstance(payload, Callable):  # type: ignore[arg-type]
            payload = payload(index, request)
        return _Completion([_Choice(_Message(json.dumps(payload)))])

    def calls_for(self, schema_name: str) -> list[RecordedCall]:
        return [c for c in self.calls if c.schema_name == schema_name]

    def call_count(self, schema_name: str) -> int:
        return self._counts.get(schema_name, 0)


class _Chat:
    def __init__(self, parent: FakeLLM) -> None:
        self.completions = _Completions(parent)


class _Completions:
    def __init__(self, parent: FakeLLM) -> None:
        self._parent = parent

    def create(self, **request) -> _Completion:
        return self._parent._respond(request)


# --------------------------------------------------------------------------
# Canned response builders
# --------------------------------------------------------------------------


def goal(text: str, topic: str = "pH indicator colour change", clear: bool = True) -> dict:
    return {"goal": text, "page_topic": topic, "reads_page_clearly": clear}


def scene(*objects: dict, notes: str = "") -> dict:
    return {"objects": list(objects), "scene_notes": notes}


def obj(name: str, category: str = "container", has_handle: bool = False,
        contents: str = "unknown", relevant: bool = True) -> dict:
    return {
        "name": name,
        "category": category,
        "has_handle": has_handle,
        "contents": contents,
        "relevant_to_goal": relevant,
    }


def subtask(description: str, rationale: str = "", container: str = "clear cup") -> dict:
    return {
        "description": description,
        "rationale": rationale or f"needed to {description}",
        "target_container": container,
    }


def workaround(constraint: str, literal: str, adopted: str,
               differs: bool = False, note: str = "") -> dict:
    return {
        "constraint": constraint,
        "literal_instruction": literal,
        "adopted_approach": adopted,
        "outcome_differs": differs,
        "note": note,
    }


def plan(subtasks: list[dict], *, modality: str = "color",
         expected: str = "the solution turns blue",
         observation_target: str = "clear cup 1",
         workarounds: list[dict] | None = None, feasible: bool = True,
         reason: str = "") -> dict:
    return {
        "feasible": feasible,
        "infeasible_reason": reason,
        "subtasks": subtasks,
        "observation_target": observation_target,
        "expected_observation": expected,
        "verification_modality": modality,
        "capability_workarounds": workarounds or [],
    }


def corrective_plan(subtasks: list[dict], *, diagnosis: str = "not enough reagent was added",
                    **kwargs) -> dict:
    payload = plan(subtasks, **kwargs)
    payload["diagnosis"] = diagnosis
    return payload


def affordance(action: str, *, tool: str = "measuring scoop",
               target: str = "clear cup", location: str = "Original Position of clear cup 1",
               needs_pickup: bool = False, feasible: bool = True, reason: str = "") -> dict:
    return {
        "feasible": feasible,
        "infeasible_reason": reason,
        "action": action,
        "tool": tool,
        "target_object": target,
        "location": location,
        "needs_pickup_first": needs_pickup,
        "rationale": f"{action} is the vocabulary action for this sub-task",
    }


def _primitive(kind: str, *, location: str = "", dx: float = 0.0, dy: float = 0.0,
               dz: float = 0.0, gripper: int = 0, tx: float = 0.0, ty: float = 0.0,
               tz: float = 0.0, explanation: str = "") -> dict:
    return {
        "kind": kind,
        "location": location,
        "delta_x_cm": dx,
        "delta_y_cm": dy,
        "delta_z_cm": dz,
        "gripper": gripper,
        "theta_x_deg": tx,
        "theta_y_deg": ty,
        "theta_z_deg": tz,
        "explanation": explanation,
    }


def low_level_step(action: str, *, location: str = "Original Position of clear cup 1",
                   feasible: bool = True, reason: str = "") -> dict:
    """A plausible GOTO/GRASP/TILT sequence, standing in for the model's output."""
    return {
        "feasible": feasible,
        "infeasible_reason": reason,
        "action": action,
        "rationale": f"realising {action} with go-to/grasp/tilt primitives",
        "primitives": [
            _primitive("GOTO", location=location, dz=15.0, explanation="approach from above"),
            _primitive("GRASP", gripper=1, explanation="close the gripper"),
            _primitive("TILT", ty=-25.0, explanation="tip toward the target"),
            _primitive("GOTO", location=location, dz=10.0, explanation="retreat"),
        ],
    }


#: A complete, coherent response set for Experiment A, used by several tests.
def experiment_a_responses(
    *,
    corrective_subtasks: list[dict] | None = None,
    modality: str = "color",
    workarounds: list[dict] | None = None,
) -> dict[str, Any]:
    base = [
        subtask("pick up the measuring scoop", container="none"),
        subtask("scoop one measure of sodium bicarbonate", container="clear cup 1"),
        subtask("add the scooped powder to the indicator cup", container="clear cup 1"),
        subtask("stir the indicator cup until the powder dissolves", container="clear cup 1"),
        subtask("put the measuring scoop back down", container="none"),
    ]
    corrective = corrective_subtasks or [
        subtask("pick up the measuring scoop", container="none"),
        subtask("scoop a second, larger measure of sodium bicarbonate", container="clear cup 1"),
        subtask("add the scooped powder to the indicator cup", container="clear cup 1"),
        subtask("stir the indicator cup until the powder dissolves", container="clear cup 1"),
        subtask("put the measuring scoop back down", container="none"),
    ]
    actions = ["PICKUP", "SCOOP", "POUR", "STIR", "PLACE"]

    def affordance_response(index: int, request: dict) -> dict:
        return affordance(actions[index % len(actions)])

    def step_response(index: int, request: dict) -> dict:
        text = request["messages"][1]["content"][0]["text"]
        action = text.split("Action:", 1)[1].split("\n", 1)[0].strip()
        return low_level_step(action)

    return {
        "extracted_goal": goal("turn the cabbage indicator solution basic"),
        "scene_description": scene(
            obj("clear cup 1", "container", False, "dark purple indicator solution"),
            obj("sodium bicarbonate", "reagent", False, "white powder"),
            obj("measuring scoop", "tool", True, "empty"),
            obj("pipette", "tool", True, "empty", relevant=False),
        ),
        "subtask_plan": plan(base, modality=modality, workarounds=workarounds),
        "corrective_plan": corrective_plan(corrective, modality=modality),
        "affordance_decision": affordance_response,
        "low_level_step": step_response,
    }
