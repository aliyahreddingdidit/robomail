"""Logging / Data Agent (CLAUDE.md architecture item 8). NEW.

Writes one JSON-lines record per trial. This is the paper's Results-section
dataset, so the schema is a real deliverable, not debug output. Per the research
plan's cross-cutting evaluation protocol it captures, for every trial:

  * the booklet page image reference and the goal extracted from it
  * the sub-task plan the High-Level Planner generated
  * per-step action, the primitives issued, and success/failure
  * the verification signal, its metrics and its reason string
  * the replanning iteration count
  * wall-clock time to completion
  * any capability-constraint workarounds the robot profile forced (item 6)
  * which steps could NOT be run, and why (e.g. no API key configured)

Every record also carries ``execution_is_fake``, because in this pass the skills
are stubs (item 10). A results table built from these records must not be
presented as manipulation success rates.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from config import models

SCHEMA_VERSION = "1.0.0"
DEFAULT_LOG_DIR = Path("data/logs")


@dataclass
class StepRecord:
    """One executed (or attempted) sub-task."""

    subtask_index: int
    subtask: str
    attempt: int
    action: str | None = None
    tool: str | None = None
    target_object: str | None = None
    location: str | None = None
    primitives: list[dict] = field(default_factory=list)
    success: bool = False
    failure_kind: str | None = None  # "planning" | "execution" | "verification" | "blocked"
    reason: str = ""
    simulated_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "subtask_index": self.subtask_index,
            "subtask": self.subtask,
            "attempt": self.attempt,
            "action": self.action,
            "tool": self.tool,
            "target_object": self.target_object,
            "location": self.location,
            "primitive_count": len(self.primitives),
            "primitives": self.primitives,
            "success": self.success,
            "failure_kind": self.failure_kind,
            "reason": self.reason,
            "simulated_seconds": round(self.simulated_seconds, 2),
        }


@dataclass
class TrialRecord:
    """One complete trial: goal in, verified outcome out."""

    trial_id: str
    experiment: str
    condition: str
    booklet_page_image: str | None = None

    extracted_goal: str | None = None
    goal_page_topic: str | None = None

    scene_objects: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)

    verification: dict | None = None
    verification_history: list[dict] = field(default_factory=list)
    replanning_iterations: int = 0
    max_replanning_iterations: int = 0

    capability_workarounds: list[dict] = field(default_factory=list)

    end_to_end_success: bool = False
    outcome: str = "incomplete"  # "success" | "failed" | "blocked" | "incomplete"
    failure_kind: str | None = None
    blocked_steps: list[dict] = field(default_factory=list)

    execution_is_fake: bool = True
    fake_success_rate: float | None = None
    hardware: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)

    started_at: float = field(default_factory=time.time)
    wall_clock_seconds: float = 0.0
    simulated_seconds: float = 0.0
    schema_version: str = SCHEMA_VERSION
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "experiment": self.experiment,
            "condition": self.condition,
            "started_at_unix": round(self.started_at, 3),
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.started_at)),
            "booklet_page_image": self.booklet_page_image,
            "extracted_goal": self.extracted_goal,
            "goal_page_topic": self.goal_page_topic,
            "scene_objects": self.scene_objects,
            "plans": self.plans,
            "steps": self.steps,
            "verification": self.verification,
            "verification_history": self.verification_history,
            "replanning_iterations": self.replanning_iterations,
            "max_replanning_iterations": self.max_replanning_iterations,
            "capability_workarounds": self.capability_workarounds,
            "end_to_end_success": self.end_to_end_success,
            "outcome": self.outcome,
            "failure_kind": self.failure_kind,
            "blocked_steps": self.blocked_steps,
            "execution_is_fake": self.execution_is_fake,
            "fake_success_rate": self.fake_success_rate,
            "hardware": self.hardware,
            "model_config": self.model_config,
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "simulated_seconds": round(self.simulated_seconds, 2),
            "notes": self.notes,
        }


class TrialLogger:
    """Accumulates one trial's record and appends it to a JSONL file."""

    def __init__(
        self,
        experiment: str,
        *,
        condition: str = "fixed_scene",
        log_dir: Path | str = DEFAULT_LOG_DIR,
        trial_id: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{experiment}.jsonl"
        self.record = TrialRecord(
            trial_id=trial_id or uuid.uuid4().hex[:12],
            experiment=experiment,
            condition=condition,
            model_config=models.describe(),
        )
        self._start = time.perf_counter()

    # -- accumulation ------------------------------------------------------
    def set_goal(self, image: str | None, goal: str | None, page_topic: str | None = None) -> None:
        self.record.booklet_page_image = image
        self.record.extracted_goal = goal
        self.record.goal_page_topic = page_topic

    def set_scene(self, objects: list[dict]) -> None:
        self.record.scene_objects = objects

    def add_plan(self, plan: dict) -> None:
        self.record.plans.append(plan)
        for workaround in plan.get("capability_workarounds", []) or []:
            entry = dict(workaround)
            entry["plan_index"] = len(self.record.plans) - 1
            self.record.capability_workarounds.append(entry)

    def add_step(self, step: StepRecord) -> None:
        self.record.steps.append(step.as_dict())
        self.record.simulated_seconds += step.simulated_seconds

    def add_verification(self, result: dict, *, attempt: int) -> None:
        entry = dict(result)
        entry["attempt"] = attempt
        self.record.verification_history.append(entry)
        self.record.verification = entry

    def add_blocked(self, stage: str, reason: str) -> None:
        """Record a step that could not be run at all (e.g. no API key).

        CLAUDE.md is explicit that a missing key must report exactly which steps
        could not be verified, rather than silently skipping them.
        """
        self.record.blocked_steps.append({"stage": stage, "reason": reason})

    def set_hardware(self, hardware: dict, *, execution_is_fake: bool,
                     fake_success_rate: float | None) -> None:
        self.record.hardware = hardware
        self.record.execution_is_fake = execution_is_fake
        self.record.fake_success_rate = fake_success_rate

    def set_outcome(
        self,
        *,
        success: bool,
        outcome: str,
        replanning_iterations: int,
        max_replanning_iterations: int,
        failure_kind: str | None = None,
        notes: str = "",
    ) -> None:
        self.record.end_to_end_success = success
        self.record.outcome = outcome
        self.record.replanning_iterations = replanning_iterations
        self.record.max_replanning_iterations = max_replanning_iterations
        self.record.failure_kind = failure_kind
        if notes:
            self.record.notes = notes

    # -- output ------------------------------------------------------------
    def finalise(self) -> dict:
        """Stamp the wall clock, append the record to disk, and return it."""
        self.record.wall_clock_seconds = time.perf_counter() - self._start
        payload = self.record.as_dict()
        payload["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload


def read_trials(path: Path | str) -> Iterator[dict[str, Any]]:
    """Read back a JSONL trial log, one record at a time."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarise(path: Path | str) -> dict:
    """Aggregate a trial log into the columns the Results section needs."""
    trials = list(read_trials(path))
    if not trials:
        return {"trials": 0}
    successes = sum(1 for t in trials if t.get("end_to_end_success"))
    replans = [t.get("replanning_iterations", 0) for t in trials]
    times = [t.get("wall_clock_seconds", 0.0) for t in trials]
    by_condition: dict[str, dict] = {}
    for trial in trials:
        bucket = by_condition.setdefault(
            trial.get("condition", "unknown"), {"trials": 0, "successes": 0}
        )
        bucket["trials"] += 1
        bucket["successes"] += int(bool(trial.get("end_to_end_success")))
    for bucket in by_condition.values():
        bucket["success_rate"] = round(bucket["successes"] / bucket["trials"], 4)
    return {
        "trials": len(trials),
        "successes": successes,
        "success_rate": round(successes / len(trials), 4),
        "mean_replanning_iterations": round(sum(replans) / len(replans), 3),
        "max_replanning_iterations": max(replans),
        "mean_wall_clock_seconds": round(sum(times) / len(times), 3),
        "any_execution_is_fake": any(t.get("execution_is_fake") for t in trials),
        "trials_with_blocked_steps": sum(1 for t in trials if t.get("blocked_steps")),
        "trials_with_capability_workarounds": sum(
            1 for t in trials if t.get("capability_workarounds")
        ),
        "by_condition": by_condition,
    }
