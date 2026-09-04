"""Skill execution (CLAUDE.md architecture item 10).

######################################################################
##                                                                  ##
##   THE ROBOT DOES NOT ACTUALLY MOVE. THIS IS A FAKE EXECUTOR.     ##
##                                                                  ##
##   Every action in the fixed vocabulary is a STUB that reports     ##
##   success or failure according to FAKE_SUCCESS_RATE below. No     ##
##   real motion, no real force, no real spillage.                   ##
##                                                                  ##
##   This is deliberate for this pass. The point is to validate the  ##
##   planning / verification / logging loop, not motion. A hard      ##
##   NotImplementedError stub would block every run and validate     ##
##   nothing.                                                        ##
##                                                                  ##
##   DO NOT mistake a green run here for a working robot.            ##
##                                                                  ##
######################################################################

The LLM-generated motion primitives ARE passed through to the arm interface and
recorded, so the trial log contains the real planner output even though the arm
that receives it is a mock. Nothing in this file hand-authors a trajectory: it
replays what the Step Planner produced.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

from agents.action_vocabulary import Action, MotionPrimitive, PlannedStep
from hardware.interfaces import ArmInterface

#: Probability that a stubbed skill reports success. 1.0 = always succeed.
#: Override with PLATO_FAKE_SUCCESS_RATE to exercise execution-failure handling.
#: THIS IS FAKE. It is not a measured success rate and must never be reported as
#: one in the paper's results.
FAKE_SUCCESS_RATE: float = float(os.environ.get("PLATO_FAKE_SUCCESS_RATE", "1.0"))

ENV_FAKE_SUCCESS_RATE = "PLATO_FAKE_SUCCESS_RATE"

#: Nominal wall-clock cost per action, seconds. Used only to make the logged
#: time-to-completion column non-degenerate in mock runs; not a measurement.
FAKE_ACTION_SECONDS: dict[Action, float] = {
    Action.PICKUP: 4.0,
    Action.POUR: 6.0,
    Action.SCOOP: 7.0,
    Action.PIPETTE_DISPENSE: 5.0,
    Action.STIR: 8.0,
    Action.PLACE: 3.0,
}


@dataclass
class ExecutionResult:
    """Outcome of executing one planned step."""

    action: Action
    success: bool
    reason: str
    primitives_issued: int
    simulated_seconds: float
    is_fake: bool = True
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "action": self.action.value,
            "success": self.success,
            "reason": self.reason,
            "primitives_issued": self.primitives_issued,
            "simulated_seconds": round(self.simulated_seconds, 2),
            "is_fake": self.is_fake,
            "detail": self.detail,
        }


class SkillExecutor:
    """Executes a :class:`PlannedStep` against an arm.

    The motion primitives are genuinely issued to the arm interface. Whether the
    skill is deemed to have *worked* is faked -- see the banner at the top of
    this module.
    """

    def __init__(
        self,
        arm: ArmInterface,
        *,
        fake_success_rate: float = FAKE_SUCCESS_RATE,
        seed: int | None = 0,
        realtime: bool = False,
    ) -> None:
        if not 0.0 <= fake_success_rate <= 1.0:
            raise ValueError("fake_success_rate must be between 0.0 and 1.0")
        self.arm = arm
        self.fake_success_rate = float(fake_success_rate)
        self._rng = random.Random(seed)
        self.realtime = realtime

    def _issue(self, primitive: MotionPrimitive) -> None:
        """Pass one LLM-generated primitive through to the arm."""
        if primitive.kind == "GOTO":
            self.arm.goto_delta(primitive.location or "Home Pose", primitive.delta_cm or (0, 0, 0))
        elif primitive.kind == "GRASP":
            self.arm.set_gripper(int(primitive.gripper or 0))
        elif primitive.kind == "TILT":
            self.arm.tilt(primitive.tilt_deg or (0.0, 0.0, 0.0))
        else:  # pragma: no cover - guarded upstream by the primitive schema
            raise ValueError(f"unknown primitive kind {primitive.kind!r}")

    def execute(self, step: PlannedStep) -> ExecutionResult:
        """Issue ``step``'s primitives, then fake a success/failure verdict."""
        for primitive in step.primitives:
            self._issue(primitive)

        # Keep the mock arm's held-object bookkeeping honest, so single-gripper
        # violations show up in the log rather than being silently absorbed.
        acquire = getattr(self.arm, "acquire", None)
        release = getattr(self.arm, "release", None)
        if step.action is Action.PICKUP and callable(acquire):
            acquire(step.tool or step.target_object or "unknown object")
        elif step.action is Action.PLACE and callable(release):
            release()

        seconds = FAKE_ACTION_SECONDS.get(step.action, 5.0)
        if self.realtime:  # pragma: no cover - only for demos on a real clock
            time.sleep(min(seconds, 0.05))

        succeeded = self._rng.random() < self.fake_success_rate
        return ExecutionResult(
            action=step.action,
            success=succeeded,
            reason=(
                f"FAKE execution of {step.action.value}: {len(step.primitives)} primitives "
                f"issued to {type(self.arm).__name__}; stub reported success "
                f"(rate {self.fake_success_rate:.2f})"
                if succeeded
                else f"FAKE execution of {step.action.value}: stub reported failure "
                f"(rate {self.fake_success_rate:.2f}). No real manipulation was attempted."
            ),
            primitives_issued=len(step.primitives),
            simulated_seconds=seconds,
            detail={
                "tool": step.tool,
                "target_object": step.target_object,
                "location": step.location,
            },
        )


def fake_execution_banner() -> str:
    """One-line warning to print at the top of every run, so nobody forgets."""
    return (
        f"[FAKE EXECUTION] Skills are stubs with FAKE_SUCCESS_RATE={FAKE_SUCCESS_RATE:.2f}. "
        "The arm does not move. Results measure the planning/verification loop only."
    )
