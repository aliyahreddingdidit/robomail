"""Closed-loop orchestrator (CLAUDE.md architecture item 9).

Runs one trial end to end, autonomously:

    booklet page photo
      -> Goal Extraction Agent      (item 2)  one goal string
      -> Scene Understanding Agent  (item 1)  grounded objects
      -> High-Level Planner Agent   (item 3)  ordered sub-task plan
      -> for each sub-task:
           Affordance Agent         (item 4)  action + tool + grasp
           Step Planner             (item 5)  LLM-generated motion primitives
           Skill executor           (item 10) FAKE pass/fail, arm records primitives
      -> Verification Agent         (item 7)  real CV on real frames
      -> on failure: Planner replans (item 9), up to MAX_REPLAN_ATTEMPTS
      -> Logging Agent              (item 8)  one JSONL trial record

There is no human input at any point. There is no approval gate. A run that
cannot proceed -- no API key, an infeasible plan, the retry cap hit -- reports a
final failure with a reason, and never crashes out of the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from agents import goal_extraction, high_level_planner, scene_understanding, step_planner
from agents.action_vocabulary import InfeasibleStep, PlannedStep, validate_sequence
from agents.affordance import resolve_affordance
from agents.high_level_planner import Plan, SubTask
from agents.llm_client import MissingAPIKeyError
from agents.logging_agent import StepRecord, TrialLogger
from agents.verification import VerificationAgent, VerificationResult
from config.robot_profile import DEFAULT_PROFILE, RobotProfile
from hardware.factory import Cell, build_cell
from hardware.mock import CUP_ROI, MockCamera
from skills.executor import FAKE_SUCCESS_RATE, SkillExecutor, fake_execution_banner

#: Cap on corrective replanning attempts per sub-task (CLAUDE.md item 9).
MAX_REPLAN_ATTEMPTS = 3

#: Frames captured for a motion-modality observation window.
MOTION_WINDOW_FRAMES = 8


@dataclass
class TrialOutcome:
    """What one trial produced, for callers that want it in memory."""

    success: bool
    outcome: str
    goal: str | None
    reason: str
    replanning_iterations: int
    record: dict = field(default_factory=dict)
    verification: VerificationResult | None = None


class Orchestrator:
    """Runs one trial of one experiment, closed-loop and unattended."""

    def __init__(
        self,
        *,
        cell: Cell | None = None,
        profile: RobotProfile = DEFAULT_PROFILE,
        verification_agent: VerificationAgent | None = None,
        max_replan_attempts: int = MAX_REPLAN_ATTEMPTS,
        verification_roi: tuple[int, int, int, int] | None = CUP_ROI,
        fake_success_rate: float = FAKE_SUCCESS_RATE,
        executor_seed: int | None = 0,
        llm_client_override=None,
        log_dir: str | Path = "data/logs",
        verbose: bool = True,
    ) -> None:
        self.cell = cell or build_cell()
        self.profile = profile
        self.verifier = verification_agent or VerificationAgent()
        self.max_replan_attempts = int(max_replan_attempts)
        self.verification_roi = verification_roi
        self.fake_success_rate = fake_success_rate
        self.executor = SkillExecutor(
            self.cell.arm, fake_success_rate=fake_success_rate, seed=executor_seed
        )
        self.llm = llm_client_override
        self.log_dir = log_dir
        self.verbose = verbose

    # -- small helpers -----------------------------------------------------
    def _say(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _capture(self, modality: str) -> list:
        """Grab the observation the Verification Agent needs for ``modality``."""
        if modality == "motion":
            return list(self.cell.camera.capture_window(MOTION_WINDOW_FRAMES))
        return [self.cell.camera.capture()]

    # -- the run -----------------------------------------------------------
    def run_trial(
        self,
        booklet_page_image: str | Path,
        *,
        experiment: str,
        condition: str = "fixed_scene",
        workspace_image: str | Path | None = None,
        expect_change: bool = True,
        expected_ph_class: str | None = None,
        expected_labels: Sequence[str] | None = None,
        on_step_executed=None,
    ) -> TrialOutcome:
        """Run one trial from a booklet page photo to a verified outcome.

        ``on_step_executed`` is an optional callback ``(planned_step, result)``
        the mock cell uses to advance its BenchState, so the frames the
        Verification Agent analyses reflect what the run actually did. It has no
        effect on real hardware.
        """
        self._say(fake_execution_banner())
        logger = TrialLogger(experiment, condition=condition, log_dir=self.log_dir)
        logger.set_hardware(
            self.cell.as_dict(),
            execution_is_fake=True,
            fake_success_rate=self.fake_success_rate,
        )

        # --- item 2: goal extraction -------------------------------------
        try:
            extracted = goal_extraction.extract_goal(booklet_page_image, client=self.llm)
        except MissingAPIKeyError as exc:
            return self._blocked(logger, "goal_extraction", str(exc), booklet_page_image)
        except goal_extraction.GoalValidationError as exc:
            logger.set_goal(str(booklet_page_image), None, None)
            logger.add_blocked("goal_extraction", str(exc))
            logger.set_outcome(
                success=False, outcome="failed", replanning_iterations=0,
                max_replanning_iterations=self.max_replan_attempts,
                failure_kind="planning",
                notes="Goal Extraction Agent produced a plan instead of a goal.",
            )
            return TrialOutcome(False, "failed", None, str(exc), 0, logger.finalise())

        logger.set_goal(str(booklet_page_image), extracted.goal, extracted.page_topic)
        self._say(f"[goal] {extracted.goal}")
        if not extracted.reads_page_clearly:
            logger.add_blocked(
                "goal_extraction",
                "agent reported the booklet page photo was not clearly legible; "
                "the goal below may be unreliable",
            )

        # --- item 1: scene understanding ---------------------------------
        scene_image = workspace_image
        if scene_image is None:
            scene_image = self._write_scene_frame(logger)
        try:
            scene = scene_understanding.comprehend_scene(
                scene_image, extracted.goal, profile=self.profile, client=self.llm
            )
        except MissingAPIKeyError as exc:
            return self._blocked(logger, "scene_understanding", str(exc), booklet_page_image,
                                 goal=extracted.goal)
        logger.set_scene([o.as_dict() for o in scene.objects])
        self._say(f"[scene] {len(scene.objects)} objects: {', '.join(scene.names) or '(none)'}")

        # --- item 3: high-level planning ---------------------------------
        try:
            plan = high_level_planner.plan_goal(
                extracted.goal, scene, profile=self.profile, client=self.llm
            )
        except MissingAPIKeyError as exc:
            return self._blocked(logger, "high_level_planner", str(exc), booklet_page_image,
                                 goal=extracted.goal)
        logger.add_plan(plan.as_dict())
        self._say(f"[plan] {len(plan.subtasks)} sub-tasks, modality={plan.verification_modality}")
        for workaround in plan.capability_workarounds:
            self._say(f"[capability] {workaround.constraint} -> {workaround.adopted_approach}")

        if not plan.feasible:
            logger.set_outcome(
                success=False, outcome="failed", replanning_iterations=0,
                max_replanning_iterations=self.max_replan_attempts,
                failure_kind="planning",
                notes=f"planner reported the goal infeasible: {plan.infeasible_reason}",
            )
            return TrialOutcome(False, "failed", extracted.goal,
                                plan.infeasible_reason, 0, logger.finalise())

        # --- items 4/5/10 + 7 + 9: execute, verify, replan ----------------
        before_frames = self._capture(plan.verification_modality)
        attempt = 0
        replans = 0
        verification: VerificationResult | None = None
        active_plan = plan
        completed: list[SubTask] = []
        last_reason = ""

        last_failure_kind: str | None = None
        while attempt <= self.max_replan_attempts:
            executed_ok, failed_subtask, blocker, exec_reason, exec_kind = self._execute_plan(
                active_plan, scene, logger, attempt, on_step_executed, completed
            )
            if blocker is not None:
                return self._blocked(logger, blocker[0], blocker[1], booklet_page_image,
                                     goal=extracted.goal, replans=replans)

            after_frames = self._capture(active_plan.verification_modality)
            frames = ([before_frames[0]] + list(after_frames)
                      if active_plan.verification_modality != "motion" else list(after_frames))

            try:
                verification = self.verifier.verify(
                    active_plan.verification_modality,
                    frames,
                    roi=self.verification_roi,
                    expect_change=expect_change,
                    expected_ph_class=expected_ph_class,
                    expected_labels=expected_labels,
                )
            except ValueError as exc:
                verification = VerificationResult(
                    success=False,
                    reason=f"verification could not run: {exc}",
                    modality=active_plan.verification_modality,
                )
            logger.add_verification(verification.as_dict(), attempt=attempt)
            if executed_ok:
                last_reason = verification.reason
                last_failure_kind = None if verification.success else "verification"
            else:
                # Execution stopped early. Whatever the camera shows now reflects
                # a partially-executed plan, so the verification verdict is not
                # the thing to act on or to report.
                last_reason = exec_reason or "a sub-task could not be executed"
                last_failure_kind = exec_kind or "execution"
            self._say(
                f"[verify:{attempt}] {'PASS' if verification.success else 'FAIL'} "
                f"({verification.modality}) {verification.reason}"
            )

            if verification.success and executed_ok:
                logger.set_outcome(
                    success=True, outcome="success", replanning_iterations=replans,
                    max_replanning_iterations=self.max_replan_attempts,
                )
                return TrialOutcome(True, "success", extracted.goal, verification.reason,
                                    replans, logger.finalise(), verification)

            if attempt >= self.max_replan_attempts:
                break

            # --- item 9: autonomous corrective replan --------------------
            attempt += 1
            replans += 1
            self._say(f"[replan] attempt {attempt}/{self.max_replan_attempts}")
            try:
                active_plan = high_level_planner.replan(
                    extracted.goal,
                    scene,
                    active_plan,
                    failed_subtask=failed_subtask,
                    completed_subtasks=completed,
                    verification_reason=last_reason,
                    attempt=attempt,
                    max_attempts=self.max_replan_attempts,
                    profile=self.profile,
                    client=self.llm,
                )
            except MissingAPIKeyError as exc:
                return self._blocked(logger, "replanning", str(exc), booklet_page_image,
                                     goal=extracted.goal, replans=replans)
            logger.add_plan(active_plan.as_dict())
            if not active_plan.feasible:
                logger.set_outcome(
                    success=False, outcome="failed", replanning_iterations=replans,
                    max_replanning_iterations=self.max_replan_attempts,
                    failure_kind="planning",
                    notes=f"planner gave up during correction: {active_plan.infeasible_reason}",
                )
                return TrialOutcome(False, "failed", extracted.goal,
                                    active_plan.infeasible_reason, replans,
                                    logger.finalise(), verification)

        # Retry cap hit: a reported failure, not a crash.
        logger.set_outcome(
            success=False, outcome="failed", replanning_iterations=replans,
            max_replanning_iterations=self.max_replan_attempts,
            failure_kind=last_failure_kind or "verification",
            notes=(
                f"exhausted {self.max_replan_attempts} corrective attempts without a "
                f"verified outcome; final failure was "
                f"{last_failure_kind or 'verification'}: {last_reason}"
            ),
        )
        return TrialOutcome(False, "failed", extracted.goal, last_reason, replans,
                            logger.finalise(), verification)

    # -- internals ---------------------------------------------------------
    def _execute_plan(self, plan: Plan, scene, logger: TrialLogger, attempt: int,
                      on_step_executed, completed: list[SubTask]):
        """Run every sub-task of ``plan``.

        Returns ``(all_ok, failed_subtask, blocker, reason, failure_kind)``. The
        reason and kind matter: when execution stops early, that -- not the
        verification verdict -- is what should be reported and fed back into
        replanning.
        """
        planned_steps: list[PlannedStep] = []
        all_ok = True
        failed: SubTask | None = None
        failure_reason = ""
        failure_kind: str | None = None

        for subtask in plan.subtasks:
            record = StepRecord(subtask_index=subtask.index, subtask=subtask.description,
                                attempt=attempt)
            try:
                affordance = resolve_affordance(
                    subtask, scene, grasp_provider=self.cell.grasp_provider,
                    profile=self.profile, client=self.llm,
                )
                step = step_planner.plan_step(
                    affordance, previous_steps=planned_steps,
                    profile=self.profile, client=self.llm,
                )
            except MissingAPIKeyError as exc:
                return False, subtask, ("step_planning", str(exc)), str(exc), "blocked"
            except InfeasibleStep as exc:
                record.success = False
                record.failure_kind = "planning"
                record.reason = exc.reason
                logger.add_step(record)
                self._say(f"[step {subtask.index}] INFEASIBLE: {exc.reason}")
                return (
                    False, subtask, None,
                    f"sub-task {subtask.index} ({subtask.description!r}) could not be "
                    f"expressed with the available actions: {exc.reason}",
                    "planning",
                )

            result = self.executor.execute(step)
            planned_steps.append(step)

            record.action = step.action.value
            record.tool = step.tool
            record.target_object = step.target_object
            record.location = step.location
            record.primitives = [p.as_dict() for p in step.primitives]
            record.success = result.success
            record.failure_kind = None if result.success else "execution"
            record.reason = result.reason
            record.simulated_seconds = result.simulated_seconds
            logger.add_step(record)
            self._say(
                f"[step {subtask.index}] {step.action.value} "
                f"({len(step.primitives)} primitives) -> "
                f"{'ok' if result.success else 'FAILED'}"
            )

            if on_step_executed is not None:
                on_step_executed(step, result)

            if result.success:
                completed.append(subtask)
            else:
                all_ok = False
                failed = subtask
                failure_reason = (
                    f"sub-task {subtask.index} ({subtask.description!r}) failed during "
                    f"execution: {result.reason}"
                )
                failure_kind = "execution"
                break

        warnings = validate_sequence([s.action for s in planned_steps])
        for warning in warnings:
            logger.add_blocked("single_gripper_check", warning)
            self._say(f"[constraint] {warning}")

        return all_ok, failed, None, failure_reason, failure_kind

    def _write_scene_frame(self, logger: TrialLogger) -> str:
        """Capture the workspace and persist it, so the trial log references a real file."""
        import cv2

        frame = self.cell.camera.capture()
        path = Path(logger.log_dir) / "frames"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{logger.record.trial_id}_scene.png"
        cv2.imwrite(str(out), frame)
        return str(out)

    def _blocked(self, logger: TrialLogger, stage: str, reason: str,
                 image, *, goal: str | None = None, replans: int = 0) -> TrialOutcome:
        """Record a stage that could not run, naming it explicitly."""
        logger.add_blocked(stage, reason)
        if goal is None:
            logger.set_goal(str(image), None, None)
        logger.set_outcome(
            success=False, outcome="blocked", replanning_iterations=replans,
            max_replanning_iterations=self.max_replan_attempts, failure_kind="blocked",
            notes=f"pipeline blocked at stage {stage!r}; downstream stages were not run",
        )
        self._say(f"[blocked] {stage}: {reason}")
        return TrialOutcome(False, "blocked", goal, reason, replans, logger.finalise())


# --------------------------------------------------------------------------
# Bench-state coupling for mock runs
# --------------------------------------------------------------------------


def bench_state_updater(cell: Cell, *, reagent_effect: str = "basic",
                        per_step_progress: float = 1.0):
    """Callback that advances the mock bench as (fake) skills execute.

    This is scene simulation for the mock camera, not a scripted robot motion:
    it decides what the *world* looks like after a reagent is added, so the real
    OpenCV detectors have real pixels to judge. On real hardware the world
    updates itself and this is unused.

    ``reagent_effect`` picks what the bench does when a reagent-transfer action
    succeeds: "basic"/"acidic" shift the indicator colour, "fizz" starts bubbles,
    "expand" raises the fill level, "none" leaves the bench unchanged (which is
    how a forced-failure trial is built, and also how Experiment D behaves).
    """
    from agents.action_vocabulary import Action

    state = cell.bench_state
    transfer_actions = {Action.SCOOP, Action.POUR, Action.PIPETTE_DISPENSE}

    def update(step: PlannedStep, result) -> None:
        if state is None or not result.success:
            return
        if step.action not in transfer_actions:
            return
        if reagent_effect == "basic":
            state.color = "blue" if state.color == "purple" else "green"
        elif reagent_effect == "acidic":
            state.color = "red" if state.color == "purple" else "pink"
        elif reagent_effect == "fizz":
            state.bubbles = 55
            state.frozen_bubbles = False
        elif reagent_effect == "expand":
            state.fill_level = min(1.0, state.fill_level + 0.5 * per_step_progress)
        # "none": deliberately do nothing -- the reagent had no effect.

    return update


def make_mock_orchestrator(**kwargs) -> Orchestrator:
    """Convenience: a fully mock cell wired to a fresh BenchState."""
    cell = build_cell("mock")
    assert isinstance(cell.camera, MockCamera)
    return Orchestrator(cell=cell, **kwargs)
