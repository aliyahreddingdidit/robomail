"""Tests for the closed loop, the fixed vocabulary, the robot profile and logging.

Covers CLAUDE.md items 5, 6, 8, 9, 10 and 11. The LLM provider is replaced by a
transport double (``tests.fake_llm``) exactly as the arm is replaced by
MockFrankaArm; the verification that decides pass/fail is the real OpenCV code
looking at real rendered pixels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.action_vocabulary import (
    ACTION_NAMES,
    Action,
    InfeasibleStep,
    parse_action,
    validate_sequence,
    vocabulary_prompt_block,
)
from agents.high_level_planner import SubTask
from agents.logging_agent import TrialLogger, read_trials, summarise
from config.robot_profile import DEFAULT_PROFILE, CapabilityWorkaround, RobotProfile
from hardware.factory import build_cell, resolve_mode
from hardware.mock import CUP_ROI, BenchState, MockCamera, MockFrankaArm
from orchestrator import Orchestrator, bench_state_updater
from skills.executor import SkillExecutor
from tests import fake_llm
from tests.fake_llm import FakeLLM, experiment_a_responses

FIXTURES = Path(__file__).parent / "fixtures"
PAGE = FIXTURES / "real_booklet_page_12.png"


def make_orchestrator(tmp_path, *, responses=None, bench=None, fake_success_rate=1.0):
    cell = build_cell("mock", bench_state=bench or BenchState(color="purple", fill_level=0.45))
    llm = FakeLLM(responses or experiment_a_responses())
    orchestrator = Orchestrator(
        cell=cell, llm_client_override=llm, log_dir=tmp_path,
        fake_success_rate=fake_success_rate, verbose=False,
    )
    return orchestrator, cell, llm


# ==========================================================================
# Item 5 -- the fixed action vocabulary
# ==========================================================================


def test_vocabulary_is_exactly_the_specified_six_actions():
    assert ACTION_NAMES == ["PICKUP", "POUR", "SCOOP", "PIPETTE_DISPENSE", "STIR", "PLACE"]


def test_vocabulary_prompt_block_declares_the_list_closed():
    block = vocabulary_prompt_block().lower()
    assert "exhaustive and closed" in block
    assert "must not invent" in block
    for name in ACTION_NAMES:
        assert name.lower() in block


def test_parse_action_accepts_reasonable_spellings():
    assert parse_action("pipette-dispense") is Action.PIPETTE_DISPENSE
    assert parse_action("  scoop ") is Action.SCOOP
    assert parse_action(Action.POUR) is Action.POUR


def test_an_invented_action_is_a_reported_failure_not_a_silent_substitution():
    with pytest.raises(InfeasibleStep) as excinfo:
        parse_action("SHAKE_VIGOROUSLY")
    assert "not one of" in excinfo.value.reason


def test_step_planner_refuses_to_substitute_a_different_action(tmp_path):
    """If the Step Planner disagrees with the Affordance Agent, that is reported."""
    from agents.affordance import Affordance
    from agents.step_planner import plan_step

    affordance = Affordance(
        subtask=SubTask(1, "scoop the powder"), action=Action.SCOOP, tool="measuring scoop",
        target_object="sodium bicarbonate", location="Original Position of measuring scoop",
        needs_pickup_first=False, rationale="",
    )
    llm = FakeLLM({"low_level_step": fake_llm.low_level_step("POUR")})
    with pytest.raises(InfeasibleStep, match="refusing to silently substitute"):
        plan_step(affordance, client=llm)


def test_step_planner_reports_an_empty_motion_sequence(tmp_path):
    from agents.affordance import Affordance
    from agents.step_planner import plan_step

    affordance = Affordance(
        subtask=SubTask(1, "stir the cup"), action=Action.STIR, tool="stirring rod",
        target_object="clear cup", location="Original Position of clear cup 1",
        needs_pickup_first=False, rationale="",
    )
    empty = fake_llm.low_level_step("STIR")
    empty["primitives"] = []
    with pytest.raises(InfeasibleStep, match="empty motion sequence"):
        plan_step(affordance, client=FakeLLM({"low_level_step": empty}))


def test_infeasible_affordance_becomes_a_reported_planning_failure(tmp_path):
    """A sub-task the vocabulary cannot express fails loudly and is logged."""
    responses = experiment_a_responses()
    responses["affordance_decision"] = fake_llm.affordance(
        "SCOOP", feasible=False, reason="this needs a centrifuge, which the cell does not have"
    )
    orchestrator, _, _ = make_orchestrator(tmp_path, responses=responses)
    outcome = orchestrator.run_trial(PAGE, experiment="infeasible", expected_ph_class="basic")

    assert not outcome.success
    record = list(read_trials(tmp_path / "infeasible.jsonl"))[0]
    planning_failures = [s for s in record["steps"] if s["failure_kind"] == "planning"]
    assert planning_failures
    assert "centrifuge" in planning_failures[0]["reason"]


# ==========================================================================
# Item 6 -- the robot capability profile
# ==========================================================================


def test_profile_states_the_single_gripper_constraints():
    block = DEFAULT_PROFILE.as_prompt_block()
    assert "ONE arm with ONE parallel-plate gripper" in block
    assert "two spatially-separate actions simultaneously" in block
    assert "two independent forces at once" in block


def test_profile_forbids_a_human_review_gate():
    """CLAUDE.md: no human-in-the-loop step anywhere in this pipeline."""
    block = DEFAULT_PROFILE.as_prompt_block()
    assert "must NOT ask a human" in block
    assert "Decide" in block and "yourself" in block


def test_profile_is_injected_into_planner_and_step_planner(tmp_path):
    orchestrator, _, llm = make_orchestrator(tmp_path)
    orchestrator.run_trial(PAGE, experiment="profile", expected_ph_class="basic")

    for schema in ("subtask_plan", "low_level_step"):
        calls = llm.calls_for(schema)
        assert calls, f"no {schema} call was made"
        assert "ROBOT CAPABILITY PROFILE" in calls[0].system_prompt


def test_profile_asks_the_planner_to_report_workarounds():
    assert "capability_workarounds" in DEFAULT_PROFILE.as_prompt_block()


def test_capability_workarounds_are_logged(tmp_path):
    """The booklet's step 7 literally says 'pour both cups at the same time'."""
    workaround = fake_llm.workaround(
        constraint="cannot perform two spatially-separate actions simultaneously",
        literal="pour the red and blue cups in at the same time",
        adopted="pour the red cup first, then the blue cup",
        differs=True,
        note="serialising delays contact, so the fizz peak may be lower",
    )
    responses = experiment_a_responses(workarounds=[workaround])
    orchestrator, _, _ = make_orchestrator(tmp_path, responses=responses)
    orchestrator.run_trial(PAGE, experiment="workaround", expected_ph_class="basic")

    record = list(read_trials(tmp_path / "workaround.jsonl"))[0]
    assert len(record["capability_workarounds"]) == 1
    logged = record["capability_workarounds"][0]
    assert logged["outcome_differs"] is True
    assert "at the same time" in logged["literal_instruction"]
    assert logged["plan_index"] == 0


def test_profile_is_frozen_so_a_run_cannot_mutate_its_constraints():
    with pytest.raises(Exception):
        DEFAULT_PROFILE.num_grippers = 2  # type: ignore[misc]


def test_a_custom_profile_flows_through():
    profile = RobotProfile(name="two-arm test cell", num_arms=2, num_grippers=2)
    assert "two-arm test cell" in profile.as_prompt_block()


# --------------------------------------------------------------------------
# Single-gripper sequence validation
# --------------------------------------------------------------------------


def test_validate_sequence_accepts_a_well_formed_plan():
    assert validate_sequence(
        [Action.PICKUP, Action.SCOOP, Action.POUR, Action.STIR, Action.PLACE]
    ) == []


def test_validate_sequence_flags_a_second_pickup_without_a_place():
    warnings = validate_sequence([Action.PICKUP, Action.PICKUP])
    assert any("already holding" in w for w in warnings)


def test_validate_sequence_flags_using_a_tool_with_an_empty_gripper():
    warnings = validate_sequence([Action.POUR])
    assert any("gripper is empty" in w for w in warnings)


def test_validate_sequence_flags_placing_nothing():
    assert any("empty gripper" in w for w in validate_sequence([Action.PLACE]))


def test_mock_arm_records_a_single_gripper_violation():
    arm = MockFrankaArm()
    arm.acquire("measuring scoop")
    arm.acquire("pipette")
    assert arm.constraint_violations
    assert "single gripper" in arm.constraint_violations[0]


# ==========================================================================
# Item 10 -- fake skill execution
# ==========================================================================


def test_executor_passes_llm_primitives_through_to_the_arm():
    from agents.action_vocabulary import MotionPrimitive, PlannedStep

    arm = MockFrankaArm()
    step = PlannedStep(
        action=Action.POUR, target_object="clear cup", tool="beaker",
        location="Original Position of clear cup 1",
        primitives=[
            MotionPrimitive("GOTO", location="Original Position of clear cup 1",
                            delta_cm=(0.0, 0.0, 12.0)),
            MotionPrimitive("TILT", tilt_deg=(0.0, -35.0, 0.0)),
            MotionPrimitive("GRASP", gripper=0),
        ],
    )
    result = SkillExecutor(arm, fake_success_rate=1.0).execute(step)

    assert result.success
    assert result.is_fake
    assert [c.kind for c in arm.commands] == ["GOTO", "TILT", "GRASP"]
    assert arm.commands[0].detail["delta_cm"] == [0.0, 0.0, 12.0]
    assert arm.commands[1].detail["theta_deg"] == [0.0, -35.0, 0.0]


def test_executor_marks_every_result_as_fake():
    from agents.action_vocabulary import MotionPrimitive, PlannedStep

    step = PlannedStep(Action.STIR, "cup", "rod", "Original Position of clear cup 1",
                       [MotionPrimitive("GRASP", gripper=1)])
    result = SkillExecutor(MockFrankaArm(), fake_success_rate=1.0).execute(step)
    assert result.is_fake
    assert "FAKE" in result.reason


def test_executor_can_be_made_to_fail_deterministically():
    from agents.action_vocabulary import MotionPrimitive, PlannedStep

    step = PlannedStep(Action.SCOOP, "powder", "scoop", "Original Position of measuring scoop",
                       [MotionPrimitive("GRASP", gripper=1)])
    result = SkillExecutor(MockFrankaArm(), fake_success_rate=0.0).execute(step)
    assert not result.success
    assert "No real manipulation was attempted" in result.reason


def test_executor_rejects_a_nonsensical_success_rate():
    with pytest.raises(ValueError):
        SkillExecutor(MockFrankaArm(), fake_success_rate=1.5)


def test_execution_failure_stops_the_plan_and_is_logged_as_execution(tmp_path):
    orchestrator, _, _ = make_orchestrator(tmp_path, fake_success_rate=0.0)
    outcome = orchestrator.run_trial(PAGE, experiment="exec_fail", expected_ph_class="basic")

    assert not outcome.success
    record = list(read_trials(tmp_path / "exec_fail.jsonl"))[0]
    assert record["execution_is_fake"] is True
    assert record["fake_success_rate"] == 0.0
    assert any(s["failure_kind"] == "execution" for s in record["steps"])


# ==========================================================================
# Item 11 -- mock/real hardware toggle
# ==========================================================================


def test_hardware_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("PLATO_HARDWARE", raising=False)
    assert resolve_mode() == "mock"
    assert build_cell().is_mock


def test_hardware_mode_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("PLATO_HARDWARE", "real")
    assert resolve_mode() == "real"
    assert resolve_mode("mock") == "mock"  # explicit argument wins


def test_an_unknown_hardware_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("PLATO_HARDWARE", "simulator")
    with pytest.raises(ValueError, match="not valid"):
        resolve_mode()


def test_real_hardware_module_imports_without_a_robot():
    """hardware.real must import on a laptop; it only reaches for frankapy on construction."""
    import hardware.real as real

    assert hasattr(real, "RealFrankaArm")
    with pytest.raises(RuntimeError, match="frankapy is not installed"):
        real.RealFrankaArm()


def test_mock_camera_frames_drive_the_real_detectors():
    """The seam: mock hardware supplies pixels, real OpenCV judges them."""
    from agents.verification import VerificationAgent

    state = BenchState(color="purple", fill_level=0.45)
    camera = MockCamera(state)
    before = camera.capture()
    state.color = "blue"
    after = camera.capture()

    result = VerificationAgent().verify("color", [before, after], roi=CUP_ROI,
                                        expected_ph_class="basic")
    assert result.success


# ==========================================================================
# Item 9 -- closed-loop replanning
# ==========================================================================


def test_successful_trial_needs_no_replanning(tmp_path):
    orchestrator, cell, llm = make_orchestrator(tmp_path)
    outcome = orchestrator.run_trial(
        PAGE, experiment="happy", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell, reagent_effect="basic"),
    )
    assert outcome.success
    assert outcome.replanning_iterations == 0
    assert llm.call_count("corrective_plan") == 0


def test_forced_verification_failure_triggers_a_real_replan(tmp_path):
    """The headline closed-loop requirement (CLAUDE.md definition of done).

    ``reagent_effect="none"`` means the bench genuinely does not change, so the
    real colour detector genuinely fails and the planner is genuinely re-invoked.
    """
    orchestrator, cell, llm = make_orchestrator(tmp_path)
    outcome = orchestrator.run_trial(
        PAGE, experiment="forced_failure", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell, reagent_effect="none"),
    )

    assert not outcome.success
    assert outcome.replanning_iterations >= 1
    assert llm.call_count("corrective_plan") >= 1

    record = list(read_trials(tmp_path / "forced_failure.jsonl"))[0]
    assert record["replanning_iterations"] >= 1
    assert len(record["plans"]) >= 2, "a corrective plan must be logged alongside the original"
    assert record["plans"][1]["is_correction"] is True
    assert record["plans"][1]["diagnosis"]
    assert len(record["verification_history"]) >= 2


def test_replanning_is_capped_and_reports_failure_rather_than_crashing(tmp_path):
    orchestrator, cell, llm = make_orchestrator(tmp_path)
    outcome = orchestrator.run_trial(
        PAGE, experiment="capped", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell, reagent_effect="none"),
    )

    assert outcome.outcome == "failed"
    assert outcome.replanning_iterations == orchestrator.max_replan_attempts == 3
    assert llm.call_count("corrective_plan") == 3

    record = list(read_trials(tmp_path / "capped.jsonl"))[0]
    assert record["failure_kind"] == "verification"
    assert "exhausted 3 corrective attempts" in record["notes"]


def test_a_trial_can_recover_on_a_corrective_attempt(tmp_path):
    """A replan that actually works: the first dose does nothing, the second does.

    This is the research plan's headline result -- the planner adds more reagent
    after the Verification Agent reports no colour shift.
    """
    orchestrator, cell, llm = make_orchestrator(tmp_path)
    state = {"plans_run": 0}
    base_update = bench_state_updater(cell, reagent_effect="basic")

    def update(step, result):
        # The first plan's reagent has no effect; corrective doses do.
        if state["plans_run"] > 0:
            base_update(step, result)

    def on_verify_done():
        state["plans_run"] += 1

    original_verify = orchestrator.verifier.verify

    def counting_verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        on_verify_done()
        return result

    orchestrator.verifier.verify = counting_verify  # type: ignore[method-assign]

    outcome = orchestrator.run_trial(
        PAGE, experiment="recovered", expected_ph_class="basic", on_step_executed=update
    )

    assert outcome.success
    assert outcome.replanning_iterations == 1
    record = list(read_trials(tmp_path / "recovered.jsonl"))[0]
    assert record["end_to_end_success"] is True
    assert record["replanning_iterations"] == 1
    assert record["verification_history"][0]["success"] is False
    assert record["verification_history"][-1]["success"] is True


def test_the_replanner_is_given_the_verification_reason(tmp_path):
    """The verification signal must actually reach the planner, not just the log."""
    orchestrator, cell, llm = make_orchestrator(tmp_path)
    orchestrator.run_trial(
        PAGE, experiment="feedback", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell, reagent_effect="none"),
    )
    first_correction = llm.calls_for("corrective_plan")[0]
    assert "Verification Agent verdict: FAILURE" in first_correction.user_text
    assert "Colour has not moved far enough" in first_correction.user_text
    assert "corrective attempt 1 of at most 3" in first_correction.user_text


def test_no_stage_ever_asks_for_human_input(tmp_path):
    """CLAUDE.md: no human-in-the-loop review or approval anywhere in this pipeline."""
    orchestrator, cell, llm = make_orchestrator(tmp_path)
    orchestrator.run_trial(
        PAGE, experiment="autonomy", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell, reagent_effect="basic"),
    )
    for call in llm.calls:
        prompt = call.system_prompt.lower()
        for banned in ("ask the operator", "wait for approval", "request confirmation",
                       "human review", "await sign-off"):
            assert banned not in prompt, f"{call.schema_name} prompt contains {banned!r}"


# ==========================================================================
# Item 8 -- logging schema
# ==========================================================================


REQUIRED_LOG_FIELDS = {
    "booklet_page_image",
    "extracted_goal",
    "plans",
    "steps",
    "verification",
    "verification_history",
    "replanning_iterations",
    "wall_clock_seconds",
    "capability_workarounds",
    "end_to_end_success",
    "execution_is_fake",
}


def test_trial_log_carries_every_field_the_results_section_needs(tmp_path):
    orchestrator, cell, _ = make_orchestrator(tmp_path)
    orchestrator.run_trial(
        PAGE, experiment="schema", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell, reagent_effect="basic"),
    )
    record = list(read_trials(tmp_path / "schema.jsonl"))[0]

    missing = REQUIRED_LOG_FIELDS - set(record)
    assert not missing, f"trial log is missing {missing}"
    assert record["booklet_page_image"].endswith("real_booklet_page_12.png")
    assert record["extracted_goal"]
    assert record["wall_clock_seconds"] > 0
    assert record["steps"] and record["steps"][0]["action"] in ACTION_NAMES
    assert record["steps"][0]["primitive_count"] > 0
    assert record["verification"]["reason"]


def test_trial_log_is_valid_jsonl_and_appends(tmp_path):
    orchestrator, cell, _ = make_orchestrator(tmp_path)
    for _ in range(2):
        orchestrator.run_trial(
            PAGE, experiment="append", expected_ph_class="basic",
            on_step_executed=bench_state_updater(cell, reagent_effect="basic"),
        )
    lines = (tmp_path / "append.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_blocked_stages_are_named_not_silently_skipped(tmp_path, monkeypatch):
    """No API key must produce a record saying exactly which stage could not run."""
    from config import models

    monkeypatch.delenv(models.ENV_API_KEY, raising=False)
    cell = build_cell("mock")
    orchestrator = Orchestrator(cell=cell, log_dir=tmp_path, verbose=False)
    outcome = orchestrator.run_trial(PAGE, experiment="blocked", expected_ph_class="basic")

    assert outcome.outcome == "blocked"
    record = list(read_trials(tmp_path / "blocked.jsonl"))[0]
    assert record["blocked_steps"][0]["stage"] == "goal_extraction"
    assert models.ENV_API_KEY in record["blocked_steps"][0]["reason"]
    assert record["failure_kind"] == "blocked"
    assert "downstream stages were not run" in record["notes"]


def test_logger_records_the_model_tiers_but_never_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-logged")
    logger = TrialLogger("secrets", log_dir=tmp_path)
    payload = logger.finalise()
    assert payload["model_config"]["api_key_present"] is True
    assert "sk-should-never-be-logged" not in json.dumps(payload)


def test_summarise_produces_results_section_columns(tmp_path):
    orchestrator, cell, _ = make_orchestrator(tmp_path)
    orchestrator.run_trial(
        PAGE, experiment="summary", condition="fixed_scene", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell, reagent_effect="basic"),
    )
    orchestrator2, cell2, _ = make_orchestrator(tmp_path)
    orchestrator2.run_trial(
        PAGE, experiment="summary", condition="randomised_position", expected_ph_class="basic",
        on_step_executed=bench_state_updater(cell2, reagent_effect="none"),
    )

    stats = summarise(tmp_path / "summary.jsonl")
    assert stats["trials"] == 2
    assert stats["successes"] == 1
    assert stats["success_rate"] == 0.5
    assert stats["any_execution_is_fake"] is True
    assert set(stats["by_condition"]) == {"fixed_scene", "randomised_position"}
    assert stats["by_condition"]["fixed_scene"]["success_rate"] == 1.0
    assert stats["max_replanning_iterations"] == 3


def test_capability_workaround_serialises_for_the_log():
    workaround = CapabilityWorkaround(
        constraint="single gripper",
        literal_instruction="pour both cups at once",
        adopted_approach="pour sequentially",
        outcome_differs=True,
    )
    payload = workaround.as_dict()
    assert payload["outcome_differs"] is True
    json.dumps(payload)


# ==========================================================================
# Scope guards -- these encode CLAUDE.md's explicit boundaries
# ==========================================================================


def _project_modules() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    paths: list[Path] = []
    for pattern in ("*.py", "agents/*.py", "config/*.py", "hardware/*.py", "skills/*.py"):
        paths.extend(root.glob(pattern))
    return paths


def _string_literals(path: Path) -> list[str]:
    """Every string constant in a module, excluding docstrings.

    Docstrings are excluded deliberately: this file's own docstrings explain that
    upstream hardcoded ``model='gpt-4o'``, and describing the fix must not count
    as committing it.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_hardcoded_gpt_4o_survives_in_project_code():
    """The env-driven tiers replaced the hardcoded model; nothing may reintroduce it.

    ``config/models.py`` is exempt: its RETIRED_MODELS table names the retired
    models on purpose, which is how selecting one raises a warning.
    """
    offenders = []
    for path in _project_modules():
        if path.name == "models.py" and path.parent.name == "config":
            continue
        for literal in _string_literals(path):
            if "gpt-4o" in literal:
                offenders.append(f"{path.name}: {literal!r}")
    assert not offenders, f"hardcoded gpt-4o reintroduced: {offenders}"


def test_model_tiers_come_from_the_environment():
    """No agent may pin a model literal; all of them resolve through config.models."""
    from config import models

    assert models.cheap_model() not in models.RETIRED_MODELS
    assert models.strong_model() not in models.RETIRED_MODELS

    offenders = []
    for path in _project_modules():
        if path.name == "models.py" and path.parent.name == "config":
            continue
        for literal in _string_literals(path):
            if literal.startswith(("gpt-", "claude-", "gemini-", "o1", "o3")):
                offenders.append(f"{path.name}: {literal!r}")
    assert not offenders, f"model id pinned outside config/models.py: {offenders}"


def test_selecting_a_retired_model_warns(monkeypatch):
    from config import models

    monkeypatch.setenv(models.ENV_STRONG, "gpt-4o")
    with pytest.warns(RuntimeWarning, match="retired"):
        assert models.strong_model() == "gpt-4o"


def test_pipeline_modules_never_import_the_test_llm_double():
    """The fake LLM is a test fixture; production code must not reach for it."""
    root = Path(__file__).resolve().parents[1]
    for pattern in ("*.py", "agents/*.py", "config/*.py", "hardware/*.py", "skills/*.py"):
        for path in root.glob(pattern):
            assert "fake_llm" not in path.read_text(encoding="utf-8"), path


# ==========================================================================
# Mock bench props -- added after a live run showed a bare bench made the
# Planner (correctly) refuse to plan, because no reagents were present.
# ==========================================================================


def test_props_never_touch_the_verification_roi():
    """Kit props must not be able to influence any measurement.

    They are drawn right of PROP_ZONE_X; the verification crop sits left of it.
    Rendering with and without props must be pixel-identical inside CUP_ROI.
    """
    import numpy as np

    from hardware.mock import CUP_ROI, PROP_ZONE_X, BenchState, render_bench

    x, y, w, h = CUP_ROI
    assert x + w <= PROP_ZONE_X, "the verification crop overlaps the prop zone"

    bare = render_bench(BenchState(props=()))
    with_props = render_bench(BenchState())
    delta = np.abs(
        bare[y:y + h, x:x + w].astype(int) - with_props[y:y + h, x:x + w].astype(int)
    ).max()
    assert delta == 0, f"props changed {delta} levels inside the verification ROI"


def test_detectors_still_work_with_a_populated_bench():
    from agents.verification import VerificationAgent
    from hardware.mock import CUP_ROI, BenchState, MockCamera

    state = BenchState(color="purple", fill_level=0.45)
    camera = MockCamera(state)
    before = camera.capture()
    state.color = "blue"
    after = camera.capture()

    result = VerificationAgent().verify(
        "color", [before, after], roi=CUP_ROI, expected_ph_class="basic"
    )
    assert result.success


def test_bench_state_records_its_props_for_the_trial_log():
    from hardware.mock import BenchState

    payload = BenchState().as_dict()
    assert "BAKING SODA" in payload["props"]
    assert BenchState(props=()).as_dict()["props"] == []


def test_execution_failure_is_not_reported_as_a_verification_failure(tmp_path):
    """A live run mislabelled every execution failure as 'verification'.

    The evaluation protocol distinguishes planning sub-failures from
    manipulation sub-failures, so getting this attribution right is a
    Results-section correctness issue, not cosmetics.
    """
    orchestrator, cell, _ = make_orchestrator(tmp_path, fake_success_rate=0.0)
    orchestrator.run_trial(PAGE, experiment="attribution", expected_ph_class="basic")

    record = list(read_trials(tmp_path / "attribution.jsonl"))[0]
    assert record["failure_kind"] == "execution", record["failure_kind"]
    assert "failed during execution" in record["notes"]


def test_replanner_is_told_about_an_execution_failure_not_the_camera(tmp_path):
    """When a step fails, the planner must hear about THAT, not the colour check."""
    orchestrator, cell, llm = make_orchestrator(tmp_path, fake_success_rate=0.0)
    orchestrator.run_trial(PAGE, experiment="feedback_kind", expected_ph_class="basic")

    corrections = llm.calls_for("corrective_plan")
    assert corrections, "no corrective plan was requested"
    assert "failed during execution" in corrections[0].user_text
