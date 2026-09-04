"""CLI entry point: run trials of the wet-chemistry experiments.

    python run_experiment.py --experiment A --trials 3
    python run_experiment.py --experiment A --page path/to/booklet_page.jpg
    python run_experiment.py --summarise data/logs/experiment_a.jsonl

Everything runs against the mock cell by default (PLATO_HARDWARE=mock). Set
PLATO_HARDWARE=real on the robot control PC to drive the real Franka.

Requires OPENAI_API_KEY. Without it the run does not silently skip the LLM
stages -- it reports exactly which stage was blocked, writes that into the trial
log, and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from agents.logging_agent import summarise
from config import models
from hardware.factory import build_cell
from hardware.mock import BenchState
from orchestrator import Orchestrator, bench_state_updater
from skills.executor import FAKE_SUCCESS_RATE, fake_execution_banner

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests" / "fixtures"


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment from the research plan's Section 4 protocol."""

    key: str
    name: str
    log_name: str
    #: Booklet page photo standing in for the robot's camera reading the manual.
    #: None means we do not yet have a photo of that page in this project.
    default_page: Path | None
    modality: str
    expect_change: bool
    expected_ph_class: str | None
    #: What the mock bench does when a reagent transfer succeeds.
    reagent_effect: str
    initial_bench: BenchState


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "A": ExperimentSpec(
        key="A",
        name="pH Indicator Colour-Change (primary demo)",
        log_name="experiment_a_ph_indicator",
        default_page=FIXTURES / "real_booklet_page_12.png",
        modality="color",
        expect_change=True,
        expected_ph_class="basic",
        reagent_effect="basic",
        initial_bench=BenchState(color="purple", fill_level=0.45),
    ),
    "B": ExperimentSpec(
        key="B",
        name="Acid-Base Effervescence (gas-evolution verification)",
        log_name="experiment_b_effervescence",
        default_page=FIXTURES / "real_booklet_page_13.png",
        modality="motion",
        expect_change=True,
        expected_ph_class=None,
        reagent_effect="fizz",
        initial_bench=BenchState(color="purple", fill_level=0.45),
    ),
    "C": ExperimentSpec(
        key="C",
        name="Granular / Powder Manipulation (instant snow)",
        log_name="experiment_c_instant_snow",
        default_page=None,  # no photo of the instant-snow booklet page yet
        modality="volume",
        expect_change=True,
        expected_ph_class=None,
        reagent_effect="expand",
        initial_bench=BenchState(color="purple", fill_level=0.18),
    ),
    "D": ExperimentSpec(
        key="D",
        name="Negative Control (hydrophobic sand)",
        log_name="experiment_d_hydrophobic_sand",
        default_page=None,  # no photo of the hydrophobic-sand booklet page yet
        modality="volume",
        expect_change=False,
        expected_ph_class=None,
        reagent_effect="none",
        initial_bench=BenchState(color="purple", fill_level=0.18),
    ),
}


def run(spec: ExperimentSpec, *, page: Path, trials: int, condition: str,
        log_dir: Path, verbose: bool) -> int:
    """Run ``trials`` trials of ``spec``. Returns the number that succeeded."""
    successes = 0
    for trial in range(1, trials + 1):
        print(f"\n--- {spec.key} trial {trial}/{trials} ({condition}) ---")
        cell = build_cell("mock", bench_state=spec.initial_bench.copy())
        orchestrator = Orchestrator(cell=cell, log_dir=log_dir, verbose=verbose)
        outcome = orchestrator.run_trial(
            page,
            experiment=spec.log_name,
            condition=condition,
            expect_change=spec.expect_change,
            expected_ph_class=spec.expected_ph_class,
            on_step_executed=bench_state_updater(cell, reagent_effect=spec.reagent_effect),
        )
        successes += int(outcome.success)
        if outcome.outcome == "blocked":
            print(f"\nBLOCKED: {outcome.reason}", file=sys.stderr)
            return -1
    return successes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), help="which experiment to run")
    parser.add_argument("--page", type=Path, help="booklet page photo (overrides the default)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--condition", default="fixed_scene",
                        help="evaluation condition label recorded in the log")
    parser.add_argument("--log-dir", type=Path, default=Path("data/logs"))
    parser.add_argument("--summarise", type=Path, help="summarise an existing JSONL log and exit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.summarise:
        print(json.dumps(summarise(args.summarise), indent=2))
        return 0

    if not args.experiment:
        parser.error("--experiment is required (or use --summarise)")

    spec = EXPERIMENTS[args.experiment]
    page = args.page or spec.default_page
    if page is None:
        print(
            f"Experiment {spec.key} ({spec.name}) has no booklet page photo in this project.\n"
            f"The Goal Extraction Agent reads the goal from a photo of the booklet page -- it is "
            f"not given the goal as text -- so this experiment cannot run until someone "
            f"photographs that page. Pass one with --page.",
            file=sys.stderr,
        )
        return 2
    if not Path(page).is_file():
        print(f"booklet page image not found: {page}", file=sys.stderr)
        return 2

    print(f"Experiment {spec.key}: {spec.name}")
    print(f"  booklet page : {page}")
    print(f"  modality     : {spec.modality}   expect_change={spec.expect_change}")
    print(f"  models       : {models.describe()}")
    print(f"  {fake_execution_banner()}")

    if models.api_key() is None:
        print(
            f"\nNo API key configured. Set {models.ENV_API_KEY} before running.\n"
            "The pipeline will report the blocked stage rather than skipping it silently.",
            file=sys.stderr,
        )

    successes = run(spec, page=Path(page), trials=args.trials, condition=args.condition,
                    log_dir=args.log_dir, verbose=not args.quiet)
    if successes < 0:
        return 1

    print(f"\n{successes}/{args.trials} trials verified successfully.")
    log_path = args.log_dir / f"{spec.log_name}.jsonl"
    if log_path.is_file():
        print(json.dumps(summarise(log_path), indent=2))
    return 0 if successes == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
