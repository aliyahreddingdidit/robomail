# Progress — Autonomous Robotic Chemist

Last updated: 2026-09-04

Status of the first implementation pass against the architecture in `CLAUDE.md`.
The research plan this extends is in [`research_plan.md`](research_plan.md).

---

## The one thing to read first

**The pipeline has now run end to end against live models, successfully.** A key
was configured on 2026-09-04 and Experiment A completed: a real booklet page
photograph in, a verified colour change out, in 45 seconds, no human input at any
point. Most of what this document previously listed as unverified is now
verified; the remaining gaps are named below.

**The robot still does not move.** Skill execution remains a stub with
`FAKE_SUCCESS_RATE` (`skills/executor.py`), as specified. Any success rate from
these runs measures the planning/verification loop, not manipulation.

**The Scene Understanding Agent has still only ever seen a rendered mock bench**,
never a photograph of the real workspace. That is the largest remaining gap, and
it is a perception claim rather than a loop claim.

---

## Setup steps

| Step | Status |
| --- | --- |
| Clone PLATO into this folder | Done — `PLATO/` |
| Write the research plan appendix to `docs/research_plan.md` | Done — verbatim copy |

---

## Environment fixes (all four confirmed facts from CLAUDE.md)

| Issue | Status |
| --- | --- |
| `scikit-image=0.17.2`, `faiss=1.5`, `pydensecrf=1.0` — single `=` | Fixed to `==`; file now parses |
| Duplicate conflicting `Pillow` pin (`10.3.0` then `9.5.0`) | Fixed — kept `10.3.0`, removed the later `9.5.0` |
| `model='gpt-4o'` hardcoded in three agents (shutdown 2026-10-23) | Replaced by env-driven tiers in `config/models.py`; a regression test asserts no model id is pinned outside that file |
| `exec_script.py` hardcoded absolute author paths | Fixed to package-relative resolution; save dir now relative and `PLATO_SAVE_DIR`-overridable |
| No API key hardcoded anywhere | Confirmed. Read from env; a missing key raises a named, actionable error and the blocked stage is written into the trial log |

Originals preserved as `requirements.txt.orig` and `exec_script.py.orig`.

`requirements.txt` now *parses*, which is what was blocking pip. Two entries in
it are still not installable under those names — `faiss` ships as
`faiss-cpu`/`faiss-gpu`, and `pydensecrf` needs a source build. Neither is
imported by anything in this repo's pipeline. Resolving them is only necessary
to run PLATO's original perception stack on the robot PC. **Open.**

---

## Architecture items

| # | Item | Status |
| --- | --- | --- |
| 1 | Scene Understanding Agent | Built — `agents/scene_understanding.py`. Kit vocabulary added; structured JSON replaces upstream's `ast.literal_eval` text scraping; PLATO's handle flag preserved for downstream grasping |
| 2 | Goal Extraction Agent (NEW) | Built — `agents/goal_extraction.py`. Reads a booklet page image, returns one goal string. `validate_goal` rejects step lists, enumerations, sequencing words and chained procedure verbs |
| 3 | High-Level Planner Agent | Built — `agents/high_level_planner.py`. Decomposes the goal itself; robot profile injected; reports capability workarounds |
| 4 | Affordance / Tool-Use Agent | Built — `agents/affordance.py`. `PlatoGraspProvider` delegates to PLATO's `do_grasp`; `MockGraspProvider` for headless runs |
| 5 | Low-Level Action Generation | Built — `agents/step_planner.py`. LLM-driven go-to/grasp/tilt; action constrained to the enum in `agents/action_vocabulary.py` via JSON schema |
| 6 | Robot capability profile | Built — `config/robot_profile.py`. Injected into Planner and Step Planner prompts; workarounds logged |
| 7 | Chemistry Outcome Verification Agent (NEW) | Built — `agents/verification.py`. Three real detectors, 28 tests against static image fixtures |
| 8 | Logging / Data Agent (NEW) | Built — `agents/logging_agent.py`. JSONL, schema v1.0.0, plus `summarise()` for the Results table |
| 9 | Closed-loop replanning | Built — `orchestrator.py`. Cap of 3, every iteration logged, reported failure at the cap |
| 10 | Skill execution stubs | Built — `skills/executor.py`. `FAKE_SUCCESS_RATE`, loudly marked |
| 11 | Mock/real hardware toggle | Built — `hardware/`. `PLATO_HARDWARE=mock|real`, default mock |

---

## Definition of done

| Criterion | Status |
| --- | --- |
| Fresh clone + documented setup → environment installs cleanly | **Done for this repo.** `requirements.txt` installs, 120 tests pass, and the Docker image builds and passes the suite too. PLATO's own `requirements.txt` parses but two entries remain non-installable (above); the perception image is the place that resolves them |
| Goal Extraction Agent produces a single reasonable goal from a booklet photo | **Done, live.** Verified against both real booklet photographs on 2026-09-04 — see "Verified" below |
| Goal runs end-to-end, autonomously, no human input, logged | **Done, live.** Experiment A: real page in, verified colour change out, 7 sub-tasks, 45 s, one JSONL record, zero human input |
| Forced verification failure triggers ≥1 real replanning iteration, logged | **Done.** `test_forced_verification_failure_triggers_a_real_replan` — the bench genuinely does not change, real OpenCV genuinely fails, the planner is genuinely re-invoked. Also `test_a_trial_can_recover_on_a_corrective_attempt` |
| All three Verification detectors tested against static image fixtures | **Done.** 28 tests; colour additionally tested on two real booklet photographs |
| No hand-scripted/fixed-trajectory motion code anywhere | **Done.** All motion comes from the LLM Step Planner. `skills/executor.py` replays what the planner emitted |
| `docs/progress.md` and `docs/research_plan.md` exist and are current | **Done** |

Test suite: **120 passed** — the two previously-skipped live booklet checks now
run and pass. The same 120 pass inside the `chemist-agents` container.

The one criterion still qualified is scene grounding: everything above was
verified with the Scene Understanding Agent looking at a rendered mock bench,
not a photograph of the real workspace. See "Still not verified".

---

## Verified on 2026-09-04, against live models

1. **The Goal Extraction Agent reads real booklet pages correctly.** Both live
   tests pass. Page 12 gave *"create a magic beaker that separates colors and
   changes liquid color"*; page 13, *"produce a fizzing reaction with liquid
   changing color back to purple"*. Both are single outcome clauses, both
   survived the anti-plan guard rails, both describe what the page actually
   asks for. This was the largest open question in the repo.
2. **The High-Level Planner produces chemically sensible plans.** It grounded 8
   kit objects, chose water then citric acid then sodium bicarbonate, and kept
   correct gripper discipline throughout -- PICKUP, use, PLACE before switching
   tools -- prompted only by the capability profile.
3. **The robot profile genuinely changes plans.** The planner independently
   recognised that the booklet's simultaneous acid/base addition is impossible
   for a single-arm robot, serialised it, and logged a workaround naming the
   constraint and noting the outcome may differ. Previously unverified item 3;
   this is exactly what item 6 exists for.
4. **Structured output works against the real provider.** No fallback to
   JSON-object mode was needed; strict `json_schema` held across all five agents.
5. **The closed loop runs unattended.** 45 seconds, 7 sub-tasks, real OpenCV
   verification, one JSONL record, zero human input.

## Still not verified

1. **Scene grounding on a real photograph.** The Scene Understanding Agent has
   only been shown `MockCamera` renderings -- labelled tubs drawn with OpenCV
   primitives. It reads them correctly, but that tests the loop, not perception.
   A photograph of the actual bench with the kit laid out is needed; the
   orchestrator already accepts one via `workspace_image=`.
2. **Experiments B, C and D end to end.** Only A has run live. B's motion
   modality and D's "successful non-reaction" logic are untested against real
   plans, and C and D still need booklet page photographs.
3. **Step Planner geometry.** Primitives are well-formed and action labels are
   enum-valid, but whether the centimetre deltas are physically sensible cannot
   be known without a real cell.
4. **Live recovery via replanning.** A live run did trigger three corrective
   attempts, but none has yet *recovered* through a corrective plan.

## Bugs the live run exposed, and their fixes

Three things the offline suite could not have caught:

1. **A bare mock bench made the Planner refuse to plan.** With only a cup
   rendered, Scene Understanding correctly reported one object and the Planner
   correctly declined to invent reagents. Both agents were right; the mock scene
   was too impoverished to exercise the loop. `hardware/mock.py` now renders the
   kit as labelled tubs, provably outside the verification ROI.
2. **Execution failures were mislabelled as verification failures.** The
   orchestrator reported `failure_kind: "verification"` whenever the retry cap
   was hit, even when every attempt had died on an infeasible sub-task. That
   corrupts the planning-vs-manipulation split the evaluation protocol depends
   on. Fixed, with a regression test.
3. **The replanner was fed the wrong reason.** When a step failed, `replan()`
   still received the verification reason -- which, after a partially executed
   plan, could read "PASS". The planner was told the outcome succeeded while
   being asked to fix a failure. Fixed, with a regression test.

**A fourth issue is known and NOT fixed.** The scene is grounded once, at the
start of a trial, and never re-grounded. So a step like "pipette the sodium
bicarbonate solution from the clear cup" is judged against the ORIGINAL scene,
where that cup was empty, and is correctly reported infeasible even though an
earlier step would have filled it. The agents' world model goes stale as the plan
executes. Fixing it properly is a design decision -- re-ground per sub-task, or
maintain a symbolic world state alongside the scene -- and is the most
substantive open question in the architecture right now.

### Model choice needs confirming

`config/models.py` defaults to `gpt-4.1-mini` (cheap tier) and `gpt-4.1`
(strong tier). These were chosen because they support vision and strict
structured output and are not the retired `gpt-4o`. **This is a judgement call,
not a decision from the research plan** — confirm the intended provider and
tier before running the experiment matrix, since it goes in the paper's
Experimental Setup section. Both are env-overridable; no code change is needed
to switch.

---

## Blockers and open questions

1. **No PLATO fork exists yet.** `docs/plato_portability_fixes.patch` and the
   local `chemist-patches` branch inside `PLATO/` hold our two portability
   fixes, but neither can be pushed anywhere or registered as a submodule
   until a fork exists — a submodule can only point at a commit its own
   remote can serve. `robomail` has a working fork
   ([rumilog/robomail](https://github.com/rumilog/robomail)) and is wired as
   a real submodule at `third_party/robomail`; PLATO is not, and `/PLATO/` is
   explicitly gitignored at the root until it is. See the README's "PLATO: a
   real checkout, not (yet) a real submodule" section for the exact unblock
   steps.
2. **No booklet photos for Experiments C and D.** The Goal Extraction Agent
   reads the goal from an image by design, so instant snow (C) and hydrophobic
   sand (D) cannot run until someone photographs those booklet pages.
   `run_experiment.py` reports this and exits rather than fabricating a goal.
   The C/D *detectors* are fully implemented and tested against fixtures.
3. **Camera setup not locked in** (research plan §6, needed before M0): single
   overhead vs. wrist-mounted + overhead. The Verification Agent currently
   assumes one fixed view with a fixed cup ROI (`hardware/mock.py::CUP_ROI`).
   A wrist camera would need ROI tracking rather than a fixed crop.
4. **Reference swatches are nominal, not photographed.** `DEFAULT_PALETTE` holds
   plausible cabbage-indicator RGBs, not measurements under lab lighting.
   Photographing real swatches is a parallel human task (out of scope here);
   `load_palette()` takes a JSON palette so nothing else changes when they exist.
5. **PLATO's grasping and segmentation submodules are not initialised.**
   `grasping/os_tog` and `SAM/GroundingDINO` are empty in a plain clone.
   `PlatoGraspProvider` raises an actionable error naming the fix.
6. **Absolute author paths remain outside `exec_script.py`.** `SAM/sam3d.py`,
   `SAM/object_segmentation.py`, `Utils/Mask_generator.py` and
   `Baselines/RobotTool/exec_script.py` still contain `/home/arvind/...` paths.
   CLAUDE.md named only `exec_script.py`, and the Baselines one belongs to the
   out-of-scope scripted baseline, so these were left alone deliberately rather
   than silently widening scope. They will need fixing before the real cell runs.

---

## Deliberately out of scope for this pass

Recorded here so they are not lost:

- **The scripted / hard-coded baseline system** — the paper's comparison
  condition, and the thing that turns this from a demo into a publishable
  result. A fully separate system, not built here. Upstream's
  `PLATO/Baselines/RobotTool/` is the closest existing starting point.
- **Real hand-scripted motion tuning** for any skill — stays fake pass/fail.
- **Real camera calibration and reference-swatch photography** under lab
  lighting — parallel human tasks.
- **Any human-in-the-loop review or approval step** — excluded by design, not
  deferred. The pipeline has none, and `test_no_stage_ever_asks_for_human_input`
  asserts the prompts do not invite one.
- **The Goal Extraction Agent producing anything beyond a single goal string.**

---

## Next actions

1. Set `OPENAI_API_KEY` and run `python -m pytest tests/ -q` — the 2 skipped
   tests become live and answer unverified item 1.
2. Then `python run_experiment.py --experiment A --trials 3` for a real
   end-to-end run, and inspect `data/logs/experiment_a_ph_indicator.jsonl`.
3. Confirm the model tier choice (above).
4. Photograph the instant-snow and hydrophobic-sand booklet pages to unblock
   Experiments C and D.
5. Lock in the camera configuration (research plan §6) before M0.
6. Run page 13 through the Planner specifically to see whether the
   "pour both cups at the same time" constraint conflict is caught and logged.
