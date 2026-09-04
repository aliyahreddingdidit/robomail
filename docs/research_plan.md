## Appendix: Research Plan (verbatim, from Barati Farimani Lab)

**AUTONOMOUS ROBOTIC CHEMIST**
Extending PLATO to Closed-Loop Wet-Chemistry Manipulation
A research plan and proof-of-concept protocol
Barati Farimani Lab — Carnegie Mellon University

### 1. Framing and Novelty Argument
PLATO (Car, Yarlagadda, Bartsch, George, Barati Farimani — CMU) already solves the core hard problem this project needs: a modular LLM-agent stack (scene understanding → high-level task planning → tool-affordance prediction → low-level action generation → step verification) running on a Franka Emika Panda, tested on long-horizon tool-manipulation tasks without hard-coded scene knowledge.

The gap PLATO doesn't currently close is closed-loop wet-chemistry manipulation with visually-verifiable chemical outcomes — pouring variable-viscosity liquids and powders, mixing in the correct order/ratio, and reading a chemical result back into the planner as a verification signal rather than just checking whether the gripper reached the target pose.

This is a natural, low-risk testbed because the National Geographic Magic Chemistry Set (NGSCIMAGIC) reagents are all food-grade / non-hazardous. No PPE, ventilation, or hazardous-material handling is required — this lets the paper's contribution focus on the agentic/manipulation problem, not chemical safety engineering, while still producing genuinely novel closed-loop verification data (color/gas/volume change as a reward-like signal for an LLM planner).

| Kit Component | Role in Protocol |
| --- | --- |
| Sodium bicarbonate | Base reagent, effervescence trigger |
| Citric acid | Acid reagent, effervescence trigger |
| Red cabbage powder (indicator) | Visual pH indicator (pink/red = acidic, green/blue = basic) |
| Instant snow powder | Superabsorbent polymer — volume-expansion manipulation task |
| Hydrophobic sand | Granular material manipulation + a "control" reagent that should visibly not react with water |
| Beaker, pipette, clear cups, paper cups, measuring implements | Manipulation targets of varying affordance (graspable, pourable, dispensable) |
| pH indicator paper | Secondary, low-cost ground-truth verification channel |

**One-line contribution**: We extend PLATO's tool-manipulation framework with a chemistry-outcome verification agent, enabling a Franka Panda arm to autonomously plan, execute, and visually verify multi-step wet-chemistry experiments under scene variation, without hard-coded knowledge of reagent identity or container layout.

### 2. System Architecture
This project extends PLATO's existing agents with one new capability: reading a chemical outcome back into the planning loop.

**2.1 Scene Understanding Agent (existing, reused)** — Identifies containers, reagents, and tools in the workspace via vision-language grounding; the object vocabulary/prompts need to be extended to include beakers, cups, powders, and pipettes.

**2.2 High-Level Planner Agent (existing, reused)** — Given a natural-language goal (e.g., "turn the cabbage indicator green"), generates an ordered sub-task plan (e.g., pour indicator → add base → observe).

**2.3 Affordance / Tool-Use Agent (existing, reused)** — Maps each sub-task to a grasp/pour/dispense primitive and the correct tool (pipette for drops, cup for pouring, spoon/scoop for powder).

**2.4 Low-Level Action Generation (existing, reused)** — Franka Panda joint/end-effector trajectories for grasp, pour, dispense, and stir.

**2.5 Chemistry Outcome Verification Agent (NEW)** — A vision-based module that checks for the chemical success condition: color classification against a reference palette for the cabbage indicator; bubble/foam detection via optical flow or frame-differencing for the acid–base reaction; liquid-level/volume change detection for the instant-snow swelling. Feeds a success/failure signal back to the Planner Agent for replanning (e.g., "add more acid, color hasn't shifted") — this is the genuinely new capability the paper demonstrates.

**2.6 Logging / Data Agent (NEW, lightweight)** — Timestamps each sub-task and saves before/after frames and the verification signal, for the results section (success rate, replanning count, time-to-completion).

### 3. Milestones (M0–M6)

| Milestone | Goal | Exit Criterion |
| --- | --- | --- |
| M0 — Environment setup | Mount kit components in the Franka workspace; calibrate camera(s); build reagent/container vocabulary | Agent correctly identifies ≥90% of kit objects across 10 random layouts |
| M1 — Manipulation primitive calibration | Validate grasp, pour, and pipette-dispense primitives in isolation | Each primitive succeeds ≥8/10 trials without spillage beyond tolerance |
| M2 — Single-step chemistry task | Autonomous single reagent transfer + outcome read | Correct color classification ≥90% across lighting/position variation |
| M3 — Two-step reactive task | Full acid–base or indicator-color-change experiment end-to-end, planner-generated | ≥80% success rate over 15 trials; correct outcome reported |
| M4 — Closed-loop replanning | Introduce a perturbation; require Verification Agent to trigger a corrective replan | ≥70% of perturbed trials recover without human intervention |
| M5 — Scene-variation robustness | Repeat M3–M4 under randomized position, lighting, distractors | Success rate drop ≤15 percentage points vs. fixed-scene baseline |
| M6 — Data collection + write-up | Run full experiment matrix (Section 4) for N≥15 per condition | Dataset and figures ready for Results section; draft complete |

Suggested pace: M0–M1 (1–2 weeks), M2 (1 week), M3 (1–2 weeks), M4 (1–2 weeks), M5 (1 week), M6 (1–2 weeks) — roughly a 6–10 week proof-of-concept arc, compressible if the PLATO stack needs minimal modification.

### 4. Step-by-Step Experimental Protocol
Each experiment below is designed to be planner-generated from a natural-language goal, not scripted — the point of the paper is that the LLM planner decomposes the goal itself using the Scene Understanding and Affordance agents. What's fixed is the evaluation protocol, not the plan.

**Experiment A — pH Indicator Color-Change (primary demo)**
Goal prompt: "Turn the cabbage indicator solution basic (blue/green)."
- Scene Understanding Agent identifies: cup of prepared red-cabbage indicator solution, sodium bicarbonate container, measuring scoop, pipette.
- Planner generates sub-tasks: (a) scoop a measured amount of sodium bicarbonate, (b) add to indicator cup, (c) stir/agitate if needed, (d) capture verification frame.
- Franka executes scoop → pour → (optional) stir trajectory.
- Verification Agent samples the cup's color via a reference-swatch classifier (pink/purple = acidic/neutral, blue/green = basic) and reports success/failure.
- On failure (no visible shift), Planner Agent re-invokes steps (a)–(c) with an incremented reagent amount — this is the headline closed-loop result.
- Repeat with citric acid instead of sodium bicarbonate to demonstrate the acidic direction (indicator turns pink/red), showing the system generalizes the procedure rather than memorizing one reagent.
Verification signal: reference-swatch color classifier on the cup.

**Experiment B — Acid–Base Effervescence (gas-evolution verification)**
Goal prompt: "Produce a fizzing reaction using the chemistry kit."
- Planner must identify citric acid + sodium bicarbonate + water as the required combination — a genuine planning step, not a single pour.
- Sequence: dispense water into cup → add citric acid → add sodium bicarbonate (or reverse order, both work) → observe.
- Verification Agent detects effervescence via frame-differencing/optical flow (bubble motion) over a short observation window — demonstrating generalization across modalities (color vs. motion).
- Failure mode to test deliberately: omit water (reagents won't dissolve/react) — confirm the system detects "no reaction" and replans to add the missing step.
Verification signal: motion-based bubble/foam detection.

**Experiment C — Granular / Powder Manipulation (instant snow)**
Goal prompt: "Make the powder expand into snow."
- Tests a manipulation primitive PLATO wasn't originally evaluated on: controlled powder dispensing followed by liquid addition to a granular material.
- Verification Agent measures volume/height change in the cup (pixel-height delta from a fixed camera angle) as the success signal.
- Useful positive-transformation counterpart to the negative control in Experiment D.
Verification signal: pixel-height / volume delta.

**Experiment D — Negative Control (hydrophobic sand)**
Goal prompt: "Show that this sand does not get wet."
- Planner must recognize this as a demonstration/verification goal rather than a transformation goal — the correct action is to add water and confirm the sand stays dry/separates, not to "fix" a non-reaction.
- Stress-tests whether the Verification Agent (and the Planner's interpretation of "success") can distinguish "no reaction happened, and that's correct" from "no reaction happened, and that's a failure" — a subtlety worth a paragraph in the Discussion section, since most agentic-manipulation papers only test for positive-outcome success.
Verification signal: confirms absence of wetting/color change is the expected, correct outcome.

**Cross-Cutting Evaluation Protocol (applies to A–D)**
- N = 15–20 trials per experiment, split across: (i) fixed scene layout, (ii) randomized container position (±10 cm), (iii) randomized lighting (2–3 conditions), (iv) added distractor objects (unused kit items in-frame).
- Metrics logged per trial: end-to-end success (binary), number of replanning iterations, time-to-completion, manipulation sub-failures (spill, missed grasp, pour miss) vs. planning sub-failures (wrong reagent/order chosen), verification-agent accuracy vs. ground truth (pH paper strip reading logged manually for Experiments A/B).
- Baseline for comparison: a hard-coded/scripted version of the same four experiments (no LLM planner, fixed trajectories), run under conditions (ii)–(iv) only, to show the core claim: the LLM-agent approach degrades less under scene variation than a scripted baseline. This comparison is what turns the proof-of-concept into a publishable result rather than a demo.

### 5. Suggested Paper Outline

| # | Section | Contents |
| --- | --- | --- |
| 1 | Abstract | One-line contribution + headline result (e.g., success % under scene variation vs. scripted baseline) |
| 2 | Introduction | Motivate autonomous experimentation; contrast against structured self-driving-lab systems (e.g., A-Lab, RoboChem-style), which typically assume pre-mapped labware |
| 3 | Related Work | PLATO and prior LLM-agent manipulation papers; self-driving-lab / autonomous-chemistry literature; hard-coded liquid-handling robots |
| 4 | Method | Section 2 architecture, emphasizing the new Chemistry Outcome Verification Agent as the delta over base PLATO |
| 5 | Experimental Setup | Hardware (Franka Panda + camera rig), kit reagents/containers, safety notes |
| 6 | Experiments & Results | Experiments A–D with cross-cutting metrics; scripted-baseline comparison; Verification Agent ablation (with vs. without closed-loop replanning) |
| 7 | Discussion | Experiment D's "successful non-reaction" framing; failure-mode analysis (perception vs. planning vs. execution); limitations (kit-scale, non-hazardous reagents only) |
| 8 | Conclusion & Future Work | Path toward standard lab glassware/instruments, multi-agent lab teams, and analytical verification (e.g., pH meter integration) beyond visual color/motion cues |

### 6. Immediate Next Actions
- Confirm which camera setup is being used with the Franka rig (single overhead vs. wrist-mounted + overhead) — this affects how the Verification Agent is built and should be locked in before M0.
- Decide whether the Scene Understanding Agent's object vocabulary needs retraining/re-prompting for the new object classes (beaker, pipette, powder scoop) or whether PLATO's existing grounding generalizes zero-shot — worth a quick pilot test before committing to the M0 timeline.
- Prepare the reference color-swatch set for the cabbage-indicator classifier (a handful of reference photos under actual lab lighting) — cheapest, fastest thing to do in parallel with hardware setup.
