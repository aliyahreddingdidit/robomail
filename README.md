# Autonomous Robotic Chemist

Extending [PLATO](https://github.com/ArvindCar/PLATO) (Car, Yarlagadda, Bartsch,
George, Barati Farimani — CMU, 2024) to closed-loop, visually-verified
wet-chemistry manipulation on a Franka Emika Panda.

The research plan is in [`docs/research_plan.md`](docs/research_plan.md); current
status, blockers and what is *not* yet verified are in
[`docs/progress.md`](docs/progress.md).

> **The robot does not move in this pass.** Skill execution is a stub with a
> configurable fake success rate (`skills/executor.py`). This pass validates the
> planning → affordance → action → verification → logging loop, not motion. Do
> not report numbers from these runs as manipulation success rates.

## Setup

```bash
git clone --recurse-submodules <this repo's URL>
```

That pulls in `third_party/robomail` (the MAIL lab's Franka utilities,
[rumilog/robomail](https://github.com/rumilog/robomail) — PLATO's
`import robomail.vision` depends on it) as a real submodule. If you cloned
without `--recurse-submodules`, catch up with:

```bash
git submodule update --init --recursive
```

**PLATO itself is not yet a submodule** — see the box below.

```bash
pip install -r requirements.txt
```

### PLATO: a real checkout, not (yet) a real submodule

```bash
git clone https://github.com/ArvindCar/PLATO.git
cd PLATO && git apply ../docs/plato_portability_fixes.patch && cd ..
```

A git submodule is a pointer to a commit on *someone else's* remote. Our two
portability fixes (`requirements.txt` syntax, `exec_script.py`'s hardcoded
paths) only exist as a local, unpushed branch (`chemist-patches`) inside a
plain PLATO checkout — there is no fork yet to point a submodule at. Wiring
one anyway, pointing at unpatched upstream, would silently hand every future
clone the broken files back. So for now: clone PLATO plainly and apply
`docs/plato_portability_fixes.patch` (a plain diff, independent of any branch)
by hand. `python scripts/check_setup.py` verifies this landed correctly.

Once a fork exists (`ArvindCar/PLATO` → Fork, or the lab's own fork):

```bash
git submodule add <fork-url> PLATO   # from the repo root
# then, inside PLATO/: point origin at the fork and push chemist-patches
```

and remove the `/PLATO/` line from `.gitignore`.

Then create your `.env` **in a text editor** (never type a key into a terminal —
shell history keeps it in plaintext):

```bash
cp .env.example .env
```

Fill in `OPENAI_API_KEY`. The file is gitignored, real environment variables
take precedence over it, and the key is never written to a log or an error
message. Confirm the checkout is ready:

```bash
python scripts/check_setup.py
```

That is enough to run everything in this repo. PLATO's own perception stack
(`PLATO/PLATO/requirements.txt`) is only needed to drive the real robot cell,
and its two git submodules (`SAM/GroundingDINO`, `grasping/os_tog`) need
`git submodule update --init --recursive` inside `PLATO/`.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | *(required)* | LLM access. Never hardcoded or read from disk. |
| `PLATO_MODEL_CHEAP` | `gpt-4.1-mini` | Scene Understanding, Goal Extraction |
| `PLATO_MODEL_STRONG` | `gpt-4.1` | Planner, Step Planner, Verification reasoning |
| `PLATO_LLM_BASE_URL` | *(unset)* | Any OpenAI-compatible endpoint |
| `PLATO_HARDWARE` | `mock` | `mock` or `real` |
| `PLATO_FAKE_SUCCESS_RATE` | `1.0` | Stubbed skill success probability |

Upstream PLATO hardcodes `model='gpt-4o'`, whose API shuts down 2026-10-23.
Nothing here defaults to it, and selecting it raises a `RuntimeWarning`.

## Running

```bash
python run_experiment.py --experiment A --trials 3
```

```bash
python run_experiment.py --summarise data/logs/experiment_a_ph_indicator.jsonl
```

```bash
python -m pytest tests/ -q
```

Experiments A and B have booklet page photographs available in this project.
C and D need photos of their booklet pages before they can run — the Goal
Extraction Agent reads the goal from an image, not from a text field, so there
is no way to run them without one. Pass `--page` once those photos exist.

## Running in Docker

Two images, scoped deliberately. Day-to-day work only needs the first.

| Image | Contents | Size | When |
| --- | --- | --- | --- |
| `chemist-agents` | The pipeline + tests. No GPU, no robot. | 982 MB (measured) | Always |
| `chemist-perception` | PLATO's vision stack: torch, GroundingDINO, SAM, faiss. | Several GB (not yet built) | Only for real scene grounding |

Most of the agents image is `opencv-python`, which bundles GUI libraries this
pipeline never uses. Switching it to `opencv-python-headless` (and dropping the
`libgl1`/`libglib2.0-0` apt packages that exist only to satisfy it) would cut
that substantially. It is left as-is so `requirements.txt` stays authoritative
for host and container alike — the container tests the same package you run
locally. Revisit if image size starts to matter.

```bash
docker compose run --rm tests
```

```bash
docker compose run --rm agents python run_experiment.py --experiment A --trials 3
```

The perception image is profile-gated so it is never built by accident:

```bash
docker compose --profile perception build perception
```

**The Franka control loop is deliberately not containerised.** `frankapy` needs a
PREEMPT_RT kernel and low-latency scheduling; it runs natively on the robot
control PC and the containers reach it over the network.

Notes: the API key arrives via compose's `env_file` at run time and is excluded
from image layers by `.dockerignore`. `data/logs` is bind-mounted, so trial
logs — research data — outlive the container. Model checkpoints are not baked
in; drop them in `./weights`, which mounts read-only at `/weights`.

## Architecture

```
booklet page photo
   │
   ├─ Goal Extraction Agent ......... agents/goal_extraction.py     (NEW, item 2)
   │     one goal string, never a plan
   │
   ├─ Scene Understanding Agent ..... agents/scene_understanding.py (item 1)
   │     grounded objects + handle flags
   │
   ├─ High-Level Planner Agent ...... agents/high_level_planner.py  (item 3)
   │     decomposes the goal itself; robot profile injected
   │
   ├─ for each sub-task:
   │     Affordance / Tool-Use ...... agents/affordance.py          (item 4)
   │     Low-Level Action Gen ....... agents/step_planner.py        (item 5)
   │        LLM-driven go-to/grasp/tilt, enum-constrained action
   │     Skill execution ............ skills/executor.py            (item 10, FAKE)
   │
   ├─ Chemistry Outcome Verification  agents/verification.py        (NEW, item 7)
   │     colour / motion / volume — real OpenCV
   │
   ├─ on failure: replan ............ orchestrator.py               (item 9, cap 3)
   │
   └─ Logging / Data Agent .......... agents/logging_agent.py       (NEW, item 8)
         one JSONL record per trial
```

Supporting modules:

| File | Role |
| --- | --- |
| `agents/action_vocabulary.py` | The closed action enum (item 5) |
| `config/robot_profile.py` | Hard capability constraints, injected at plan time (item 6) |
| `config/models.py` | Env-driven model tiers |
| `hardware/` | `MockFrankaArm`/`MockCamera` ↔ frankapy/RealSense toggle (item 11) |

### Two things that are mocked, and one that is not

- **Hardware is mocked.** `MockFrankaArm` records the primitives the LLM Step
  Planner emits; `MockCamera` renders bench frames.
- **The LLM transport is mocked in tests only** (`tests/fake_llm.py`), the same
  way the arm is. Production code never imports it.
- **Verification is not mocked.** The three detectors are real OpenCV running on
  real pixels, in tests and in mock runs alike. That is the seam: mock
  *hardware*, real *verification*.

## Design constraints this repo holds to

- The pipeline stays **LLM-driven at the motion-planning level**. There are no
  fixed or hand-authored trajectories anywhere. The paper's claim depends on it.
- **No human-in-the-loop** step exists anywhere: no review gate, no approval, no
  clarification prompt. Capability conflicts are reasoned around at plan time and
  logged, not escalated.
- The **Goal Extraction Agent emits one goal and nothing else.** Decomposition is
  the Planner's job; `validate_goal` rejects anything that has become a plan.
- An action outside the fixed vocabulary is a **reported failure**, never an
  invented action name.
