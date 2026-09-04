"""Mock arm and camera (CLAUDE.md architecture item 11).

`MockFrankaArm` records the commands it is given and tracks what the gripper
holds, so single-gripper violations surface in the trial log instead of being
silently absorbed.

`MockCamera` renders frames from a tiny :class:`BenchState` scene model. This
matters: the Verification Agent's detectors are real OpenCV, so they need real
pixels to look at. The mock camera supplies the pixels; the analysis of those
pixels is not mocked. That is the seam -- mock *hardware*, real *verification*.

None of this is a scripted trajectory. The arm executes whatever primitives the
LLM Step Planner emits; the mock simply records them and reports success/failure
according to the configured fake success rate (item 10).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

# Frame geometry shared with the fixtures, so a mock run and a fixture test see
# the same camera framing.
FRAME_W, FRAME_H = 640, 360
#: The verification crop. Deliberately unchanged by the frame widening above --
#: the cup sits where it always did, so the colour/motion/volume detectors and
#: their fixtures are untouched. Kit props are drawn to the RIGHT of x=260, well
#: clear of this rectangle, so they can never contaminate a measurement.
CUP_ROI: tuple[int, int, int, int] = (80, 120, 160, 170)
PROP_ZONE_X = 270


@dataclass
class CommandRecord:
    """One primitive as actually issued to the arm."""

    kind: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"kind": self.kind, **self.detail}


class MockFrankaArm:
    """Stand-in for frankapy's FrankaArm. Same interface, no hardware."""

    def __init__(self) -> None:
        self.commands: list[CommandRecord] = []
        self._holding: str | None = None
        self._gripper_closed = False
        self.constraint_violations: list[str] = []

    # -- ArmInterface ------------------------------------------------------
    def reset_joints(self) -> None:
        self.commands.append(CommandRecord("RESET"))

    def goto_delta(self, location: str, delta_cm: tuple[float, float, float]) -> None:
        self.commands.append(
            CommandRecord("GOTO", {"location": location, "delta_cm": list(delta_cm)})
        )

    def set_gripper(self, closed: int) -> None:
        self._gripper_closed = bool(closed)
        self.commands.append(CommandRecord("GRASP", {"gripper": int(closed)}))

    def tilt(self, theta_deg: tuple[float, float, float]) -> None:
        self.commands.append(CommandRecord("TILT", {"theta_deg": list(theta_deg)}))

    @property
    def holding(self) -> str | None:
        return self._holding

    # -- bookkeeping used by the skill executor ---------------------------
    def acquire(self, object_name: str) -> None:
        """Record that the gripper took hold of ``object_name``."""
        if self._holding is not None:
            self.constraint_violations.append(
                f"attempted to pick up {object_name!r} while already holding "
                f"{self._holding!r} (single gripper)"
            )
        self._holding = object_name

    def release(self) -> str | None:
        released, self._holding = self._holding, None
        return released

    def as_dict(self) -> dict:
        return {
            "command_count": len(self.commands),
            "commands": [c.as_dict() for c in self.commands],
            "holding": self._holding,
            "constraint_violations": list(self.constraint_violations),
        }


# --------------------------------------------------------------------------
# Bench scene model
# --------------------------------------------------------------------------

INDICATOR_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (209, 43, 60),
    "pink": (200, 62, 122),
    "purple": (123, 63, 158),
    "blue": (59, 91, 196),
    "green": (79, 158, 84),
}


#: The kit items laid out beside the indicator cup, as (label, swatch RGB).
#:
#: These exist because a live run exposed the gap: with only a cup on the bench,
#: the Scene Understanding Agent correctly reported one object, and the Planner
#: correctly refused to plan a reagent addition with no reagent present. The
#: mock scene has to contain the kit for the loop to be exercisable at all.
#:
#: They are drawn as labelled tubs. A real bench presents printed labels too, so
#: reading them is legitimate grounding -- but be clear about what this does and
#: does not test: it exercises the PLANNING loop end to end. It is not evidence
#: that the Scene Understanding Agent grounds real labware in a real photograph.
#: That needs a photograph of the actual bench (pass `--workspace`).
DEFAULT_PROPS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("BAKING SODA", (242, 242, 238)),
    ("CITRIC ACID", (236, 232, 214)),
    ("CABBAGE POWDER", (128, 74, 148)),
    ("WATER", (206, 226, 238)),
    ("SCOOP", (176, 176, 182)),
    ("PIPETTE", (198, 206, 210)),
    ("EMPTY CUP", (214, 218, 222)),
)


@dataclass
class BenchState:
    """What the mock camera should render.

    The orchestrator mutates this as (fake) skills execute, so the frames the
    Verification Agent analyses actually reflect what the run did.
    """

    color: str = "purple"
    #: 0.0 = empty cup, 1.0 = full. Drives the volume detector.
    fill_level: float = 0.22
    #: Number of bubbles to sprinkle per frame. 0 = a still surface.
    bubbles: int = 0
    #: When True, every render reuses the same bubble seed, so consecutive
    #: frames are identical and the motion detector correctly sees no fizzing.
    frozen_bubbles: bool = True
    seed: int = 0
    #: Kit items on the bench beside the cup. Set to () for a bare-bench scene.
    props: tuple[tuple[str, tuple[int, int, int]], ...] = DEFAULT_PROPS

    def copy(self) -> "BenchState":
        return BenchState(
            color=self.color,
            fill_level=self.fill_level,
            bubbles=self.bubbles,
            frozen_bubbles=self.frozen_bubbles,
            seed=self.seed,
            props=self.props,
        )

    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "fill_level": round(self.fill_level, 3),
            "bubbles": self.bubbles,
            "frozen_bubbles": self.frozen_bubbles,
            "props": [label for label, _ in self.props],
        }


def render_bench(state: BenchState, frame_index: int = 0) -> np.ndarray:
    """Render one BGR frame of the bench for ``state``.

    Deliberately includes an illumination gradient, sensor noise, a specular
    glint and the cup's own rim strokes, because those are exactly the things
    that break naive detectors on real frames.
    """
    rng = np.random.RandomState((state.seed * 7919 + frame_index) % (2**31))
    img = np.zeros((FRAME_H, FRAME_W, 3), np.float64)
    gradient = np.linspace(150, 118, FRAME_H).reshape(FRAME_H, 1)
    img[:, :, 0] = gradient + 8
    img[:, :, 1] = gradient + 4
    img[:, :, 2] = gradient
    img += rng.normal(0, 3.0, img.shape)

    x, y, w, h = CUP_ROI
    cup_left, cup_right = x + 8, x + w - 8
    cup_bottom = y + h - 6
    cup_top = y + 4
    inner_height = cup_bottom - cup_top
    fill_top = int(cup_bottom - max(0.0, min(1.0, state.fill_level)) * inner_height)

    cv2.rectangle(img, (cup_left - 6, cup_top), (cup_right + 6, cup_bottom + 4),
                  (188, 186, 182), thickness=3)

    rgb = INDICATOR_COLORS.get(state.color, INDICATOR_COLORS["purple"])
    bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float64)
    for row in range(fill_top, cup_bottom):
        t = (row - fill_top) / max(1, cup_bottom - fill_top)
        img[row, cup_left:cup_right] = bgr * (1.0 - 0.22 * t)

    if fill_top < cup_bottom:
        cv2.ellipse(img, ((cup_left + cup_right) // 2, fill_top),
                    ((cup_right - cup_left) // 2, 6), 0, 0, 360,
                    tuple(float(v) for v in bgr * 1.25), -1)
        cv2.circle(img, (cup_left + 26, fill_top + 24), 9, (248, 248, 248), -1)

    if state.bubbles:
        bubble_seed = state.seed if state.frozen_bubbles else state.seed + frame_index
        brng = np.random.RandomState((bubble_seed * 104729) % (2**31))
        for _ in range(state.bubbles):
            bx = brng.randint(cup_left + 6, cup_right - 6)
            by = brng.randint(fill_top + 4, max(fill_top + 5, cup_bottom))
            cv2.circle(img, (bx, by), int(brng.randint(2, 6)), (232, 226, 240), -1)

    img += rng.normal(0, 2.5, img.shape)
    out = np.clip(img, 0, 255).astype(np.uint8)

    # Props are drawn after the float buffer is converted: cv2.putText requires
    # an 8-bit image. They carry no sensor noise as a result, which is fine --
    # nothing measures them.
    _draw_props(out, state)
    return out


def _draw_props(img: np.ndarray, state: BenchState) -> None:
    """Draw the kit items to the right of the cup, clear of ``CUP_ROI``.

    Everything here is confined to x >= PROP_ZONE_X, which is well right of the
    verification crop, so no prop can ever influence a colour, motion or volume
    measurement. :func:`test_props_never_touch_the_verification_roi` enforces it.
    """
    if not state.props:
        return

    columns = 3
    cell_w, cell_h = 115, 88
    origin_x, origin_y = PROP_ZONE_X, 40

    for index, (label, rgb) in enumerate(state.props):
        col, row = index % columns, index // columns
        cx = origin_x + col * cell_w
        cy = origin_y + row * cell_h
        if cx + 96 > FRAME_W or cy + 70 > FRAME_H:
            continue  # ran out of bench; skip rather than draw off-frame

        bgr = (float(rgb[2]), float(rgb[1]), float(rgb[0]))
        # Tub body with a darker rim, so it reads as an object rather than a patch.
        cv2.rectangle(img, (cx, cy), (cx + 92, cy + 46), bgr, -1)
        cv2.rectangle(img, (cx, cy), (cx + 92, cy + 46), (96, 94, 98), 2)
        cv2.line(img, (cx, cy + 12), (cx + 92, cy + 12), (128, 126, 130), 1)

        cv2.putText(img, label, (cx + 3, cy + 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (46, 44, 50), 1, cv2.LINE_AA)


class MockCamera:
    """Renders frames from a shared :class:`BenchState`.

    The orchestrator holds the same BenchState object and mutates it as skills
    execute, so ``capture()`` always reflects the current bench.
    """

    def __init__(self, state: BenchState | None = None) -> None:
        self.state = state if state is not None else BenchState()
        self.captures = 0

    def capture(self) -> np.ndarray:
        frame = render_bench(self.state, frame_index=self.captures)
        self.captures += 1
        return frame

    def capture_window(self, n_frames: int, interval_s: float = 0.2) -> Sequence[np.ndarray]:
        # interval_s is accepted for interface parity; the mock does not sleep.
        frames = []
        for _ in range(max(2, n_frames)):
            frames.append(self.capture())
        return frames


class FixtureCamera:
    """Replays a fixed list of image files as frames.

    Used to force a specific verification outcome from static fixtures -- most
    importantly the forced-failure case that must trigger real replanning.
    """

    def __init__(self, frames: Sequence[np.ndarray]) -> None:
        if not frames:
            raise ValueError("FixtureCamera needs at least one frame")
        self._frames = list(frames)
        self.captures = 0

    def capture(self) -> np.ndarray:
        frame = self._frames[min(self.captures, len(self._frames) - 1)]
        self.captures += 1
        return frame

    def capture_window(self, n_frames: int, interval_s: float = 0.2) -> Sequence[np.ndarray]:
        return [self.capture() for _ in range(max(2, n_frames))]


def make_rng(seed: int | None) -> random.Random:
    """Seeded RNG for the fake-success skill executor, so runs reproduce."""
    return random.Random(seed)
