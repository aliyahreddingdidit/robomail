"""Mock/real hardware toggle (CLAUDE.md architecture item 11).

Switched by the ``PLATO_HARDWARE`` environment variable or an explicit argument:

    PLATO_HARDWARE=mock   (default)  -- MockFrankaArm + MockCamera, no robot needed
    PLATO_HARDWARE=real              -- frankapy FrankaArm + RealSense camera

The default is deliberately ``mock``: reaching for real hardware should be an
explicit act, not something that happens because an env var was forgotten.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agents.affordance import GraspProvider, MockGraspProvider
from hardware.interfaces import ArmInterface, CameraInterface
from hardware.mock import BenchState, MockCamera, MockFrankaArm

ENV_HARDWARE = "PLATO_HARDWARE"
MOCK = "mock"
REAL = "real"


@dataclass
class Cell:
    """One assembled robot cell: arm, camera, grasp provider."""

    arm: ArmInterface
    camera: CameraInterface
    grasp_provider: GraspProvider
    mode: str
    bench_state: BenchState | None = None

    @property
    def is_mock(self) -> bool:
        return self.mode == MOCK

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "arm": type(self.arm).__name__,
            "camera": type(self.camera).__name__,
            "grasp_provider": type(self.grasp_provider).__name__,
            "bench_state": self.bench_state.as_dict() if self.bench_state else None,
        }


def resolve_mode(mode: str | None = None) -> str:
    """Resolve the hardware mode from an argument, then the env var, then mock."""
    chosen = (mode or os.environ.get(ENV_HARDWARE, "") or MOCK).strip().lower()
    if chosen not in (MOCK, REAL):
        raise ValueError(
            f"{ENV_HARDWARE}={chosen!r} is not valid; expected {MOCK!r} or {REAL!r}"
        )
    return chosen


def build_cell(
    mode: str | None = None,
    *,
    bench_state: BenchState | None = None,
    camera: CameraInterface | None = None,
) -> Cell:
    """Assemble the arm, camera and grasp provider for the chosen mode."""
    chosen = resolve_mode(mode)

    if chosen == MOCK:
        state = bench_state or BenchState()
        return Cell(
            arm=MockFrankaArm(),
            camera=camera or MockCamera(state),
            grasp_provider=MockGraspProvider(),
            mode=MOCK,
            bench_state=state,
        )

    # Imported lazily: hardware.real pulls in frankapy/pyrealsense2 on construction.
    from agents.affordance import PlatoGraspProvider
    from hardware.real import RealFrankaArm, RealSenseCamera

    arm = RealFrankaArm()
    cam = camera or RealSenseCamera()
    return Cell(
        arm=arm,
        camera=cam,
        grasp_provider=PlatoGraspProvider(save_path="data/grasps", gripper_cam=cam, arm=arm),
        mode=REAL,
    )
