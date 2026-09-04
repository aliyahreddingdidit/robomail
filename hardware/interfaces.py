"""Hardware interfaces (CLAUDE.md architecture item 11).

The pipeline talks to the arm and cameras only through these two Protocols, so
the mock cell and the real frankapy cell are interchangeable and every one of
items 1-10 runs and is verifiable without robot access.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class ArmInterface(Protocol):
    """The subset of frankapy's FrankaArm this pipeline actually uses."""

    def reset_joints(self) -> None:
        """Return to the home pose."""

    def goto_delta(self, location: str, delta_cm: tuple[float, float, float]) -> None:
        """Move the end effector to ``location`` offset by ``delta_cm``."""

    def set_gripper(self, closed: int) -> None:
        """Close (1) or open (0) the parallel-plate gripper."""

    def tilt(self, theta_deg: tuple[float, float, float]) -> None:
        """Apply a relative roll/pitch/yaw in degrees."""

    @property
    def holding(self) -> str | None:
        """Name of the object currently held, or None."""


class CameraInterface(Protocol):
    """A single camera view of the workspace."""

    def capture(self) -> np.ndarray:
        """Return one BGR frame."""

    def capture_window(self, n_frames: int, interval_s: float = 0.2) -> Sequence[np.ndarray]:
        """Return ``n_frames`` BGR frames, oldest first.

        Used by the Verification Agent's motion modality, which needs an
        observation window rather than a single frame.
        """
