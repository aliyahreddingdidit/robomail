"""Real hardware adapters (CLAUDE.md architecture item 11).

Thin wrappers that present PLATO's frankapy arm and RealSense cameras through
:mod:`hardware.interfaces`. Every hardware import is deferred to construction
time so that importing this module on a laptop with no robot, no frankapy and no
pyrealsense2 still works -- which is what lets the rest of the pipeline be
imported and tested headless.

Upstream PLATO drives the arm from `exec_script.py`'s `run_command`, which
converts an LLM primitive into a frankapy pose command. The same conversion
lives here, behind the interface, so nothing above this layer knows which cell
it is talking to.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

CM_TO_M = 0.01


class RealFrankaArm:
    """Adapter over frankapy's FrankaArm."""

    def __init__(self, position_lookup: dict[str, np.ndarray] | None = None) -> None:
        try:
            from frankapy import FrankaArm  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - requires the real cell
            raise RuntimeError(
                "frankapy is not installed, so the real arm is unavailable. Run with "
                "PLATO_HARDWARE=mock to use MockFrankaArm, or install frankapy on the "
                "robot control PC."
            ) from exc
        self._fa = FrankaArm()
        self._positions = position_lookup or {}
        self._holding: str | None = None

    def reset_joints(self) -> None:
        self._fa.reset_joints()

    def goto_delta(self, location: str, delta_cm: tuple[float, float, float]) -> None:
        base = self._positions.get(location)
        if base is None:
            raise KeyError(
                f"no calibrated pose for workspace position {location!r}; the position "
                "lookup is populated by the perception stack at run start"
            )
        pose = self._fa.get_pose()
        pose.translation = np.asarray(base, dtype=float) + np.asarray(delta_cm, float) * CM_TO_M
        self._fa.goto_pose(pose)

    def set_gripper(self, closed: int) -> None:
        if closed:
            self._fa.close_gripper()
        else:
            self._fa.open_gripper()

    def tilt(self, theta_deg: tuple[float, float, float]) -> None:
        from autolab_core import RigidTransform  # type: ignore[import-not-found]

        rx, ry, rz = (float(np.deg2rad(v)) for v in theta_deg)
        pose = self._fa.get_pose()
        pose.rotation = pose.rotation @ RigidTransform.rotation_from_euler(
            rx, ry, rz
        )
        self._fa.goto_pose(pose)

    @property
    def holding(self) -> str | None:
        return self._holding

    def acquire(self, object_name: str) -> None:
        self._holding = object_name

    def release(self) -> str | None:
        released, self._holding = self._holding, None
        return released


class RealSenseCamera:
    """Adapter over a RealSense camera, matching :class:`CameraInterface`."""

    def __init__(self, serial: str | None = None) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - requires the real cell
            raise RuntimeError(
                "pyrealsense2 is not installed, so the real camera is unavailable. Run "
                "with PLATO_HARDWARE=mock to use MockCamera."
            ) from exc
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(config)

    def capture(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("camera returned no colour frame")
        return np.asanyarray(color.get_data())

    def capture_window(self, n_frames: int, interval_s: float = 0.2) -> Sequence[np.ndarray]:
        import time

        frames = []
        for i in range(max(2, n_frames)):
            frames.append(self.capture())
            if i < n_frames - 1:
                time.sleep(interval_s)
        return frames

    def close(self) -> None:
        self._pipeline.stop()
