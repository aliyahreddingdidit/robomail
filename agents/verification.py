"""Chemistry Outcome Verification Agent (CLAUDE.md architecture item 7). NEW.

Reads a chemical outcome back into the planning loop. This is the delta over base
PLATO: PLATO verifies that the gripper reached a pose; this verifies that the
*chemistry* happened.

Three real detectors, one per modality in the research plan:

  color  -- :class:`ColorShiftDetector`. Mean ROI colour classified against a set
            of reference swatches by circular hue distance in HSV, with a
            saturation/value tie-break. Experiment A (cabbage indicator).
  motion -- :class:`EffervescenceDetector`. OpenCV frame-differencing across a
            short observation window; the metric is the fraction of ROI pixels
            that change between consecutive frames. Experiment B (fizzing).
  volume -- :class:`VolumeDeltaDetector`. Height-of-contents measured from a
            fixed crop by finding the fill line against the background sampled
            from the top of the crop. Experiment C (instant snow).

All three take a ``expect_change`` flag, because Experiment D (hydrophobic sand)
is a demonstration whose SUCCESS condition is that nothing changed. Distinguishing
"no reaction, and that is correct" from "no reaction, and that is a failure" is a
first-class case here, not an afterthought.

Note on calibration: the reference swatch RGB values below are nominal starting
points. Photographing real swatches under lab lighting is a parallel human task
(explicitly out of scope for this pass) -- when those photos exist, drop them in
as a JSON palette via :func:`load_palette` and nothing else needs to change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

AGENT_NAME = "ChemistryOutcomeVerificationAgent"

Frame = np.ndarray  # BGR uint8, as produced by OpenCV and the camera interface
ROI = tuple[int, int, int, int]  # (x, y, w, h)


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Success/failure plus the reason string fed back to the Planner."""

    success: bool
    reason: str
    modality: str
    metrics: dict = field(default_factory=dict)
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "reason": self.reason,
            "modality": self.modality,
            "metrics": self.metrics,
            "confidence": round(self.confidence, 4),
        }


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _as_bgr(frame: Frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR frame, got shape {arr.shape}")
    return arr.astype(np.uint8)


def crop(frame: Frame, roi: ROI | None) -> np.ndarray:
    """Crop to ``roi``; the whole frame when roi is None. Clamps to bounds."""
    img = _as_bgr(frame)
    if roi is None:
        return img
    x, y, w, h = (int(v) for v in roi)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} does not overlap a {img.shape[1]}x{img.shape[0]} frame")
    return img[y0:y1, x0:x1]


def load_frame(path: str | Path) -> np.ndarray:
    """Read an image file as a BGR frame."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def _hue_distance(h1: float, h2: float) -> float:
    """Circular distance between two OpenCV hues (0-179), returned in 0-90."""
    diff = abs(float(h1) - float(h2)) % 180.0
    return min(diff, 180.0 - diff)


def mean_solution_color(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean BGR and HSV of the coloured liquid in ``patch``.

    Specular highlights (near-white) and deep shadow (near-black) are masked out
    before averaging -- on a glossy cup of liquid those pixels carry no hue
    information and would drag the mean toward grey. Returns
    ``(mean_bgr, mean_hsv, coverage)`` where coverage is the fraction of pixels
    that survived masking.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    mask = (v > 35) & (v < 245) & (s > 40)
    coverage = float(mask.mean())
    if coverage < 0.02:
        # Almost nothing chromatic in the ROI -- fall back to an unmasked mean so
        # the caller still gets a defined answer, and report the low coverage.
        mask = np.ones_like(mask, dtype=bool)
        coverage = float(mask.mean()) * 0.0

    sel = mask.reshape(-1)
    bgr_mean = patch.reshape(-1, 3)[sel].mean(axis=0)

    # Hue is circular: average it as a unit vector, not arithmetically, or a
    # red solution straddling the 0/180 wrap averages to cyan.
    hue = hsv[:, :, 0].reshape(-1)[sel].astype(np.float64) * (2.0 * math.pi / 180.0)
    hue_mean = (math.degrees(math.atan2(np.sin(hue).mean(), np.cos(hue).mean())) / 2.0) % 180.0
    sat_mean = float(hsv[:, :, 1].reshape(-1)[sel].mean())
    val_mean = float(hsv[:, :, 2].reshape(-1)[sel].mean())
    return bgr_mean, np.array([hue_mean, sat_mean, val_mean]), coverage


# --------------------------------------------------------------------------
# Detector 1: colour classification against reference swatches
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceSwatch:
    """One reference colour for the red-cabbage pH indicator."""

    label: str
    rgb: tuple[int, int, int]
    ph_class: str  # "acidic" | "neutral" | "basic"

    @property
    def hsv(self) -> np.ndarray:
        bgr = np.uint8([[[self.rgb[2], self.rgb[1], self.rgb[0]]]])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float64)


#: Nominal red-cabbage indicator palette. Replace with photographed swatches
#: under lab lighting (parallel human task) via :func:`load_palette`.
DEFAULT_PALETTE: tuple[ReferenceSwatch, ...] = (
    ReferenceSwatch("red", (209, 43, 60), "acidic"),
    ReferenceSwatch("pink", (200, 62, 122), "acidic"),
    ReferenceSwatch("purple", (123, 63, 158), "neutral"),
    ReferenceSwatch("blue", (59, 91, 196), "basic"),
    ReferenceSwatch("green", (79, 158, 84), "basic"),
    ReferenceSwatch("yellow-green", (196, 196, 63), "basic"),
)

#: Hue distance (degrees, 0-90) beyond which a classification is not trusted.
MAX_TRUSTED_HUE_DISTANCE = 28.0


def load_palette(path: str | Path) -> tuple[ReferenceSwatch, ...]:
    """Load a swatch palette from JSON: ``[{label, rgb: [r,g,b], ph_class}, ...]``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        ReferenceSwatch(
            label=str(entry["label"]),
            rgb=(int(entry["rgb"][0]), int(entry["rgb"][1]), int(entry["rgb"][2])),
            ph_class=str(entry["ph_class"]),
        )
        for entry in data
    )


@dataclass
class ColorClassification:
    """Nearest-swatch result for one frame."""

    label: str
    ph_class: str
    hue_distance: float
    mean_rgb: tuple[int, int, int]
    coverage: float
    trusted: bool

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "ph_class": self.ph_class,
            "hue_distance": round(self.hue_distance, 2),
            "mean_rgb": list(self.mean_rgb),
            "coverage": round(self.coverage, 4),
            "trusted": self.trusted,
        }


class ColorShiftDetector:
    """Classify solution colour against reference swatches (Experiment A)."""

    modality = "color"

    def __init__(
        self,
        palette: Sequence[ReferenceSwatch] = DEFAULT_PALETTE,
        *,
        max_trusted_hue_distance: float = MAX_TRUSTED_HUE_DISTANCE,
    ) -> None:
        if not palette:
            raise ValueError("palette must contain at least one reference swatch")
        self.palette = tuple(palette)
        self.max_trusted_hue_distance = float(max_trusted_hue_distance)

    def classify(self, frame: Frame, roi: ROI | None = None) -> ColorClassification:
        """Nearest reference swatch to the mean solution colour in ``roi``."""
        patch = crop(frame, roi)
        mean_bgr, mean_hsv, coverage = mean_solution_color(patch)
        hue = float(mean_hsv[0])
        sat = float(mean_hsv[1])

        scored: list[tuple[float, ReferenceSwatch]] = []
        for swatch in self.palette:
            ref = swatch.hsv
            d_hue = _hue_distance(hue, float(ref[0]))
            # Saturation acts as a tie-break: a washed-out pale sample should
            # prefer a pale reference over a vivid one at the same hue.
            d_sat = abs(sat - float(ref[1])) / 255.0 * 6.0
            scored.append((d_hue + d_sat, swatch))
        scored.sort(key=lambda pair: pair[0])
        score, best = scored[0]
        d_hue_best = _hue_distance(hue, float(best.hsv[0]))

        return ColorClassification(
            label=best.label,
            ph_class=best.ph_class,
            hue_distance=d_hue_best,
            mean_rgb=(int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])),
            coverage=coverage,
            trusted=d_hue_best <= self.max_trusted_hue_distance and coverage > 0.02,
        )

    def verify(
        self,
        before: Frame,
        after: Frame,
        *,
        roi: ROI | None = None,
        expected_ph_class: str | None = None,
        expected_labels: Iterable[str] | None = None,
        expect_change: bool = True,
        min_hue_shift: float = 12.0,
    ) -> VerificationResult:
        """Verify a colour outcome from a before/after frame pair.

        ``expected_ph_class`` / ``expected_labels`` state what the plan expected.
        With ``expect_change=False`` (Experiment D style) the success condition
        inverts: the colour must have stayed put.
        """
        b = self.classify(before, roi)
        a = self.classify(after, roi)
        shift = _hue_distance(
            cv2.cvtColor(np.uint8([[[b.mean_rgb[2], b.mean_rgb[1], b.mean_rgb[0]]]]),
                         cv2.COLOR_BGR2HSV)[0, 0][0],
            cv2.cvtColor(np.uint8([[[a.mean_rgb[2], a.mean_rgb[1], a.mean_rgb[0]]]]),
                         cv2.COLOR_BGR2HSV)[0, 0][0],
        )
        metrics = {
            "before": b.as_dict(),
            "after": a.as_dict(),
            "hue_shift_deg": round(float(shift), 2),
            "min_hue_shift_deg": min_hue_shift,
            "expect_change": expect_change,
        }

        if not a.trusted:
            return VerificationResult(
                success=False,
                reason=(
                    f"colour reading not trustworthy: nearest swatch '{a.label}' is "
                    f"{a.hue_distance:.1f} deg away (limit {self.max_trusted_hue_distance:.0f}) "
                    f"with {a.coverage:.0%} usable pixels. Check the cup is in the ROI and lit."
                ),
                modality=self.modality,
                metrics=metrics,
                confidence=0.2,
            )

        if not expect_change:
            changed = shift >= min_hue_shift or a.ph_class != b.ph_class
            return VerificationResult(
                success=not changed,
                reason=(
                    f"colour changed from {b.label} to {a.label} ({shift:.1f} deg shift) "
                    "but this demonstration expected no change"
                    if changed
                    else f"colour stayed {a.label} ({shift:.1f} deg shift), as expected for "
                    "a no-reaction demonstration"
                ),
                modality=self.modality,
                metrics=metrics,
                confidence=0.85 if not changed else 0.8,
            )

        wanted_labels = {s.lower() for s in (expected_labels or ())}
        if wanted_labels:
            ok = a.label.lower() in wanted_labels
            target = "/".join(sorted(wanted_labels))
        elif expected_ph_class:
            ok = a.ph_class.lower() == expected_ph_class.lower()
            target = expected_ph_class
        else:
            ok = shift >= min_hue_shift
            target = f"any shift of at least {min_hue_shift:.0f} deg"

        if ok and shift < min_hue_shift and (wanted_labels or expected_ph_class):
            # It already read as the target before we did anything -- that is not
            # evidence our step worked, so report it rather than claiming success.
            return VerificationResult(
                success=False,
                reason=(
                    f"solution already read as {a.label} ({a.ph_class}) before the step "
                    f"and shifted only {shift:.1f} deg; no evidence the reagent acted"
                ),
                modality=self.modality,
                metrics=metrics,
                confidence=0.6,
            )

        return VerificationResult(
            success=ok,
            reason=(
                f"solution went {b.label} -> {a.label} ({a.ph_class}), {shift:.1f} deg hue shift; "
                f"expected {target}"
                if ok
                else f"solution reads {a.label} ({a.ph_class}) after only a {shift:.1f} deg shift "
                f"from {b.label}; expected {target}. Colour has not moved far enough."
            ),
            modality=self.modality,
            metrics=metrics,
            confidence=0.9 if ok else 0.85,
        )


# --------------------------------------------------------------------------
# Detector 2: effervescence via frame-differencing
# --------------------------------------------------------------------------


class EffervescenceDetector:
    """Detect bubbling/fizzing as inter-frame motion (Experiment B)."""

    modality = "motion"

    def __init__(
        self,
        *,
        pixel_threshold: int = 18,
        motion_fraction_threshold: float = 0.02,
        blur_ksize: int = 5,
    ) -> None:
        self.pixel_threshold = int(pixel_threshold)
        self.motion_fraction_threshold = float(motion_fraction_threshold)
        self.blur_ksize = int(blur_ksize) | 1  # force odd

    def motion_profile(self, frames: Sequence[Frame], roi: ROI | None = None) -> list[float]:
        """Fraction of ROI pixels changing between each consecutive frame pair."""
        if len(frames) < 2:
            raise ValueError("effervescence detection needs at least 2 frames")
        greys = []
        for f in frames:
            g = cv2.cvtColor(crop(f, roi), cv2.COLOR_BGR2GRAY)
            greys.append(cv2.GaussianBlur(g, (self.blur_ksize, self.blur_ksize), 0))
        shape = greys[0].shape
        if any(g.shape != shape for g in greys):
            raise ValueError("all frames must have the same ROI dimensions")

        profile: list[float] = []
        for prev, nxt in zip(greys, greys[1:]):
            diff = cv2.absdiff(prev, nxt)
            _, mask = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
            # Opening removes single-pixel sensor noise while keeping bubble blobs.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            profile.append(float(np.count_nonzero(mask)) / float(mask.size))
        return profile

    def verify(
        self,
        frames: Sequence[Frame],
        *,
        roi: ROI | None = None,
        expect_change: bool = True,
    ) -> VerificationResult:
        """Verify effervescence over a short observation window."""
        profile = self.motion_profile(frames, roi)
        peak = max(profile)
        mean = float(np.mean(profile))
        active = sum(1 for p in profile if p >= self.motion_fraction_threshold)
        metrics = {
            "frame_count": len(frames),
            "motion_profile": [round(p, 5) for p in profile],
            "peak_motion_fraction": round(peak, 5),
            "mean_motion_fraction": round(mean, 5),
            "active_intervals": active,
            "threshold": self.motion_fraction_threshold,
            "expect_change": expect_change,
        }
        fizzing = peak >= self.motion_fraction_threshold

        if not expect_change:
            return VerificationResult(
                success=not fizzing,
                reason=(
                    f"unexpected motion detected: peak {peak:.3%} of the ROI changed between "
                    f"frames (threshold {self.motion_fraction_threshold:.1%}); this "
                    "demonstration expected no reaction"
                    if fizzing
                    else f"no effervescence detected (peak {peak:.3%} below the "
                    f"{self.motion_fraction_threshold:.1%} threshold), which is the expected "
                    "outcome for this demonstration"
                ),
                modality=self.modality,
                metrics=metrics,
                confidence=0.85,
            )

        return VerificationResult(
            success=fizzing,
            reason=(
                f"effervescence detected: peak {peak:.2%} of the ROI changed between frames "
                f"across {active}/{len(profile)} intervals (threshold "
                f"{self.motion_fraction_threshold:.1%})"
                if fizzing
                else f"no effervescence: peak inter-frame change was only {peak:.3%} of the ROI, "
                f"below the {self.motion_fraction_threshold:.1%} threshold. The reagents may "
                "not have contacted, or a required component (e.g. water) is missing."
            ),
            modality=self.modality,
            metrics=metrics,
            confidence=0.9 if fizzing else 0.85,
        )


# --------------------------------------------------------------------------
# Detector 3: contents height / volume delta
# --------------------------------------------------------------------------


class VolumeDeltaDetector:
    """Measure height-of-contents change in a fixed crop (Experiment C)."""

    modality = "volume"

    def __init__(
        self,
        *,
        background_strip_frac: float = 0.15,
        content_threshold: int = 30,
        min_row_coverage: float = 0.30,
        min_growth_frac: float = 0.20,
    ) -> None:
        self.background_strip_frac = float(background_strip_frac)
        self.content_threshold = int(content_threshold)
        self.min_row_coverage = float(min_row_coverage)
        self.min_growth_frac = float(min_growth_frac)

    def _background_model(self, patch: np.ndarray) -> np.ndarray:
        """Expected background colour for every row of the crop, shape (h, 3).

        The top ``background_strip_frac`` of the crop is assumed to be empty
        space above the fill line. A flat mean over that strip is not enough:
        benches are lit unevenly, and a vertical illumination gradient alone
        will make every lower row look like "contents". So we fit a straight
        line to the strip's per-row channel means and extrapolate it down the
        crop, which cancels a linear gradient and leaves genuine contents as the
        only large residual.

        The fit is doubly robust, because the strip is not pure bench: a fixed
        crop over a cup always contains the rim and the wall. Row *medians* (not
        means) reject the thin vertical wall strokes, and a Theil-Sen slope
        (median of pairwise slopes) rejects the bright horizontal rim, which can
        easily be 20% of the strip. Ordinary least squares is not enough here --
        five contaminated rows out of twenty-five tilt it far enough that the
        whole crop reads as contents.
        """
        h = patch.shape[0]
        strip = max(4, int(h * self.background_strip_frac))
        strip_rows = np.median(patch[:strip], axis=1)  # (strip, 3), median over columns
        idx = np.arange(strip, dtype=np.float64)
        rows = np.arange(h, dtype=np.float64)
        model = np.empty((h, 3), dtype=np.float64)

        i, j = np.triu_indices(strip, k=1)
        dx = idx[j] - idx[i]
        for ch in range(3):
            values = strip_rows[:, ch]
            slope = float(np.median((values[j] - values[i]) / dx))
            intercept = float(np.median(values - slope * idx))
            model[:, ch] = slope * rows + intercept
        return model

    def content_height_px(self, frame: Frame, roi: ROI | None = None) -> int:
        """Height in pixels of the contents column inside the fixed crop.

        A row counts as contents when at least ``min_row_coverage`` of its pixels
        differ from the modelled background by more than ``content_threshold``
        (L1 over BGR). The height is the length of the LONGEST contiguous run of
        such rows -- the liquid or powder column. Taking the longest run rather
        than "everything below the topmost content row" is what stops the cup's
        own rim strokes, which are short runs of their own, from being measured
        as contents.
        """
        patch = crop(frame, roi).astype(np.float64)
        background = self._background_model(patch)  # (h, 3)

        distance = np.abs(patch - background[:, None, :]).sum(axis=2)  # (h, w)
        content = distance > self.content_threshold
        is_content = content.mean(axis=1) >= self.min_row_coverage
        if not is_content.any():
            return 0

        longest = current = 0
        for flag in is_content:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        return int(longest)

    def verify(
        self,
        before: Frame,
        after: Frame,
        *,
        roi: ROI | None = None,
        expect_change: bool = True,
    ) -> VerificationResult:
        """Verify an expansion (or the absence of one) from a before/after pair."""
        h_before = self.content_height_px(before, roi)
        h_after = self.content_height_px(after, roi)
        crop_h = crop(before, roi).shape[0]
        growth = (h_after - h_before) / h_before if h_before > 0 else (
            float("inf") if h_after > 0 else 0.0
        )
        metrics = {
            "height_before_px": h_before,
            "height_after_px": h_after,
            "crop_height_px": crop_h,
            "growth_fraction": None if math.isinf(growth) else round(float(growth), 4),
            "min_growth_fraction": self.min_growth_frac,
            "expect_change": expect_change,
        }
        expanded = growth >= self.min_growth_frac

        if not expect_change:
            return VerificationResult(
                success=not expanded,
                reason=(
                    f"contents grew from {h_before}px to {h_after}px "
                    f"({growth:.0%}) but this demonstration expected no change"
                    if expanded
                    else f"contents height held at {h_before}px -> {h_after}px, as expected "
                    "for a no-reaction demonstration"
                ),
                modality=self.modality,
                metrics=metrics,
                confidence=0.85,
            )

        if h_before == 0 and h_after == 0:
            return VerificationResult(
                success=False,
                reason=(
                    "no contents detected in the crop either before or after; the cup may be "
                    "empty or outside the ROI"
                ),
                modality=self.modality,
                metrics=metrics,
                confidence=0.3,
            )

        pct = "unbounded" if math.isinf(growth) else f"{growth:.0%}"
        return VerificationResult(
            success=expanded,
            reason=(
                f"contents expanded from {h_before}px to {h_after}px ({pct} growth, "
                f"threshold {self.min_growth_frac:.0%})"
                if expanded
                else f"contents barely changed: {h_before}px -> {h_after}px ({pct} growth) "
                f"against a {self.min_growth_frac:.0%} threshold. Not enough water may have "
                "reached the powder."
            ),
            modality=self.modality,
            metrics=metrics,
            confidence=0.9 if expanded else 0.85,
        )


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


class VerificationAgent:
    """Routes a plan's verification modality to the right detector."""

    def __init__(
        self,
        *,
        color: ColorShiftDetector | None = None,
        motion: EffervescenceDetector | None = None,
        volume: VolumeDeltaDetector | None = None,
    ) -> None:
        self.color = color or ColorShiftDetector()
        self.motion = motion or EffervescenceDetector()
        self.volume = volume or VolumeDeltaDetector()

    def verify(
        self,
        modality: str,
        frames: Sequence[Frame],
        *,
        roi: ROI | None = None,
        expect_change: bool = True,
        expected_ph_class: str | None = None,
        expected_labels: Iterable[str] | None = None,
    ) -> VerificationResult:
        """Verify an outcome. ``frames`` is the observation window, oldest first.

        Colour and volume use the first and last frame; motion uses all of them.
        """
        key = (modality or "").strip().lower()
        if len(frames) < 2:
            raise ValueError("verification needs at least a before and an after frame")

        if key == "color":
            return self.color.verify(
                frames[0],
                frames[-1],
                roi=roi,
                expect_change=expect_change,
                expected_ph_class=expected_ph_class,
                expected_labels=expected_labels,
            )
        if key == "motion":
            return self.motion.verify(frames, roi=roi, expect_change=expect_change)
        if key == "volume":
            return self.volume.verify(frames[0], frames[-1], roi=roi, expect_change=expect_change)
        raise ValueError(f"unknown verification modality {modality!r}; expected color/motion/volume")
