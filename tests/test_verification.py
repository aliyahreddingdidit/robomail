"""Tests for the Chemistry Outcome Verification Agent (CLAUDE.md item 7).

Every detector is exercised against STATIC IMAGE FIXTURES on disk -- the
generated scenes from ``make_fixtures.py`` and real crops from the chemistry-kit
booklet photographs. Nothing here builds arrays inline and calls that a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.verification import (
    DEFAULT_PALETTE,
    ColorShiftDetector,
    EffervescenceDetector,
    VerificationAgent,
    VolumeDeltaDetector,
    load_frame,
)

FIXTURES = Path(__file__).parent / "fixtures"
CUP_ROI = (80, 120, 160, 170)


def fx(name: str):
    return load_frame(FIXTURES / name)


# --------------------------------------------------------------------------
# Detector 1: colour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("swatch", DEFAULT_PALETTE, ids=lambda s: s.label)
def test_every_reference_swatch_classifies_as_itself(swatch, tmp_path):
    """A rendered patch of a reference colour must classify back to that swatch."""
    import cv2
    import numpy as np

    patch = np.full((80, 80, 3), (swatch.rgb[2], swatch.rgb[1], swatch.rgb[0]), np.uint8)
    path = tmp_path / f"{swatch.label}.png"
    cv2.imwrite(str(path), patch)

    result = ColorShiftDetector().classify(load_frame(path))
    assert result.label == swatch.label
    assert result.ph_class == swatch.ph_class
    assert result.trusted


def test_color_classifies_fixture_scenes():
    detector = ColorShiftDetector()
    assert detector.classify(fx("color_before_purple.png"), CUP_ROI).ph_class == "neutral"
    assert detector.classify(fx("color_after_basic_blue.png"), CUP_ROI).ph_class == "basic"
    assert detector.classify(fx("color_after_acidic_red.png"), CUP_ROI).ph_class == "acidic"


def test_color_verifies_a_basic_shift():
    result = ColorShiftDetector().verify(
        fx("color_before_purple.png"), fx("color_after_basic_blue.png"),
        roi=CUP_ROI, expected_ph_class="basic",
    )
    assert result.success
    assert result.modality == "color"
    assert result.metrics["hue_shift_deg"] > 12
    assert "basic" in result.reason


def test_color_verifies_an_acidic_shift():
    """The same procedure must generalise to the acid direction, not memorise one reagent."""
    result = ColorShiftDetector().verify(
        fx("color_before_purple.png"), fx("color_after_acidic_red.png"),
        roi=CUP_ROI, expected_ph_class="acidic",
    )
    assert result.success
    assert result.metrics["after"]["ph_class"] == "acidic"


def test_color_reports_failure_when_nothing_shifted():
    """The forced-failure fixture that drives closed-loop replanning."""
    result = ColorShiftDetector().verify(
        fx("color_before_purple.png"), fx("color_after_unchanged.png"),
        roi=CUP_ROI, expected_ph_class="basic",
    )
    assert not result.success
    assert result.metrics["hue_shift_deg"] < 12
    assert result.reason  # a non-empty reason string is fed back to the planner


def test_color_no_change_is_success_for_a_negative_control():
    """Experiment D: 'no reaction, and that is correct' must not read as failure."""
    result = ColorShiftDetector().verify(
        fx("color_before_purple.png"), fx("color_after_unchanged.png"),
        roi=CUP_ROI, expect_change=False,
    )
    assert result.success
    assert "as expected" in result.reason


def test_color_change_is_failure_for_a_negative_control():
    result = ColorShiftDetector().verify(
        fx("color_before_purple.png"), fx("color_after_basic_blue.png"),
        roi=CUP_ROI, expect_change=False,
    )
    assert not result.success


def test_color_classifies_a_real_photograph_of_cabbage_indicator():
    """Real camera pixels, not a rendered scene: the booklet's indicator photo."""
    result = ColorShiftDetector().classify(fx("real_indicator_purple.png"))
    assert result.label == "purple"
    assert result.ph_class == "neutral"
    assert result.trusted


def test_color_discriminates_two_real_photographs():
    """The acidic cup and the neutral indicator must not collapse to one class."""
    detector = ColorShiftDetector()
    acidic = detector.classify(fx("real_acid_cup_red.png"))
    neutral = detector.classify(fx("real_indicator_purple.png"))
    assert acidic.ph_class == "acidic"
    assert neutral.ph_class == "neutral"
    assert acidic.label != neutral.label


# --------------------------------------------------------------------------
# Detector 2: effervescence
# --------------------------------------------------------------------------


def fizz_frames():
    return [fx(f"fizz_{i:02d}.png") for i in range(8)]


def still_frames():
    return [fx(f"still_{i:02d}.png") for i in range(8)]


def test_effervescence_detected_in_a_bubbling_window():
    result = EffervescenceDetector().verify(fizz_frames(), roi=CUP_ROI)
    assert result.success
    assert result.modality == "motion"
    assert result.metrics["peak_motion_fraction"] >= result.metrics["threshold"]
    assert result.metrics["active_intervals"] > 0


def test_no_effervescence_in_a_still_window():
    result = EffervescenceDetector().verify(still_frames(), roi=CUP_ROI)
    assert not result.success
    assert result.metrics["peak_motion_fraction"] < result.metrics["threshold"]


def test_still_window_is_success_when_no_reaction_is_expected():
    result = EffervescenceDetector().verify(still_frames(), roi=CUP_ROI, expect_change=False)
    assert result.success


def test_effervescence_needs_at_least_two_frames():
    with pytest.raises(ValueError):
        EffervescenceDetector().motion_profile([fx("fizz_00.png")], roi=CUP_ROI)


def test_motion_profile_has_one_entry_per_interval():
    profile = EffervescenceDetector().motion_profile(fizz_frames(), roi=CUP_ROI)
    assert len(profile) == 7
    assert all(0.0 <= p <= 1.0 for p in profile)


# --------------------------------------------------------------------------
# Detector 3: volume
# --------------------------------------------------------------------------


def test_volume_measures_a_low_and_a_high_fill_differently():
    detector = VolumeDeltaDetector()
    low = detector.content_height_px(fx("snow_before.png"), CUP_ROI)
    high = detector.content_height_px(fx("snow_after.png"), CUP_ROI)
    assert 0 < low < high
    assert high > 2 * low


def test_volume_detects_instant_snow_expansion():
    result = VolumeDeltaDetector().verify(
        fx("snow_before.png"), fx("snow_after.png"), roi=CUP_ROI
    )
    assert result.success
    assert result.modality == "volume"
    assert result.metrics["growth_fraction"] > 0.2


def test_volume_reports_failure_when_nothing_expanded():
    result = VolumeDeltaDetector().verify(
        fx("sand_before.png"), fx("sand_after.png"), roi=CUP_ROI
    )
    assert not result.success


def test_hydrophobic_sand_no_change_is_the_correct_outcome():
    """Experiment D again, on the volume modality."""
    result = VolumeDeltaDetector().verify(
        fx("sand_before.png"), fx("sand_after.png"), roi=CUP_ROI, expect_change=False
    )
    assert result.success
    assert "as expected" in result.reason


def test_volume_is_not_fooled_by_the_bench_illumination_gradient():
    """The crop has a vertical light gradient and the cup's own rim strokes in it.

    A naive flat-background threshold reads the whole crop as contents. The
    measured height must stay well under the crop height.
    """
    detector = VolumeDeltaDetector()
    crop_height = CUP_ROI[3]
    height = detector.content_height_px(fx("snow_before.png"), CUP_ROI)
    assert height < crop_height * 0.5


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


def test_agent_routes_each_modality():
    agent = VerificationAgent()
    colour = agent.verify("color", [fx("color_before_purple.png"), fx("color_after_basic_blue.png")],
                          roi=CUP_ROI, expected_ph_class="basic")
    motion = agent.verify("motion", fizz_frames(), roi=CUP_ROI)
    volume = agent.verify("volume", [fx("snow_before.png"), fx("snow_after.png")], roi=CUP_ROI)
    assert (colour.modality, motion.modality, volume.modality) == ("color", "motion", "volume")
    assert colour.success and motion.success and volume.success


def test_agent_rejects_an_unknown_modality():
    with pytest.raises(ValueError, match="unknown verification modality"):
        VerificationAgent().verify("smell", [fx("snow_before.png"), fx("snow_after.png")])


def test_agent_requires_a_before_and_an_after():
    with pytest.raises(ValueError):
        VerificationAgent().verify("color", [fx("snow_before.png")])


def test_results_serialise_for_the_trial_log():
    result = VerificationAgent().verify(
        "color", [fx("color_before_purple.png"), fx("color_after_basic_blue.png")],
        roi=CUP_ROI, expected_ph_class="basic",
    )
    payload = result.as_dict()
    assert set(payload) == {"success", "reason", "modality", "metrics", "confidence"}
    import json

    json.dumps(payload)  # must be JSON-serialisable for the JSONL log
