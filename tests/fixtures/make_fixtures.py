"""Generate the static image fixtures the Verification Agent tests run against.

Run once; the PNGs it writes are committed and the tests read them from disk, so
the detectors are exercised against static image pairs exactly as CLAUDE.md
requires (item 7) rather than against arrays built inside the test.

Two kinds of fixture:

  * Real photographs -- crops from the chemistry-kit booklet photos already in
    this project (IMG_5808.jpeg, IMG_5810.jpeg). ``real_indicator_purple.png`` is
    a genuine photograph of red-cabbage indicator; ``real_acid_cup_red.png`` is
    the printed acidic-solution cup. These are the ground truth for "does the
    classifier work on a real camera frame at all".

  * Synthetic scenes -- rendered cups with known ground truth, used where a real
    photograph does not exist yet (nobody has filmed the robot fizzing a cup).
    They are rendered from a fixed seed so the fixtures are reproducible, and
    they are deliberately noisy, shaded and specular so the detectors have to
    cope with the same things that break naive thresholding on real frames.

Regenerate with:  python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

W, H = 320, 320
CUP_ROI = (80, 120, 160, 170)  # (x, y, w, h) -- the fixed camera crop over the cup

#: ROIs into the booklet photographs, measured once by inspection.
BOOKLET_CROPS = {
    "real_indicator_purple.png": ("IMG_5810.jpeg", (4169, 2113, 714, 1142)),
    "real_acid_cup_red.png": ("IMG_5810.jpeg", (2739, 1413, 189, 92)),
    "real_booklet_page_12.png": ("IMG_5808.jpeg", None),
    "real_booklet_page_13.png": ("IMG_5810.jpeg", None),
}


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _base_scene(rng: np.random.RandomState) -> np.ndarray:
    """Bench background with a vertical light gradient and sensor noise."""
    img = np.zeros((H, W, 3), np.float64)
    gradient = np.linspace(150, 118, H).reshape(H, 1)
    img[:, :, 0] = gradient + 8
    img[:, :, 1] = gradient + 4
    img[:, :, 2] = gradient
    img += rng.normal(0, 3.0, img.shape)
    return img


def _draw_cup(img: np.ndarray, fill_top: int, rgb: tuple[int, int, int],
              rng: np.random.RandomState) -> np.ndarray:
    """Draw a cup whose liquid surface sits at row ``fill_top``.

    The cup body is drawn above and below the fill line so the volume detector
    has to find the *contents*, not the cup.
    """
    x, y, w, h = CUP_ROI
    cup_left, cup_right = x + 8, x + w - 8
    cup_bottom = y + h - 6

    out = img.copy()
    # Cup wall: pale translucent grey, drawn over the full cup height.
    cv2.rectangle(out, (cup_left - 6, y + 4), (cup_right + 6, cup_bottom + 4),
                  (188, 186, 182), thickness=3)

    # Liquid body, with a slight vertical shade so it is not a flat colour.
    bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float64)
    for row in range(fill_top, cup_bottom):
        t = (row - fill_top) / max(1, cup_bottom - fill_top)
        shade = 1.0 - 0.22 * t
        out[row, cup_left:cup_right] = bgr * shade

    # Meniscus highlight and a specular glint -- these are exactly the pixels the
    # colour detector must mask out before averaging.
    cv2.ellipse(out, ((cup_left + cup_right) // 2, fill_top),
                ((cup_right - cup_left) // 2, 6), 0, 0, 360,
                tuple(float(v) for v in bgr * 1.25), -1)
    cv2.circle(out, (cup_left + 26, fill_top + 24), 9, (248, 248, 248), -1)

    out += rng.normal(0, 2.5, out.shape)
    return out


def _finish(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _write(name: str, img: np.ndarray) -> None:
    cv2.imwrite(str(HERE / name), img)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------
# Colour fixtures (Experiment A)
# --------------------------------------------------------------------------

INDICATOR_PURPLE = (123, 63, 158)
INDICATOR_BLUE = (59, 91, 196)
INDICATOR_RED = (209, 43, 60)


def make_color_fixtures() -> None:
    print("colour fixtures:")
    fill_top = CUP_ROI[1] + 45
    _write("color_before_purple.png",
           _finish(_draw_cup(_base_scene(_rng(1)), fill_top, INDICATOR_PURPLE, _rng(2))))
    _write("color_after_basic_blue.png",
           _finish(_draw_cup(_base_scene(_rng(1)), fill_top, INDICATOR_BLUE, _rng(3))))
    _write("color_after_acidic_red.png",
           _finish(_draw_cup(_base_scene(_rng(1)), fill_top, INDICATOR_RED, _rng(4))))
    # The forced-failure fixture: the reagent did not act, colour barely moved.
    # This is what drives the closed-loop replanning test (item 9).
    _write("color_after_unchanged.png",
           _finish(_draw_cup(_base_scene(_rng(1)), fill_top, (127, 66, 154), _rng(5))))


# --------------------------------------------------------------------------
# Effervescence fixtures (Experiment B)
# --------------------------------------------------------------------------


def _bubble_frame(seed: int, fill_top: int, n_bubbles: int) -> np.ndarray:
    rng = _rng(seed)
    img = _draw_cup(_base_scene(_rng(1)), fill_top, INDICATOR_PURPLE, _rng(6))
    x, y, w, h = CUP_ROI
    cup_left, cup_right = x + 14, x + w - 14
    cup_bottom = y + h - 10
    for _ in range(n_bubbles):
        bx = rng.randint(cup_left, cup_right)
        by = rng.randint(fill_top + 4, cup_bottom)
        r = rng.randint(2, 6)
        cv2.circle(img, (bx, by), r, (232, 226, 240), -1)
    return _finish(img)


def make_effervescence_fixtures(n: int = 8) -> None:
    print("effervescence fixtures:")
    fill_top = CUP_ROI[1] + 45
    for i in range(n):
        # Each frame gets a different bubble seed -> real inter-frame motion.
        _write(f"fizz_{i:02d}.png", _bubble_frame(100 + i, fill_top, n_bubbles=55))
    for i in range(n):
        # Same bubble seed every frame -> a still surface. Only sensor noise
        # differs, which the detector must reject as sub-threshold.
        _write(f"still_{i:02d}.png", _bubble_frame(200, fill_top, n_bubbles=0))


# --------------------------------------------------------------------------
# Volume fixtures (Experiment C) and negative control (Experiment D)
# --------------------------------------------------------------------------

SNOW_WHITE = (238, 240, 244)
SAND_TAN = (196, 168, 120)


def make_volume_fixtures() -> None:
    print("volume / negative-control fixtures:")
    x, y, w, h = CUP_ROI
    low_fill = y + h - 38          # a thin layer of dry powder
    high_fill = y + 32             # expanded, near the top of the cup

    _write("snow_before.png",
           _finish(_draw_cup(_base_scene(_rng(1)), low_fill, SNOW_WHITE, _rng(7))))
    _write("snow_after.png",
           _finish(_draw_cup(_base_scene(_rng(1)), high_fill, SNOW_WHITE, _rng(8))))

    # Hydrophobic sand: water added, nothing happens. Same level, different noise.
    _write("sand_before.png",
           _finish(_draw_cup(_base_scene(_rng(1)), low_fill, SAND_TAN, _rng(9))))
    _write("sand_after.png",
           _finish(_draw_cup(_base_scene(_rng(1)), low_fill + 1, SAND_TAN, _rng(10))))


# --------------------------------------------------------------------------
# Real booklet crops
# --------------------------------------------------------------------------


def make_booklet_fixtures() -> None:
    print("real booklet crops:")
    for name, (source, roi) in BOOKLET_CROPS.items():
        src = ROOT / source
        if not src.is_file():
            print(f"  SKIP {name}: {source} not found at project root")
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if roi is not None:
            cx, cy, cw, ch = roi
            img = img[cy:cy + ch, cx:cx + cw]
        else:
            # Full pages are downscaled -- they feed the Goal Extraction Agent,
            # which does not need 24 megapixels.
            scale = 1600.0 / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _write(name, img)


def main() -> None:
    make_color_fixtures()
    make_effervescence_fixtures()
    make_volume_fixtures()
    make_booklet_fixtures()
    print("done")


if __name__ == "__main__":
    main()
