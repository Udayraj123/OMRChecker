"""Realistic scan simulation for prefilled answer sheets.

The goal is to make synthetic prefilled sheets look like they passed through
a real scanner: subtle geometry drift, uneven lighting, paper texture, and
JPEG artefacts.  The feature is deterministic for a given candidate number
and preset so test fixtures and reproduced issue reports stay stable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

REALISM_PRESETS = ("none", "subtle", "moderate", "adversarial")


@dataclass(frozen=True)
class BubbleGeometry:
    """Pixel-space geometry for one candidate-number bubble."""

    column: int
    digit: int
    cx: int
    cy: int
    radius: int
    filled: bool = False


@dataclass(frozen=True)
class MarkerBox:
    """Pixel-space rectangle for one ArUco marker."""

    corner: int
    x0: int
    y0: int
    x1: int
    y1: int


def normalize_realism_preset(preset: str | None) -> str:
    """Return a supported preset name or raise ``ValueError``."""
    value = (preset or "none").strip().lower()
    if value not in REALISM_PRESETS:
        allowed = ", ".join(REALISM_PRESETS)
        raise ValueError(f"realism_preset must be one of: {allowed}.")
    return value


def _stable_seed(*parts: object) -> int:
    """Create a deterministic uint32 seed from arbitrary values."""
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little") & 0xFFFFFFFF


def _rng_for(candidate_number: str | None, preset: str, seed: int | None):
    if seed is not None:
        return np.random.default_rng(seed & 0xFFFFFFFF)
    return np.random.default_rng(_stable_seed("scan-simulation", candidate_number or "", preset))


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_imperfect_bubbles(
    arr: np.ndarray,
    bubbles: Iterable[BubbleGeometry],
    *,
    preset: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace machine-perfect solid bubbles with pencil-like fills."""
    if preset == "none":
        return arr

    out = arr.copy()
    filled = [b for b in bubbles if b.filled]
    if not filled:
        return out

    if preset == "subtle":
        gray_range = (18, 42)
        shift_px = 1
        scale_range = (0.88, 1.00)
        noise_sigma = 5
    elif preset == "moderate":
        gray_range = (25, 70)
        shift_px = 3
        scale_range = (0.74, 1.08)
        noise_sigma = 9
    else:
        gray_range = (30, 95)
        shift_px = 5
        scale_range = (0.45, 1.15)
        noise_sigma = 14

    for bubble in filled:
        # Lightly erase the original perfect fill first, then redraw an
        # off-centre noisy ellipse so it looks hand-filled.
        erase_pad = max(2, bubble.radius // 4)
        x0 = max(0, bubble.cx - bubble.radius - erase_pad)
        y0 = max(0, bubble.cy - bubble.radius - erase_pad)
        x1 = min(out.shape[1], bubble.cx + bubble.radius + erase_pad + 1)
        y1 = min(out.shape[0], bubble.cy + bubble.radius + erase_pad + 1)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], 225)

        dx = int(rng.integers(-shift_px, shift_px + 1))
        dy = int(rng.integers(-shift_px, shift_px + 1))
        scale_x = float(rng.uniform(*scale_range))
        scale_y = float(rng.uniform(*scale_range))
        axes = (
            max(2, int(bubble.radius * scale_x)),
            max(2, int(bubble.radius * scale_y)),
        )
        angle = float(rng.uniform(-12, 12))
        fill_gray = int(rng.integers(gray_range[0], gray_range[1] + 1))
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            mask,
            (bubble.cx + dx, bubble.cy + dy),
            axes,
            angle,
            0,
            360,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
        local_noise = rng.normal(0, noise_sigma, out.shape[:2]).astype(np.int16)
        fill_plane = _clip_uint8(np.full(out.shape[:2], fill_gray, dtype=np.int16) + local_noise)
        fill_rgb = np.dstack([fill_plane, fill_plane, fill_plane])
        alpha = (cv2.GaussianBlur(mask, (3, 3), 0.5).astype(np.float32) / 255.0)[..., None]
        out = _clip_uint8(out.astype(np.float32) * (1.0 - alpha) + fill_rgb.astype(np.float32) * alpha)

    if preset in {"moderate", "adversarial"} and len(filled) >= 2:
        victim = filled[int(rng.integers(0, len(filled)))]
        # Erasure smear: a pale rubbed patch crossing the selected filled bubble.
        patch_w = max(8, int(victim.radius * rng.uniform(1.2, 2.0)))
        patch_h = max(5, int(victim.radius * rng.uniform(0.45, 0.8)))
        x0 = max(0, victim.cx - patch_w // 2)
        y0 = max(0, victim.cy - patch_h // 2)
        x1 = min(out.shape[1], x0 + patch_w)
        y1 = min(out.shape[0], y0 + patch_h)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], int(rng.integers(175, 215)))

    if preset == "adversarial":
        victim = filled[int(rng.integers(0, len(filled)))]
        line_y = victim.cy + int(rng.integers(-victim.radius, victim.radius + 1))
        cv2.line(
            out,
            (max(0, victim.cx - victim.radius * 6), line_y),
            (min(out.shape[1] - 1, victim.cx + victim.radius * 6), line_y + int(rng.integers(-2, 3))),
            (55, 55, 55),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        same_col = [b for b in bubbles if b.column == victim.column and not b.filled]
        if same_col:
            extra = same_col[int(rng.integers(0, len(same_col)))]
            cv2.ellipse(
                out,
                (extra.cx + int(rng.integers(-3, 4)), extra.cy),
                (max(2, extra.radius - 2), max(2, extra.radius - 3)),
                0,
                0,
                360,
                (45, 45, 45),
                -1,
                lineType=cv2.LINE_AA,
            )

    return out


def _apply_geometry(
    arr: np.ndarray,
    *,
    preset: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rotation + translation + perspective jitter.

    Returns ``(warped_image, perspective_matrix)``. The perspective matrix
    is the **composed** 3x3 homography mapping original coordinates into the
    warped image, so callers can re-project ArUco marker bounding boxes
    without rebuilding the math.
    """
    h, w = arr.shape[:2]
    if preset == "none":
        return arr, np.eye(3, dtype=np.float32)
    if preset == "subtle":
        rot = float(rng.uniform(-0.45, 0.45))
        shift = 0.004
        jitter = 0.003
    elif preset == "moderate":
        # Moderate kept visibly skewed but bounded so ArUco corner search
        # still succeeds without protection (and the marker-restore pass
        # below covers the residual cases).
        rot = float(rng.uniform(-1.4, 1.4))
        shift = 0.008
        jitter = 0.0045
    else:
        rot = float(rng.uniform(-4.5, 4.5))
        shift = 0.018
        jitter = 0.016

    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, rot, 1.0)
    m[0, 2] += float(rng.uniform(-shift, shift) * w)
    m[1, 2] += float(rng.uniform(-shift, shift) * h)
    # Promote 2x3 affine to 3x3 so it can be composed with the perspective
    # matrix below.
    affine = np.vstack([m, [0.0, 0.0, 1.0]]).astype(np.float32)

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    max_dx = jitter * w
    max_dy = jitter * h
    dst = src + rng.uniform([-max_dx, -max_dy], [max_dx, max_dy], size=(4, 2)).astype(np.float32)
    pm = cv2.getPerspectiveTransform(src, dst).astype(np.float32)

    combined = pm @ affine

    warped = cv2.warpPerspective(
        arr,
        combined,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped, combined


_ARUCO_DICT_ID_FOR_PROTECT = cv2.aruco.DICT_4X4_50


def _aruco_marker_for_protect(marker_id: int, size_px: int) -> np.ndarray:
    """Generate a fresh axis-aligned ArUco marker as an RGB ``uint8`` array.

    The image is the binary ArUco pattern plus a 1-cell white quiet zone
    border (so the detector can find the marker against any background).
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_ID_FOR_PROTECT)
    gray = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), int(size_px))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


def _protect_markers(
    pristine: np.ndarray,
    damaged: np.ndarray,
    markers: Iterable[MarkerBox],
    transform: np.ndarray,
    *,
    quiet_zone_ratio: float = 0.45,
    stamp_fresh_markers: bool = True,
) -> np.ndarray:
    """Restore ArUco marker regions after damaging effects.

    Two-stage approach:

    1.  Compute the post-warp bounding box of each marker (via the
        composed homography) plus a generous quiet-zone pad. Paste the
        ``pristine`` (post-geometry, pre-damage) pixels back into that
        rectangle on the ``damaged`` image. This restores the quiet zone
        background to its undegraded state.

    2.  At small marker sizes (~40-60 px) the ``warpPerspective`` bilinear
        resample fuzzes the binary pattern enough that detection becomes
        unreliable. When ``stamp_fresh_markers`` is on we therefore also
        re-stamp a freshly generated ArUco bitmap at the post-warp marker
        centroid. The fresh marker is axis-aligned in the warped image
        which the ArUco detector handles natively (rotation invariant).
    """
    marker_list = list(markers)
    if not marker_list:
        return damaged

    h, w = damaged.shape[:2]
    out = damaged.copy()

    # ------------------------------------------------------------------
    # Stage 1: restore quiet-zone backgrounds from the pristine snapshot.
    # ------------------------------------------------------------------
    for box in marker_list:
        bw = box.x1 - box.x0
        bh = box.y1 - box.y0
        pad = int(round(max(bw, bh) * quiet_zone_ratio))
        original_corners = np.array(
            [
                [box.x0 - pad, box.y0 - pad],
                [box.x1 + pad, box.y0 - pad],
                [box.x1 + pad, box.y1 + pad],
                [box.x0 - pad, box.y1 + pad],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(original_corners, transform).reshape(-1, 2)
        xs = np.clip(warped[:, 0], 0, w - 1).astype(np.int32)
        ys = np.clip(warped[:, 1], 0, h - 1).astype(np.int32)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] = pristine[y0:y1, x0:x1]

    if not stamp_fresh_markers:
        return out

    # ------------------------------------------------------------------
    # Stage 2: stamp a fresh axis-aligned ArUco bitmap for each marker.
    #
    # We also flood the marker's bounding rectangle with white BEFORE the
    # stamp so any black/grey damage from the lighting/noise/jpeg passes
    # doesn't bleed into the quiet zone. The stamp position is clamped to
    # keep the marker fully on-page; the warp can push the original corner
    # close enough to the image edge that a naïve placement clips the
    # stamp and breaks detection (this was the main failure mode at
    # `moderate` on small templates).
    # ------------------------------------------------------------------
    quiet_zone_px_default = 6
    for box in marker_list:
        center_src = np.array([[(box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0]], dtype=np.float32)
        center_warped = cv2.perspectiveTransform(center_src.reshape(-1, 1, 2), transform).reshape(-1)
        cx_w, cy_w = float(center_warped[0]), float(center_warped[1])

        edge_src = np.array(
            [[box.x0, box.y0], [box.x1, box.y0], [box.x0, box.y1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        edge_warped = cv2.perspectiveTransform(edge_src, transform).reshape(-1, 2)
        side_x = float(np.linalg.norm(edge_warped[1] - edge_warped[0]))
        side_y = float(np.linalg.norm(edge_warped[2] - edge_warped[0]))
        stamp_size = int(round(max(side_x, side_y)))
        if stamp_size < 12:
            continue

        quiet_zone = max(quiet_zone_px_default, stamp_size // 8)

        # Clamp the stamp so it (plus quiet zone) stays inside the image.
        # If the warp pushed the marker past the image edge, slide it back.
        min_cx = quiet_zone + stamp_size / 2.0
        min_cy = quiet_zone + stamp_size / 2.0
        max_cx = w - quiet_zone - stamp_size / 2.0
        max_cy = h - quiet_zone - stamp_size / 2.0
        if max_cx < min_cx or max_cy < min_cy:
            continue  # Image too small for stamp+quiet zone; skip.
        cx_w = float(np.clip(cx_w, min_cx, max_cx))
        cy_w = float(np.clip(cy_w, min_cy, max_cy))

        sx0 = int(round(cx_w - stamp_size / 2.0))
        sy0 = int(round(cy_w - stamp_size / 2.0))
        sx1 = sx0 + stamp_size
        sy1 = sy0 + stamp_size

        # White quiet zone first (covers any residual damage), then marker.
        qx0 = max(0, sx0 - quiet_zone)
        qy0 = max(0, sy0 - quiet_zone)
        qx1 = min(w, sx1 + quiet_zone)
        qy1 = min(h, sy1 + quiet_zone)
        out[qy0:qy1, qx0:qx1] = 255

        marker_rgb = _aruco_marker_for_protect(box.corner, stamp_size)
        out[sy0:sy1, sx0:sx1] = marker_rgb

    return out


def _apply_lighting(arr: np.ndarray, *, preset: str, rng: np.random.Generator) -> np.ndarray:
    if preset == "none":
        return arr
    out = arr.astype(np.float32)
    h, w = out.shape[:2]

    if preset == "subtle":
        strength = float(rng.uniform(0.025, 0.055))
        band_strength = float(rng.uniform(0.025, 0.055))
    elif preset == "moderate":
        strength = float(rng.uniform(0.07, 0.13))
        band_strength = float(rng.uniform(0.06, 0.12))
    else:
        strength = float(rng.uniform(0.12, 0.22))
        band_strength = float(rng.uniform(0.12, 0.25))

    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial = np.clip(xx * xx + yy * yy, 0, 1)
    vignette = 1.0 - strength * radial
    out *= vignette[..., None]

    band_y = int(rng.integers(max(1, h // 8), max(2, h - h // 8)))
    band_h = max(12, int(h * (0.035 if preset == "subtle" else 0.07)))
    band = np.exp(-((np.arange(h, dtype=np.float32) - band_y) ** 2) / (2 * (band_h ** 2)))
    out *= (1.0 - band_strength * band[:, None, None])
    return _clip_uint8(out)


def _apply_noise_texture(arr: np.ndarray, *, preset: str, rng: np.random.Generator) -> np.ndarray:
    if preset == "none":
        return arr
    if preset == "subtle":
        sigma = 2.5
        texture_alpha = 0.018
    elif preset == "moderate":
        sigma = 6.0
        texture_alpha = 0.035
    else:
        sigma = 11.0
        texture_alpha = 0.055

    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    out = arr.astype(np.float32) + noise

    h, w = arr.shape[:2]
    small_h = max(8, h // 48)
    small_w = max(8, w // 48)
    texture = rng.normal(0, 1, (small_h, small_w)).astype(np.float32)
    texture = cv2.resize(texture, (w, h), interpolation=cv2.INTER_CUBIC)
    texture = (texture - texture.min()) / max(1e-6, texture.max() - texture.min())
    texture = (texture - 0.5) * 255
    out += texture[..., None] * texture_alpha

    if preset == "adversarial":
        amount = max(8, int(0.0007 * h * w))
        ys = rng.integers(0, h, amount)
        xs = rng.integers(0, w, amount)
        vals = rng.choice([0, 255], amount)
        out[ys, xs] = vals[:, None]

    return _clip_uint8(out)


def _apply_page_artifacts(
    arr: np.ndarray,
    markers: Iterable[MarkerBox],
    *,
    preset: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if preset == "none":
        return arr
    out = arr.copy()
    h, w = out.shape[:2]

    if preset in {"moderate", "adversarial"}:
        # Three-hole-punch shadow just inside the left margin.
        x = int(w * 0.035)
        for y_frac in (0.26, 0.50, 0.74):
            y = int(h * y_frac + rng.integers(-8, 9))
            radius = max(7, int(h * 0.010))
            cv2.circle(out, (x, y), radius, (32, 32, 32), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + 2, y + 2), max(3, radius - 4), (235, 235, 235), -1, lineType=cv2.LINE_AA)

        # Soft water/coffee stain away from markers.
        overlay = out.copy()
        cx = int(rng.uniform(0.25, 0.75) * w)
        cy = int(rng.uniform(0.45, 0.82) * h)
        axes = (int(rng.uniform(0.04, 0.08) * w), int(rng.uniform(0.025, 0.06) * h))
        color = (150, 132, 95) if out.ndim == 3 else 150
        cv2.ellipse(overlay, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, color, -1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(9, axes[0] / 4))
        alpha = 0.08 if preset == "moderate" else 0.18
        out = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)

    if preset == "adversarial":
        marker_list = list(markers)
        if marker_list:
            victim = marker_list[int(rng.integers(0, len(marker_list)))]
            # Dog-ear / paper-slip occlusion over ~30% of one marker.
            pad = max(10, (victim.x1 - victim.x0) // 2)
            pts = np.array(
                [
                    [max(0, victim.x0 - pad), max(0, victim.y0 - pad)],
                    [min(w - 1, victim.x1 + pad), max(0, victim.y0 - pad)],
                    [max(0, victim.x0 - pad), min(h - 1, victim.y1 + pad)],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(out, [pts], (250, 250, 250), lineType=cv2.LINE_AA)

        # Pencil scribble / X line across the candidate number area.
        y = int(h * rng.uniform(0.22, 0.38))
        cv2.line(
            out,
            (int(w * 0.60), y),
            (int(w * 0.97), y + int(rng.integers(-20, 21))),
            (65, 65, 65),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return out


def _apply_jpeg_roundtrip(arr: np.ndarray, *, preset: str, rng: np.random.Generator) -> np.ndarray:
    if preset == "none":
        return arr
    if preset == "subtle":
        qualities = [int(rng.integers(86, 93))]
    elif preset == "moderate":
        qualities = [int(rng.integers(74, 86))]
    else:
        qualities = [int(rng.integers(82, 91)), int(rng.integers(55, 70))]

    out = arr
    for quality in qualities:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            continue
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        out = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return out


def apply_scan_simulation(
    image: Image.Image,
    *,
    preset: str = "none",
    candidate_number: str | None = None,
    bubbles: Iterable[BubbleGeometry] | None = None,
    markers: Iterable[MarkerBox] | None = None,
    seed: int | None = None,
) -> Image.Image:
    """Apply a deterministic scan-realism preset to a PIL image."""
    preset = normalize_realism_preset(preset)
    if preset == "none":
        return image

    rng = _rng_for(candidate_number, preset, seed)
    arr = _to_rgb_array(image)
    arr = _draw_imperfect_bubbles(arr, bubbles or (), preset=preset, rng=rng)
    arr, geo_transform = _apply_geometry(arr, preset=preset, rng=rng)
    # Snapshot the post-geometry image while markers are still pristine.
    # The `subtle` and `moderate` presets restore these patches at the end
    # of the pipeline so vignette / noise / JPEG can't blow up ArUco
    # detection. `adversarial` intentionally skips the restore: the whole
    # point of that preset is to stress-test the engine.
    pristine_post_geo = arr.copy() if preset in {"subtle", "moderate"} else None
    arr = _apply_lighting(arr, preset=preset, rng=rng)
    arr = _apply_noise_texture(arr, preset=preset, rng=rng)
    arr = _apply_page_artifacts(arr, markers or (), preset=preset, rng=rng)
    arr = _apply_jpeg_roundtrip(arr, preset=preset, rng=rng)
    if pristine_post_geo is not None:
        arr = _protect_markers(
            pristine_post_geo,
            arr,
            markers or (),
            geo_transform,
        )
    return Image.fromarray(arr, mode="RGB")


def image_difference_score(a: Image.Image, b: Image.Image) -> float:
    """Return mean absolute per-channel difference for test assertions."""
    aa = _to_rgb_array(a).astype(np.int16)
    bb = _to_rgb_array(b).astype(np.int16)
    if aa.shape != bb.shape:
        raise ValueError(f"Image sizes differ: {aa.shape} != {bb.shape}")
    return float(np.abs(aa - bb).mean())
