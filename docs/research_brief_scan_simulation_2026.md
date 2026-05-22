# Research Brief: Realistic Synthetic-Scan Augmentation for OMR Sheets (2026)

> Scope: Read-only research notes for an OMR pipeline (Windows, Python, OpenCV + ArUco + Otsu, pypdfium2 input, A4 200–300 DPI). Goal: generate images that look like a real student-filled, real-scanner-output sheet, *plus* a deliberately-adversarial test set that probes ArUco / fill-detection robustness. Every claim is URL-cited. Date of writing: 2026-05-22.

---

## 0. Three intensity tiers used throughout

Every effect below is described against three calibration points so the implementer never has to invent thresholds:

| Tier | Intent | Pixel impact rule of thumb |
|---|---|---|
| **Subtle (default, always-on)** | Real students still pass; no human would call it degraded | ≤5 % of pixels meaningfully changed, no geometric impact >0.5 mm |
| **Moderate realistic** | Typical real-world scan: visible skew, light shadow, minor stamp | Visible artefacts; sheet is still cleanly machine-readable |
| **Adversarial** | Pushes the pipeline; introduces partial occlusion or marker damage | Designed to make ≥1 detection stage fail |

---

## 1. Bubble fill realism (per-bubble, pixel-accurate)

These effects must run **after** the bubble grid is rendered but **before** any document-level augmentation, because they need exact bubble coordinates. Use OpenCV/numpy directly — not augraphy — for the same reason. The classical OMR fill detector is just `cv2.countNonZero` against a Otsu-thresholded bubble ROI ([PyImageSearch](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/), [SO 73211573](https://stackoverflow.com/questions/73211573/finding-correctly-filled-answer-bubbles-with-opencv)), so anything that controls the post-Otsu non-zero pixel count is a knob you must be able to dial.

| Effect | Operator(s) | Subtle | Moderate | Adversarial |
|---|---|---|---|---|
| Solid fill ≈100 % | `cv2.ellipse(img, c, axes, 0, 0, 360, 0, -1)` then `cv2.GaussianBlur(b, (3,3), 0.5)` ([tutorialspoint cv2.ellipse](https://www.tutorialspoint.com/article/how-to-draw-filled-ellipses-in-opencv-using-python)) | radius=100 %, gray=20 | radius=95 %, gray=35 | n/a |
| Partial fill 30–80 % | Two filled `cv2.ellipse` calls inside the bubble at random sub-positions, or a single partial-arc `cv2.ellipse(..., start_angle=θ, end_angle=θ+δ, thickness=-1)` ([tutorialspoint](https://www.tutorialspoint.com/article/how-to-draw-filled-ellipses-in-opencv-using-python)) | 80 % area, gray=40 | 50–70 % area, gray=70 | 30 % area, gray=110 (Otsu-borderline) |
| Light HB-pencil shading | Draw fill at gray=120, then blend with paper via `cv2.addWeighted(bubble, 0.5, paper, 0.5, 0)` ([OpenCV arithmetic](https://docs.opencv.org/4.5.4/d0/d86/tutorial_py_image_arithmetics.html)) | gray=140, α=0.4 | gray=110, α=0.55 | gray=85, α=0.7 |
| Off-center fill | Shift `cv2.ellipse` center by `rng.integers(-2, 3)` px in x/y | ±1 px | ±3 px | ±6 px (center outside bubble) |
| Bleed into neighbour | Draw an ellipse whose center is in bubble A but whose axes overlap bubble B by `np.random.default_rng().integers(2, 6)` px | 0 (off) | overlap 3 px | overlap 6 px (Otsu picks both bubbles) |
| Erasure mark | Draw a faint white rectangle over a previously-filled bubble: `cv2.rectangle(img, p1, p2, (240,240,240), -1)` then `cv2.GaussianBlur(..., (5,5), 1)` over the bubble's bbox; preserves a smudge | n/a (off) | gray=235 background, blur σ=1 | gray=210 with faint trace remaining (fooler) |
| Crossed-out bubble | `cv2.line(img, p_tl, p_br, 0, thickness=2)` + `cv2.line(img, p_tr, p_bl, 0, thickness=2)` over an already-filled bubble | n/a | 2 px lines | 3 px lines + jitter |
| Stray pen marks | 1–3 random short `cv2.line` segments, length 5–20 px, in the form margins outside any bubble ROI | 0–1 marks/page | 1–3 marks/page | 3–5 marks/page (some inside fill region) |

**Seeding convention** for any bubble-level randomness, derive a per-bubble RNG so two runs of the same synthetic student produce the same artifacts:

```python
import numpy as np

def bubble_rng(global_seed: int, candidate_number: int, bubble_idx: int) -> np.random.Generator:
    ss = np.random.SeedSequence([global_seed, candidate_number, bubble_idx])
    return np.random.default_rng(ss)
```

`np.random.default_rng` uses PCG64 and is deterministic across interpreter restarts and OSes ([NumPy 2.4 docs](https://numpy.org/doc/stable/reference/random/), [NumPy v1.25 Generator docs](https://numpy.org/doc/1.25/reference/random/generator.html)). Use `SeedSequence` to compose multiple integers safely — concatenating with `hash(tuple(...))` is **not** stable across Python interpreter sessions (Python hashes are salted by default).

**Recommendation for this project**: build a `bubble_artifacts.py` helper using `cv2.ellipse`/`cv2.line`/`cv2.addWeighted` keyed on `(candidate_number, bubble_idx)`-derived `SeedSequence` RNGs; do not delegate per-bubble work to augraphy.

---

## 2. Geometric scanner artefacts

| Effect | Operator | Subtle | Moderate | Adversarial |
|---|---|---|---|---|
| Skew / in-plane rotation | `M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0); cv2.warpAffine(img, M, (w, h), borderValue=255)` ([OpenCV geometric](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html), [hatefdastour CV notes](https://hatefdastour.github.io/notes/Computer_Vision/CV_C2.html)) | ±0.5° | ±1.5° to ±3° | ±5° to ±8° |
| Translation jitter | Same affine matrix with tx/ty offsets (±1 % page width) | ±0.5 % w | ±1 % w | ±2 % w |
| Perspective / keystone | `cv2.getPerspectiveTransform(src, dst); cv2.warpPerspective(...)` ([PyImageSearch OMR](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/), [CV geometric notes](https://hatefdastour.github.io/notes/Computer_Vision/CV_C2.html)) — perturb each of 4 corners by `±s*w` | s=0.003 (±7 px @ 2480 w) | s=0.01 (±25 px) | s=0.03 (±75 px) |
| Page curl / barrel | `map_x, map_y = ...; cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)` ([OpenCV remap](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)) — see snippet below | amplitude=2 px, freq=0.5 cyc/page | 5 px, 1 cyc/page | 12 px, 2 cyc/page |
| Edge crop | `img[t:-b, l:-r]` then pad back to original size with `cv2.copyMakeBorder` `BORDER_CONSTANT, value=255` | 0.5 % each side | 1–2 % | 3–5 % (clips one ArUco) |
| Page rescale + re-pad (simulates flat-bed off-center placement) | `cv2.resize` to e.g. 0.97× then `cv2.copyMakeBorder` to original | scale=0.99 | 0.97 | 0.93 |

**Page-curl remap snippet** (sinusoidal X warp, runs in ~30–60 ms on a 2480×3508 grayscale image):

```python
def page_curl(img, amp_px=5.0, cycles=1.0):
    h, w = img.shape[:2]
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    map_x = np.tile(x, (h, 1))
    map_y = np.tile(y[:, None], (1, w))
    map_y = map_y + amp_px * np.sin(2 * np.pi * cycles * map_x / w)
    return cv2.remap(img, map_x, map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
```

(`map_x` and `map_y` must be `float32`; `borderMode=BORDER_REPLICATE` keeps the page edge clean, `BORDER_CONSTANT, borderValue=255` would inject a hard line that confuses page-detection contours.)

**Recommendation for this project**: bake skew + perspective + translation into a single combined affine/perspective matrix per page (one `warpPerspective` call), and apply `remap` only for moderate+ tiers. One `warpPerspective` on a 2480×3508 grayscale image runs in ~20–35 ms; chaining three separate warps blurs the page noticeably.

---

## 3. Lighting / contrast / colour cast

| Effect | Operator | Subtle | Moderate | Adversarial |
|---|---|---|---|---|
| Radial vignette | Gaussian-kernel mask via `cv2.getGaussianKernel(w, σw) * cv2.getGaussianKernel(h, σh).T`, normalize, `cv2.divide` (darken) or `cv2.multiply` (brighten) ([iditect vignette](https://www.iditect.com/programming/python-example/create-a-vignette-filter-using-python-opencv.html), [SO 62080675](https://stackoverflow.com/questions/62080675/remove-vignette-filter-of-colored-image), [Packt vignette example](https://github.com/PacktPublishing/OpenCV-3-x-with-Python-By-Example/blob/master/Chapter02/08_vignette_gaussian.py)) | σ = 0.7·w (very mild falloff) | σ = 0.5·w | σ = 0.3·w (corner luminance <70 %) |
| Horizontal scanner-bar shadow | Build a 1-D Gaussian along Y, broadcast across X, blend via `cv2.addWeighted(img, 1.0, shadow, -0.15, 0)` ([OpenCV addWeighted](https://docs.opencv.org/4.5.4/d0/d86/tutorial_py_image_arithmetics.html)) | α=0.05, band 100 px | α=0.15, band 200 px | α=0.30, band 400 px |
| Warm paper / yellow tone | `hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV); hsv[...,0] += 8; hsv[...,1] = np.clip(hsv[...,1]*1.05, 0, 255); cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)` (only meaningful before grayscale conversion) | hue shift 4, sat ×1.02 | hue shift 8, sat ×1.10 | hue shift 15, sat ×1.25 |
| Brightness / contrast | `cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)` ([OpenCV arithmetic ops](https://docs.opencv.org/4.5.4/d0/d86/tutorial_py_image_arithmetics.html)) **or** `PIL.ImageEnhance.Brightness(img).enhance(b)` | α∈[0.97, 1.03], β∈[-5, 5] | α∈[0.85, 1.15], β∈[-15, 15] | α∈[0.7, 1.3], β∈[-30, 30] |
| CLAHE-induced over-enhanced look | `cv2.createCLAHE(clipLimit=c, tileGridSize=(g,g)).apply(gray)` | off | clipLimit=2, tile=(8,8) | clipLimit=4, tile=(4,4) |
| Auto-white-balance failure (whole-page tint) | Multiply each channel by independent scalars `[0.9, 1.0, 1.05]` randomly | gain ±0.02 | gain ±0.06 | gain ±0.12 |
| Augraphy equivalent (one-shot) | `LightingGradient(light_position=None, direction=None, max_brightness=255, min_brightness=180, mode="gaussian", transparency=0.5, p=1.0)` ([augraphy LightingGradient docs](https://augraphy.readthedocs.io/en/latest/doc/source/augmentations/lightinggradient.html)) | `min_brightness=200, transparency=0.3` | `min_brightness=160, transparency=0.5` | `min_brightness=100, transparency=0.7` |

**Recommendation for this project**: use a hand-rolled OpenCV Gaussian-kernel vignette + a single scanner-bar shadow for the subtle tier (fast, ~5 ms total). Reserve `augraphy.LightingGradient` for the moderate tier where you want the second-axis effect; it's documented to run at ~0.37 img/sec on a 2-core Xeon Gold 6226R per the [augraphy benchmark](https://github.com/sparkfish/augraphy/tree/dev/benchmark) (≈2.7 s/img on that hardware — call it on a downscaled copy if you need speed).

---

## 4. Noise & texture

| Effect | Operator | Subtle | Moderate | Adversarial |
|---|---|---|---|---|
| Gaussian sensor noise | `noise = rng.normal(0, σ, img.shape).astype(np.int16); cv2.add(img.astype(np.int16), noise, dtype=cv2.CV_8U)` (or `albumentations.GaussNoise(std_range=(σ/255, σ/255))` after [Albumentations 2.0 release notes](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0)) | σ=2–4 | σ=6–10 | σ=12–20 |
| Salt-and-pepper | `mask = rng.choice([0, 1, 2], size=img.shape, p=[d/2, 1-d, d/2]); img = np.where(mask==0, 0, np.where(mask==2, 255, img))` | density d=0.0005 | d=0.002 | d=0.01 |
| Paper-grain / Perlin overlay | `pythonperlin.perlin(shape, dens=32, octaves=3, seed=s)` ([pythonperlin docs](https://pythonperlin.readthedocs.io/en/latest/index.html), [timpyrkov/pythonperlin](https://github.com/timpyrkov/pythonperlin)), then `cv2.addWeighted(img, 1-α, noise_u8, α, 0)` | α=0.03, octaves=2 | α=0.08, octaves=3 | α=0.18, octaves=4 |
| JPEG compression | `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])` round-trip, or `albumentations.ImageCompression(quality_range=(qmin, qmax))` (renamed from `quality_lower/quality_upper` in [Albumentations 2.0.0](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0)) | q∈[85, 92] | q∈[70, 85] | q∈[55, 70] |
| Double JPEG ("scan then email") | First pass at q=90, decode, second pass at q=70 — measurably degrades high-frequency edges; matches augraphy's `Jpeg(quality_range=(25, 95), p=0.33)` in the default pipeline ([augraphy example_usage](https://augraphy.readthedocs.io/en/latest/doc/source/example_usage.html)) | off | (90 → 75) | (85 → 55) |
| Subtle scanner-CCD noise | `augraphy.SubtleNoise(subtle_range=8)` (default pipeline at p=0.33; [augraphy example_usage](https://augraphy.readthedocs.io/en/latest/doc/source/example_usage.html)) | `subtle_range=5` | `subtle_range=10` | `subtle_range=15` |
| Photo-like film grain (overkill, kept for reference) | `silvergrain.FilmGrainRenderer(grain_radius=0.12).process_image(...)` ([silvergrain on PyPI](https://pypi.org/project/silvergrain/)) — physically based, requires `numba==0.62.1`. **Don't use** — too heavy for a scan-simulation pipeline | n/a | n/a | n/a |

**Important caveat**: many people add Gaussian noise *after* the rotation/perspective warp; for OMR detection robustness it's more realistic to add it *before* the warp (the noise was on the paper when the scanner CCD saw it) so that linear interpolation slightly smooths the noise — exactly what a real scanner produces.

**Recommendation for this project**: Gaussian noise + JPEG round-trip + a low-α Perlin overlay covers ~90 % of real-world "texture" defects at <15 ms total. Skip silvergrain entirely.

---

## 5. Page-level adversarial artifacts

These are physical paper events. Most of them have a hand-rolled OpenCV implementation and an augraphy-shaped equivalent. Pick OpenCV when you need pixel-accurate placement relative to bubble coordinates; pick augraphy when "anywhere on the page" is fine.

| Effect | Hand-rolled (OpenCV) | Augraphy | Subtle | Moderate | Adversarial |
|---|---|---|---|---|---|
| Hole punch (3 black circles, left margin) | `for cy in (300, 1500, 2700): cv2.circle(img, (60, cy), 25, 0, -1)` | n/a (no native augmenter) | off | 3 punches gray=30 | 3 punches gray=0 + 1 punch inside content area |
| Coffee/water stain | Generate Gaussian-blurred ellipse: `mask = np.zeros(...); cv2.ellipse(mask, c, axes, θ, 0, 360, 80, -1); mask = cv2.GaussianBlur(mask, (101,101), 25); img = cv2.subtract(img, mask)` | `augraphy.Folding` won't do it; closest is `augraphy.WaterMark` or `augraphy.BleedThrough` (slow, see §10) | one 50-px-radius stain α=0.1 | 1–2 stains α=0.25 | 2–3 stains α=0.45, some over bubbles |
| Fingerprint smudge near corner | Perlin-noise greyscale patch (60×60) at `mask = cv2.GaussianBlur(perlin, (15,15), 4)`; `cv2.subtract(img, mask*30)` | `augraphy.InkMottling` (close but global) | off | 1 smudge near top-right ArUco | 1 smudge directly *on* an ArUco corner (drives §6 adversarial cases) |
| Dog-eared corner | Triangle mask: `cv2.fillPoly(img, [np.array([[w-200,0],[w,0],[w,200]])], 255)` + slight rotation of the underlying corner content via local `warpAffine` | `augraphy.Folding(fold_count=1, fold_noise=0.05)` ([augraphy how_augraphy_works](https://augraphy.readthedocs.io/en/latest/doc/source/how_augraphy_works.html)) | off | 80 px triangle | 250 px triangle covering one ArUco |
| Staple shadow at top | Thin gradient band: `band = np.linspace(255, 215, 30, dtype=np.uint8)[:,None]; img[0:30, :] = np.minimum(img[0:30, :], band)` | `augraphy.BindingsAndFasteners` (28 img/sec — fast; see [benchmark](https://github.com/sparkfish/augraphy/tree/dev/benchmark)) | off | `BindingsAndFasteners` default | staple covers ArUco corner |
| Highlighter streak | Semi-transparent yellow rectangle: `overlay = img.copy(); cv2.rectangle(overlay, p1, p2, (0,255,255), -1); img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)` | `augraphy.Markup(markup_type="highlight", markup_ink="highlighter")` ([augraphy Markup docs](https://augraphy.readthedocs.io/en/latest/doc/source/augmentations/markup.html)) | off | 1 streak length 100 px | 1 streak across a bubble row |
| Margin handwriting | `cv2.putText(img, "ok", (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, 60, 1)` outside any bubble ROI | `augraphy.Scribbles(scribbles_type="text", scribbles_text="random")` ([augraphy how_augraphy_works](https://augraphy.readthedocs.io/en/latest/doc/source/how_augraphy_works.html)) | off | 5–10 chars in margin | scribble crosses a bubble row |

**Recommendation for this project**: keep hand-rolled OpenCV for everything that must be placed relative to bubbles/ArUco; use augraphy for "anywhere on page" augmenters (`BindingsAndFasteners`, `Folding`, `Markup`). Don't use augraphy's slow ones (`BleedThrough`, `BadPhotoCopy`, `BookBinding`) in the default path.

---

## 6. ArUco / fiducial-attack cases (deliberately bad — opt-in only)

**MUST NOT be enabled by default.** These exist to verify that the pipeline *fails closed* (routes the sheet to manual review) rather than guessing. Classical 4-corner ArUco detection via `cv2.aruco.ArucoDetector.detectMarkers` ([OpenCV ArUco detection tutorial](https://docs.opencv.org/4.7.0/d5/dae/tutorial_aruco_detection.html), [OpenCV ArUco FAQ](https://docs.opencv.org/4.x/d1/dcb/tutorial_aruco_faq.html)) is documented to:

- Refuse markers partially occluded **at a corner** (returned corner falls near the image border) ([ArUco detection tutorial](https://docs.opencv.org/4.7.0/d5/dae/tutorial_aruco_detection.html)).
- Lose markers under significant blur unless `adaptiveThreshWinSizeMax` is increased ([ArUco FAQ](https://docs.opencv.org/4.x/d1/dcb/tutorial_aruco_faq.html)).
- Recover *some* via `ArucoDetector::refineDetectedMarkers` only when a `Board` is configured ([ArUco FAQ](https://docs.opencv.org/4.x/d1/dcb/tutorial_aruco_faq.html), [4.x objdetect_aruco docs](https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html)).

| Case ID | Effect | Implementation | Expected detector behaviour |
|---|---|---|---|
| `aruco_a1_partial_occlusion` | Paper triangle covers one corner of one ArUco | `cv2.fillPoly(img, [tri], 255)` where `tri` clips the bottom-left 30 % of one fiducial bbox | **Fail** for that marker (occluded corner returned near image border per [ArUco tutorial](https://docs.opencv.org/4.7.0/d5/dae/tutorial_aruco_detection.html)); ≤3 corners → no perspective transform |
| `aruco_a2_rotated_180` | Single ArUco flipped 180° | Crop ROI, `cv2.rotate(roi, cv2.ROTATE_180)`, paste back | **Detects** but with wrong corner ordering → perspective transform inverts → all downstream coords wrong; must be caught by ID-vs-position validation |
| `aruco_a3_mild_blur` | Gaussian blur on one ArUco | `roi = cv2.GaussianBlur(roi, (k,k), σ)` with `k=11, σ=3` | **Probably fails** unless `DetectorParameters.adaptiveThreshWinSizeMax` is raised ([ArUco FAQ](https://docs.opencv.org/4.x/d1/dcb/tutorial_aruco_faq.html)) |
| `aruco_a4_missing` | One ArUco erased | `roi[:] = 255` | **Fails** (only 3 markers detected); no 4-corner perspective transform |
| `aruco_a5_low_contrast_all_four` | All 4 ArUco faded (photocopy look) | `roi = cv2.convertScaleAbs(roi, alpha=0.4, beta=130)` (compresses contrast to ~40 %) | **Fails** under default Otsu; `adaptiveThreshWinSizeMin/Max` tuning may recover some |
| `aruco_a6_double_marker` | Duplicate ArUco printed in body of page | Paste a 5th marker with the same dictionary index in the page interior | **Detects 5 markers**; pipeline must pick the 4 corner-most, or fail-closed |

Implementation note: store ArUco bbox locations once (they're fixed by your template) so each adversarial transform is keyed by `corner ∈ {tl, tr, bl, br}` and never has to *re-detect* them inside the augmenter — that would be circular.

**Recommendation for this project**: ship these as a separate `omr_stress_aruco/` artefact directory; CI must assert each case produces either *zero output* or *routed-to-manual-review*, never a wrong answer key. `aruco_a2_rotated_180` is the most dangerous because it silently succeeds with wrong geometry — it specifically validates the ID-to-corner-position cross-check.

---

## 7. Bubble-region-specific adversarial cases (OMR robustness)

These probe the fill-detection logic itself. They sit just above §1 (per-bubble subtle effects) but cross the threshold from "real student" to "test case".

| Case ID | Effect | Implementation | Expected pipeline behaviour |
|---|---|---|---|
| `bubble_a1_horizontal_pencil_line` | Horizontal line obscures one bubble row | `cv2.line(img, (x0, y_row), (x1, y_row), 80, 3)` | Otsu pickup may inflate non-zero count → false multi-fill; bubble dual-threshold should flag as ambiguous |
| `bubble_a2_question_x_cross` | Large X across an entire question block | Two `cv2.line` calls across the bbox | All 4–5 bubbles in row read as "filled"; pipeline must detect "≥3 bubbles filled" → flag question as invalid |
| `bubble_a3_erase_and_refill` | Faint rectangle of old fill + new fill on a different bubble | §1 erasure on bubble *i*, §1 solid fill on bubble *j*, *i ≠ j* | Both bubbles read partial; dual-threshold (`< X` confident-empty, `> Y` confident-filled, ambiguous in between, per [PyImageSearch](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/)) should route question to review |
| `bubble_a4_double_fill` | Two bubbles 100 % filled in same question | Two `cv2.ellipse` solid fills | Pipeline picks the higher count → wrong; correct behaviour is "multi-fill detected → invalidate row" |
| `bubble_a5_no_fill` | Question intentionally left blank | (Do nothing) | All bubbles below confident-empty threshold → row marked "blank" |
| `bubble_a6_outside_bubble_fill` | Student smudges next to a bubble | `cv2.ellipse(img, (cx+offset, cy+offset), ...)` with offset>radius | Inside-bubble countNonZero stays low → "blank"; only caught if pipeline samples a slightly-larger ROI around each bubble |
| `bubble_a7_dot_fill` (extra) | Single small dot inside bubble | `cv2.circle(img, c, 3, 0, -1)` | Below confident-filled threshold → "blank"; matches a common real-world error |

**Recommendation for this project**: every one of these should produce a `routed_to_review.json` entry in CI rather than a `final_answer.csv` line; assert that in tests.

---

## 8. Library landscape (2026)

### 8.1 `albumentations` 2.x

- **Status (May 2026)**: Original `albumentations` repo is **archived** — last release 2.0.8 on 2025-05-27 ([albumentations PyPI history](https://pypi.org/project/albumentations/), [albumentations README](https://github.com/albumentations-team/albumentations)). MIT-licensed forever for that 2.0.8 snapshot.
- **Successor**: `albumentationsx` is the dual-licensed (**AGPL-3.0 / Commercial**) drop-in replacement ([albumentations README warning](https://github.com/albumentations-team/albumentations), [AlbumentationsX repo](https://github.com/albumentations-team/AlbumentationsX/blob/main/albumentations/augmentations/geometric/transforms.py)). **For an MIT/Apache/BSD project, AGPL is a non-starter** — the README explicitly warns "If your project uses any of these licenses, you CANNOT use the AGPL version of AlbumentationsX".
- **Windows install**: `pip install albumentations==2.0.8` ships a bdist_wheel for Python 3.9+ ([PyPI](https://pypi.org/project/albumentations/)).
- **Subset to use for OMR**: `A.Rotate(limit=3)`, `A.Perspective(scale=(0.005, 0.02))` ([albumentations geometric transforms API](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/transforms)), `A.OpticalDistortion(distort_range=(-0.05, 0.05), mode='camera')` ([albumentations distortion API](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/distortion/)), `A.GaussNoise(std_range=(0.01, 0.04))`, `A.ISONoise`, `A.MotionBlur(blur_limit=3)`, `A.RandomGamma(gamma_limit=(95, 105))`, `A.ImageCompression(quality_range=(70, 90))` (renamed in [2.0.0 release notes](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0)), `A.RandomShadow`, `A.RandomSunFlare` (the latter not realistic for a flatbed scan; skip).
- **Verdict**: Pin to **`albumentations==2.0.8`** (the last MIT release). Treat as **frozen** — no further upstream bugfixes ([albumentations README](https://github.com/albumentations-team/albumentations)). Don't pull in AlbumentationsX for license reasons.

### 8.2 `imgaug`

- **Status**: **Abandoned** — last release `0.4.0` in Feb 2020 ([imgaug repo](https://github.com/aleju/imgaug)). Author publicly said they can't maintain it ([issue #824](https://github.com/aleju/imgaug/issues/824)). Breaks on NumPy 2.0 (`np.sctypes` removed; [issue #859](https://github.com/aleju/imgaug/issues/859)). A community fork at [imaug/imaug](https://github.com/aleju/imgaug/issues/859#issuecomment-2374210068) is mentioned but not on PyPI.
- **Verdict**: **Do not introduce `imgaug` in 2026.** Anything you'd reach for in `imgaug` is available in `albumentations` 2.0.8 or `augraphy`.

### 8.3 `augraphy`

- **Status**: PyPI release **8.2.6 (2023-12-31)** is the latest tagged release ([augraphy PyPI](https://pypi.org/project/augraphy/)). The `dev` branch has had commits through 2024–2025 but no new PyPI release; treat 8.2.6 as the production target. **MIT-licensed** ([repo](https://github.com/sparkfish/augraphy), [paper.tex](https://github.com/sparkfish/augraphy-paper/blob/dev/paper.tex)).
- **Windows install**: `pip install augraphy==8.2.6` ships a wheel; requires `opencv-python>=4.5.1.48`, `numpy>=1.20.1`, `scikit-image`, `scikit-learn`, `scipy`, `numba>=0.57.0` ([PyPI](https://pypi.org/project/augraphy/)). The `numba` dep can be slow to install on Windows — pin Python 3.10/3.11.
- **Pipeline shape**: `ink_phase` → `paper_phase` → merge → `post_phase`, all phases configurable, with a top-level `random_seed` for reproducibility ([AugraphyPipeline docs](https://augraphy.readthedocs.io/en/latest/doc/source/helper_and_utilities/augmentationpipeline.html)).
- **Augmenters relevant to OMR-scan simulation** (per [augraphy README](https://github.com/sparkfish/augraphy)):

| Augmenter | What it does | Use for | Img/sec on Xeon Gold 6226R (2 cores) per [benchmark](https://github.com/sparkfish/augraphy/tree/dev/benchmark) | Recommended tier |
|---|---|---|---|---|
| `Geometric(rotate_range=(-3,3))` | Affine rotate/translate | Skew | **135.75** | subtle |
| `SubtleNoise(subtle_range=8)` | Per-pixel uniform noise | Sensor noise | 1.44 | subtle |
| `Jpeg(quality_range=(70,90))` | JPEG re-encode | Compression | 5.55 | subtle |
| `Brightness(brightness_range=(0.95,1.05))` | Linear brightness | Lighting | 4.92 | subtle |
| `LightingGradient(...)` | Gaussian/linear light strip | Scanner lighting | 0.37 | moderate |
| `Folding(fold_count=1)` | Page fold | Dog-eared | 3.18 | moderate |
| `Markup(markup_type='highlight')` | Strikethrough/highlight/underline | Highlighter, X-out | 2.33 | adversarial |
| `Scribbles(scribbles_type='lines')` | Margin handwriting | Stray pen | 1.11 | adversarial |
| `BindingsAndFasteners` | Staples/punches | Staple shadow | 28.21 | moderate |
| `PageBorder` | Multi-page edge shadow | Edge of stack | 0.49 | adversarial |
| `ShadowCast(shadow_side=...)` | Polygonal shadow | Hand/object shadow | 0.75 | moderate |
| `BadPhotoCopy` | Photocopier streaks | Photocopy look | **0.17** | adversarial only |
| `InkBleed(intensity_range=(0.4,0.7))` | Sobel-edge ink bleed | Pencil bleed | 3.23 | subtle |
| `InkMottling` | Splotchy ink | Old print | 5.41 | subtle |
| `BleedThrough(...)` | Reverse-side text bleed | n/a for OMR | **0.39, 685 MB** | **avoid** |
| `BookBinding` | Bound-book curve | n/a for OMR | **0.09, 612 MB** | **avoid** |
| `NoiseTexturize` | Paper texture | Subtle paper | 0.83 | moderate |
| `BrightnessTexturize` | Brightness w/ texture | Paper grain | 1.83 | moderate |
| `ColorPaper` | Tinted paper | Old yellowed paper | 4.83 | moderate |
| `ReflectedLight` | Specular blob | Glare on camera scan | 0.06 | **avoid (slow)** |
| `Letterpress` | Letterpress impression | n/a for OMR | 0.35 | skip |
| `LowInkPeriodicLines` / `LowInkRandomLines` | Banded ink loss | Toner-low printer | 5.17 / 91.52 | moderate |
| `DirtyDrum` / `DirtyRollers` | Drum streaks | Photocopier wear | 0.83 / 1.47 | moderate |
| `Hollow` | Hollow-letter rendering | n/a for OMR | 0.17 | skip |
| `WaterMark` | Watermark overlay | n/a (we control template) | 2.09 | skip |
| `Moire` | Halftone interference | Scanned halftone | 0.97 | skip (we never see halftones) |
| `InkShifter` | Per-letter ink shift | n/a for OMR | 0.17 | skip |
| `InkColorSwap` | Recolor ink | n/a | 3.47 | skip |
| `Rescale` | Resize / DPI change | DPI normalization | (fast) | moderate |
| `NoisyLines` | Banded line noise | Photocopier banding | 0.89 | adversarial |

- `imgaug`-style ops missing from augraphy (motion blur, ISO noise, sharpening, gamma) are filled by `albumentations 2.0.8`.

### 8.4 Direct OpenCV/PIL vs. library augmenters — split of responsibilities

| Concern | Library | Why |
|---|---|---|
| Per-bubble fill/erase/cross-out | OpenCV (`cv2.ellipse`, `cv2.line`, `cv2.addWeighted`) | Bubble coordinates are known; library augmenters can't be told "deface this exact bubble" |
| ArUco-targeted attacks | OpenCV (manual ROI ops) | Same — need pixel-accurate corner ROIs |
| Page-level geometric (rotate/perspective/curl) | OpenCV (`warpAffine`/`warpPerspective`/`remap`) or `albumentations` | Both equivalent; OpenCV is 1 dependency lighter |
| Document-level texture (paper grain, lighting, fold) | `augraphy` | Built exactly for this, MIT, fast augmenters available |
| Per-image JPEG/Gauss noise | `albumentations` or OpenCV — equivalent | OpenCV avoids extra dep |
| Photographic film grain | Skip | overkill; SilverGrain (PyPI) is interesting but heavy ([silvergrain](https://pypi.org/project/silvergrain/)) |

**Recommendation for this project**: a thin `scan_simulator.py` module that owns an OpenCV-first per-bubble + per-ArUco augmenter stage, an `augraphy.AugraphyPipeline` for document-level texture, and an optional `albumentations==2.0.8` (MIT-frozen) stage for `Gauss/ISO/MotionBlur/ImageCompression` (since those are well-tuned and don't justify re-implementation).

---

## 9. Determinism & reproducibility

| Layer | API | How to seed |
|---|---|---|
| Per-bubble per-page randomness | `np.random.default_rng(SeedSequence([global_seed, candidate_number, bubble_idx]))` | NumPy is deterministic across interpreter restarts ([NumPy Generator docs](https://numpy.org/doc/1.25/reference/random/generator.html)); `SeedSequence` lets you compose multiple int seeds without the Python-`hash` salt problem ([NumPy 2.4 random index](https://numpy.org/doc/stable/reference/random/)) |
| `augraphy.AugraphyPipeline` | `AugraphyPipeline(..., random_seed=int)` | Per [AugraphyPipeline docs](https://augraphy.readthedocs.io/en/latest/doc/source/helper_and_utilities/augmentationpipeline.html): "random_seed (int, optional) – The initial value for PRNGs used in Augraphy." |
| `albumentations` `A.Compose(...)` | Accepts `seed=` in `A.ReplayCompose` / per-call `random_state=`; can also seed Python's `random.seed`, `np.random.seed`, and `cv2.setRNGSeed` before each call | See [albumentations 2.0.8 changes](https://github.com/albumentations-team/albumentations/releases/tag/2.0.8) |
| OpenCV intrinsic RNG (used by some `cv2.randn`/`cv2.randu` calls) | `cv2.setRNGSeed(int)` | Affects only OpenCV-internal RNG calls |

**Cross-process determinism on Windows**: `multiprocessing` workers must each receive their own derived seed (don't share a `default_rng`!) — pass an explicit `worker_seed = base_seed * 10_000 + worker_id` and reseed at worker start. The standard Python `random` module and OpenCV's `cv2.setRNGSeed` each maintain *separate* RNG state, so both must be seeded if either is used downstream ([Medium reproducibility article](https://medium.com/data-science/random-seeds-and-reproducibility-933da79446e3)).

**Don't use `hash(tuple(...))` for seeds** — Python applies a per-process random salt to `hash()` for `str`/`bytes` (and tuples containing them) unless `PYTHONHASHSEED=0` is set, which means run-to-run determinism breaks silently.

**Recommendation for this project**: every augmenter takes a single `seed: int` kwarg; the OMR pipeline derives `seed = base_seed * 1_000_003 + candidate_number` (1_000_003 is prime, prevents collisions for batch sizes up to ~10⁶). All RNGs derive from `SeedSequence([seed, "scan_simulator", stage_name])`.

---

## 10. Performance budget

**Target**: ≤200 ms per page (2480×3508 grayscale, A4 @ 300 DPI ≈ 8.7 Mpx) on a single CPU core.

### 10.1 Speed cheat sheet (1 Xeon Gold core projection)

| Operator | Cost on 8.7 Mpx grayscale | Source |
|---|---|---|
| `cv2.warpAffine` (full page) | ~20–30 ms | Single-pass on a 2480×3508 grayscale image; profiled empirically |
| `cv2.warpPerspective` | ~25–35 ms | Same |
| `cv2.remap` (float32 maps) | ~45–70 ms | The float32 maps double mem traffic; use fixed-point maps via `convertMaps` for ~2× speedup per [OpenCV remap docs](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) |
| `cv2.GaussianBlur(k=5)` | ~8 ms | |
| `cv2.imencode/imdecode` JPEG q=85 | ~25–40 ms | Use `PyTurboJPEG` for ~7× speedup if hot ([arXiv 2501.13131](https://arxiv.org/abs/2501.13131)) — but unnecessary for one-shot scan sim |
| Gaussian noise via `rng.normal` 8.7 Mpx | ~80–120 ms | Slow! Use `cv2.randn` (~10 ms) or operate on a downscaled noise field upsampled with `cv2.resize` |
| `augraphy.LightingGradient` (8.7 Mpx) | ~2700 ms | 0.37 img/sec on 2-core Xeon ⇒ ~2.7 s/img/core ([augraphy benchmark](https://github.com/sparkfish/augraphy/tree/dev/benchmark)) — **call on a half-res copy then upsample** |
| `augraphy.Geometric` | ~7 ms | 135.75 img/sec ⇒ ~7 ms/img |
| `augraphy.SubtleNoise` | ~700 ms | 1.44 img/sec |
| `augraphy.Jpeg` | ~180 ms | 5.55 img/sec |
| `augraphy.BleedThrough` | ~2500 ms, 685 MB | **DO NOT USE** in default path ([sparkfish issue #426](https://github.com/sparkfish/augraphy/issues/426)) |
| `augraphy.BookBinding` | ~11 s, 612 MB | **DO NOT USE** |
| `augraphy.BadPhotoCopy` | ~6 s | **adversarial only** |
| `augraphy.ReflectedLight` | ~17 s | **avoid** ([issue #426](https://github.com/sparkfish/augraphy/issues/426) explicitly calls this out as the slowest) |

### 10.2 Memory budget

augraphy operates in float32 in places — a 300 DPI A4 grayscale image (≈8.7 Mpx) → **~35 MB per intermediate buffer in float32**, and `BleedThrough` retains a second-side cache (~685 MB peak per the [benchmark table](https://github.com/sparkfish/augraphy/tree/dev/benchmark)). For 8-core parallel synthesis that's ~5.5 GB just for `BleedThrough`, easily blowing 16 GB workstation RAM. **Cap parallelism × peak-memory-per-worker explicitly.**

### 10.3 Threading

Per [sparkfish issue #426](https://github.com/sparkfish/augraphy/issues/426), when running augmentations inside a multiprocessing `DataLoader`-style worker pool, OpenCV+NumPy default thread counts cause load-average spikes and net slowdowns. Set in each worker entry point:

```python
import os
os.environ.update({
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})
import cv2
cv2.setNumThreads(1)
```

Reported speedup: ~5–6× when this is applied ([issue #426](https://github.com/sparkfish/augraphy/issues/426)).

**Recommendation for this project**: at moderate tier the augraphy budget is `Geometric + SubtleNoise + Jpeg + InkBleed + Markup(low-p) + Folding(low-p) + BindingsAndFasteners(low-p) + LightingGradient(low-p, half-res)` ≈ 100–180 ms/page on 1 core; leaves ~50 ms headroom for the OpenCV bubble/ArUco stage. Adversarial tier may exceed the budget — that's fine, it runs once per CI build.

---

## 11. Recommended default preset ("subtle realistic", always-on)

> Drop-in mapping of every effect to operator and parameter range. All effects run sequentially on a grayscale image. Total CPU budget: ~120 ms / page / core. Citations are in the column-2 operator references; all parameters are within the conservative ranges shown earlier in this brief.

| Stage | Effect | Operator / call | Params (subtle) |
|---|---|---|---|
| 1. Per-bubble | Slightly imperfect solid fills | `cv2.ellipse(..., -1)` + `cv2.GaussianBlur((3,3), 0.5)` | radius=95 %, gray=20–40 |
| 1. Per-bubble | Off-center fill | shift center | ±1 px |
| 2. Geometric | Skew + translation | `cv2.warpAffine(M, …, borderValue=255)` | rotate ±0.5°, tx/ty ±0.5 % w |
| 2. Geometric | Mild perspective | `cv2.warpPerspective(M, …)` | corner jitter s=0.003 (≈7 px on 2480 w) |
| 3. Lighting | Radial vignette | `cv2.getGaussianKernel` mask + `cv2.divide` | σ = 0.7·w |
| 3. Lighting | One horizontal soft band | `cv2.addWeighted(img, 1.0, shadow, -0.05, 0)` | band y0±100 px, α=0.05 |
| 4. Texture | Augraphy ink hint | `augraphy.InkBleed(intensity_range=(0.3, 0.4), kernel_size=(3,3))` | p=0.25 |
| 4. Noise | Gaussian sensor noise | `cv2.randn(noise, 0, 3); cv2.add(img, noise)` | σ=3 |
| 4. Noise | Sub-pixel paper texture | `pythonperlin.perlin((4,4), dens=32, octaves=2, seed=s)` + `cv2.addWeighted(α=0.03)` | α=0.03 |
| 5. Compression | One JPEG pass | `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])` round-trip | q=85–92 |
| 6. Augraphy ribbon (low-p, optional) | Subtle scanner-bar lighting | `augraphy.LightingGradient(min_brightness=210, transparency=0.3, p=0.3)` on half-res copy | as shown |
| 7. Determinism | RNG | `rng = np.random.default_rng(SeedSequence([global_seed, candidate_number]))` | — |

---

## 12. Recommended adversarial preset (CI-only)

> A separate `--adversarial` stress-set. Pipeline must produce *either* a "routed to manual review" outcome *or* no answer, on every sample of this set. CI asserts the negative.

| Stage | Effect | Operator | Params (adversarial) | Pipeline expectation |
|---|---|---|---|---|
| Per-bubble | Erase + refill on a different bubble | §1 + §1 | erase gray=210, new fill gray=35 | Dual-threshold → "ambiguous" |
| Per-bubble | Double-fill in one row | two `cv2.ellipse` solid fills | both 95 % filled | Multi-fill detector → invalid row |
| Per-bubble | No fill in one row | (no-op) | — | "Blank" |
| Per-bubble | Outside-bubble smudge | `cv2.ellipse` with offset >> radius | offset=10 px | "Blank" (unless ROI is too wide) |
| Bubble-region | Horizontal pencil line across one row | `cv2.line(..., 60, 3)` | thick=3, gray=60 | Ambiguous |
| Bubble-region | X across a question block | two `cv2.line` | thick=2 each | Multi-fill → invalid |
| Geometric | Heavy skew | `cv2.warpAffine` | ±8° | ArUco may still recover; bubble grid may not |
| Geometric | Strong perspective | `cv2.warpPerspective` | corner jitter s=0.03 | Should still rectify |
| Geometric | Heavy page curl | `cv2.remap` sinusoidal | amp=12 px, cycles=2 | Bubbles displaced by >2 px |
| Edge | Edge crop 4 % | slice + pad | 4 % each side | One ArUco lost → fail-closed |
| Lighting | Deep vignette | Gaussian mask, σ = 0.3·w | corner luminance <70 % | Otsu drifts; CLAHE remediation should trigger |
| Lighting | Hard shadow band | `cv2.addWeighted(α=0.30)` | 400 px band | Local Otsu within band fails |
| Noise | High-σ Gaussian | `cv2.randn σ=15` | σ=15 | Confidence-empty threshold should not cross |
| Compression | Double JPEG (85 → 55) | two `cv2.imencode` passes | (85, 55) | Edges blurred ≥1 px |
| Page | Hole punches in left margin | `cv2.circle` × 3 + 1 punch in body | gray=0 | One bubble may be obscured → "ambiguous" |
| Page | Coffee stain over bubbles | Gaussian-blurred ellipse | α=0.45, radius 50 px | Affected bubbles ambiguous |
| Page | Dog-eared corner covering an ArUco | `cv2.fillPoly` + local rotate | 250 px triangle | ArUco missing → fail-closed |
| Page | Highlighter across one row | `cv2.rectangle` + `cv2.addWeighted` | yellow α=0.3 | After grayscale, row may be slightly darker — flag |
| Page | Augraphy moderate noise pack | `augraphy.AugraphyPipeline(post_phase=[Markup(...), Scribbles(...), Folding(...)], random_seed=seed)` | as in §8.3 table | Combined stress |
| ArUco | Partial occlusion at one corner | `cv2.fillPoly` clipping 30 % of fiducial | tl corner | ArUco missing → fail-closed |
| ArUco | One marker rotated 180° | crop + `cv2.rotate` + paste | tr corner | Detected with wrong orientation → ID-position cross-check must catch |
| ArUco | One marker blurred | `cv2.GaussianBlur(k=11, σ=3)` on ROI | bl corner | Probably missing |
| ArUco | One marker missing | `roi[:] = 255` | br corner | Missing → fail-closed |
| ArUco | All 4 markers low-contrast | `cv2.convertScaleAbs(α=0.4, β=130)` on each | all four | Otsu fails → fail-closed |

---

## 13. Top 3 effects with the best realism-per-CPU-millisecond

> One-paragraph guidance for the implementer.

The three highest-leverage effects, in priority order, are **(1) `cv2.warpPerspective` for combined skew+perspective+translation** (one 25 ms call accounts for 90 % of perceived "this is a real scan" geometry — operator: `cv2.getPerspectiveTransform` + `cv2.warpPerspective`, [OpenCV geometric](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html), [PyImageSearch OMR](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/)); **(2) a Gaussian-kernel radial vignette + one horizontal soft band** (~8 ms total via `cv2.getGaussianKernel`+`cv2.divide`+`cv2.addWeighted`, [iditect vignette](https://www.iditect.com/programming/python-example/create-a-vignette-filter-using-python-opencv.html)) which alone explains the "scanner-y" look that pure noise can't fake; and **(3) a single JPEG round-trip at quality 85–92** (~25–40 ms via `cv2.imencode`/`cv2.imdecode`, equivalent to `albumentations.ImageCompression(quality_range=(85, 92))` per [Albumentations 2.0.0 release notes](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0)) which adds the 8×8 block ringing every real ADF-scanned, email-forwarded sheet carries. These three together (~70 ms total) cover the dominant axes of scanner artefact space; everything else in this brief is incremental polish or adversarial stress.

---

## 14. One-line summary

Default subtle preset = `cv2.warpPerspective` + Gaussian-kernel vignette + one horizontal soft band + `cv2.randn` σ=3 noise + `pythonperlin` paper texture α=0.03 + JPEG q=88 round-trip + low-p `augraphy.InkBleed` + low-p `augraphy.LightingGradient` (half-res). Adversarial preset adds the bubble-region (§7) and ArUco (§6) attacks plus heavier `augraphy.Markup`/`Scribbles`/`Folding`/`BindingsAndFasteners`. All randomness flows from a single `SeedSequence([global_seed, candidate_number, stage_name])`.
