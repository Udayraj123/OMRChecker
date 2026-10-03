# Hardcodings in Preprocessors

Survey for issue #85: list hardcoded values in `src/processors/` that should later come from configuration or command line.

Scope: `CropPage.py`, `CropOnMarkers.py`, `FeatureBasedAlignment.py`, `builtins.py`. Values already moved to `src/constants/image_processing.py` are still listed because they are fixed in code, not set from template or config. Line numbers match commit `5cf44a5`.

Related note: `src/core.py:74` says all numbers should come from either constants or config. No TODO tags were found inside `src/processors/`.

## CropPage.py

| File | Line | Value | What it controls | Suggested source |
|------|------|-------|------------------|------------------|
| src/processors/CropPage.py | 26 | `0, 255` in `cv2.normalize` | Normalize output range before page search | `preProcessors.CropPage.normalizeRange` in template options |
| src/processors/CropPage.py | 31, 32 | `max_cosine = 0`, `min_cosine = 1.5` | Initial bounds for rectangle angle check | Internal init, keep as code constant |
| src/processors/CropPage.py | 33 | `range(2, 5)` | Iterates 4 quad vertices in `check_max_cosine` | Structural, no config needed |
| src/processors/CropPage.py | 45, 126 | `== 4`, `reshape(4, 2)`, `reshape(approx, (4, -1))` | Quad vertex count assumption | Structural, no config needed |
| src/processors/CropPage.py | 54 | `1e-10` | Epsilon against divide by zero in `angle` | Code constant in `src/constants/image_processing.py` |
| src/processors/CropPage.py | 63 | `[10, 10]` default `morphKernel` | Closing kernel size for edge gaps | Already in template options as `morphKernel`, keep default in schema |
| src/processors/CropPage.py | 67 | `DEFAULT_GAUSSIAN_BLUR_KERNEL = (3, 3)`, sigma `0` | Pre blur before `find_page` | `preProcessors.CropPage.blurKernel`, `blurSigma` in template options |
| src/processors/CropPage.py | 90-94 | `PAGE_THRESHOLD_PARAMS = 200, 255`, `THRESH_TRUNC` | Truncate bright pixels before Canny | `preProcessors.CropPage.thresholdValue`, `maxPixelValue`, `thresholdType` |
| src/processors/CropPage.py | 103-104 | `CANNY_PARAMS = 185, 55` | Canny edge thresholds, the issue example | `preProcessors.CropPage.cannyParams` such as `self.canny_params` |
| src/processors/CropPage.py | 107 | `show_image_level >= 5` | Debug view level for edge image | `config.json` outputs section, no change needed |
| src/processors/CropPage.py | 112 | `RETR_LIST`, `CHAIN_APPROX_SIMPLE` | Contour retrieval and approximation mode | `preProcessors.CropPage.contourMode` if tuning needed |
| src/processors/CropPage.py | 116 | `[:5]` | Keep 5 largest contours as page candidates | `preProcessors.CropPage.maxContours` |
| src/processors/CropPage.py | 119 | `MIN_PAGE_AREA_THRESHOLD = 80000` | Minimum contour area accepted as page | `preProcessors.CropPage.minPageArea` |
| src/processors/CropPage.py | 122-123 | `APPROX_POLY_EPSILON_FACTOR = 0.025` | Polygon approximation factor times perimeter | `preProcessors.CropPage.approxEpsilonFactor` |
| src/processors/CropPage.py | 38 | `MAX_COSINE_THRESHOLD = 0.35` | Max cosine allowed for rectangle check | `preProcessors.CropPage.maxCosineThreshold` |
| src/processors/CropPage.py | 127-133 | `DEFAULT_CONTOUR_COLOR = (0, 255, 0)`, width `2`, index `-1` | Debug draw of found page contour | `config.json` debug draw style |
| src/processors/CropPage.py | 134-140 | `DEFAULT_CONTOUR_FILL_COLOR = (255, 255, 255)`, width `10` | Fill of found contour on edge image | `config.json` debug draw style |

## CropOnMarkers.py

| File | Line | Value | What it controls | Suggested source |
|------|------|-------|------------------|------------------|
| src/processors/CropOnMarkers.py | 35 | `"omr_marker.jpg"` default `relativePath` | Default marker template file | Already in template options, keep default in schema |
| src/processors/CropOnMarkers.py | 37 | `0.3` default `min_matching_threshold` | Minimum template match score per quadrant | Already in template options, keep |
| src/processors/CropOnMarkers.py | 38 | `0.41` default `max_matching_variation` | Allowed score spread across quadrants | Already in template options, keep |
| src/processors/CropOnMarkers.py | 40 | `(35, 100)` default `marker_rescale_range` | Percent range used in marker scale search | Already in template options, keep |
| src/processors/CropOnMarkers.py | 42 | `10` default `marker_rescale_steps` | Steps used for scale search | Already in template options, keep |
| src/processors/CropOnMarkers.py | 43 | `True` default `apply_erode_subtract` | Whether erode subtract path is used | Already in template options, keep |
| src/processors/CropOnMarkers.py | 62-63, 222-223 | `EROSION_PARAMS = (5, 5)`, iterations `5` | Erode kernel used for marker emphasis | `preProcessors.CropOnMarkers.erodeKernel`, `erodeIterations` |
| src/processors/CropOnMarkers.py | 70-73 | `QUADRANT_DIVISION = 3, 2`, `//` split | Image split point for 4 quadrant search | `preProcessors.CropOnMarkers.quadrantFactors` |
| src/processors/CropOnMarkers.py | 74 | `[[0, 0], [midw, 0], [0, midh], [midw, midh]]` | Quadrant origins | Derived, no config needed |
| src/processors/CropOnMarkers.py | 81-82 | `midw : midw + 2`, `DEFAULT_WHITE_COLOR = 255` | Quad separator lines drawn on debug image | Debug style constant |
| src/processors/CropOnMarkers.py | 97 | `range(0, 4)` | Loop over 4 quadrants | Structural, no config needed |
| src/processors/CropOnMarkers.py | 100 | `round(max_t, 3)` | Log precision for match scores | Logging only, no config needed |
| src/processors/CropOnMarkers.py | 138-143 | `MARKER_RECTANGLE_COLOR = (150, 150, 150)`, `DEFAULT_LINE_WIDTH = 2` | Marker box drawn on output image | `config.json` debug draw style |
| src/processors/CropOnMarkers.py | 146-152 | `ERODE_RECT_COLOR = (50, 50, 50)`, `NORMAL_RECT_COLOR = (155, 155, 155)`, width `4` | Marker box drawn on eroded debug image | `config.json` debug draw style, merge literal `4` into `DEFAULT_MARKER_LINE_WIDTH` |
| src/processors/CropOnMarkers.py | 153 | `w / 2`, `_h / 2` | Marker centre from box corner | Derived, no config needed |
| src/processors/CropOnMarkers.py | 159 | `sum_t / 4` | Mean match score over 4 quadrants | Derived, no config needed |
| src/processors/CropOnMarkers.py | 165 | `append_save_img(2, ...)` | Save level for warped debug image | `config.json` outputs section |
| src/processors/CropOnMarkers.py | 171 | `>= 2` and `< 4` | Show level window for warped view | `config.json` outputs section |
| src/processors/CropOnMarkers.py | 175 | `DEFAULT_BORDER_REMOVE = 5`, `DEFAULT_BLACK_COLOR = 0` | Black strip width on stacked debug view | `config.json` debug style |
| src/processors/CropOnMarkers.py | 180 | `* 1.6` on `display_width` | Resize factor for stacked debug view | `config.json` dimensions section |
| src/processors/CropOnMarkers.py | 197 | `exit(31)` | Exit code when marker file is missing | Keep as code constant or use shared error codes |
| src/processors/CropOnMarkers.py | 204-205 | `processing_width / sheetToMarkerWidthRatio` | Marker resize from sheet ratio option | Already in template options, keep |
| src/processors/CropOnMarkers.py | 207-210 | `DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER = (5, 5)`, sigma `0` | Blur on marker template | `preProcessors.CropOnMarkers.blurKernel`, `blurSigma` |
| src/processors/CropOnMarkers.py | 212-218 | `DEFAULT_NORMALIZE_PARAMS = 0, 255` | Normalize range for marker template | `preProcessors.CropOnMarkers.normalizeRange` |
| src/processors/CropOnMarkers.py | 233-235 | `// self.marker_rescale_steps` | Step size for scale search | Derived from options above, no new config |
| src/processors/CropOnMarkers.py | 243 | `-1 * descent_per_step` | Reverse order of scale search | Structural, no config needed |
| src/processors/CropOnMarkers.py | 245 | `r0 * 1 / 100`, guard `== 0.0` | Percent integer to float scale | Structural, no config needed |

## FeatureBasedAlignment.py

| File | Line | Value | What it controls | Suggested source |
|------|------|-------|------------------|------------------|
| src/processors/FeatureBasedAlignment.py | 32 | `DEFAULT_MAX_FEATURES = 500` | ORB feature count | Already in template options as `maxFeatures`, keep |
| src/processors/FeatureBasedAlignment.py | 33 | `DEFAULT_GOOD_MATCH_PERCENT = 0.15` | Fraction of matches kept | Already in template options as `goodMatchPercent`, keep |
| src/processors/FeatureBasedAlignment.py | 34 | `options.get("2d", False)` | Affine versus homography warp | Already in template options, keep |
| src/processors/FeatureBasedAlignment.py | 53 | `0, 255` in `cv2.normalize` | Normalize range before ORB detect | `preProcessors.FeatureBasedAlignment.normalizeRange` |
| src/processors/FeatureBasedAlignment.py | 59-60 | `DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING` | Matcher type for ORB descriptors | `preProcessors.FeatureBasedAlignment.matcherType` |
| src/processors/FeatureBasedAlignment.py | 72 | `int(len(matches) * percent)` | Count of good matches kept | Derived from `goodMatchPercent`, no new config |
| src/processors/FeatureBasedAlignment.py | 76 | `show_image_level > 2` | Debug view level for match image | `config.json` outputs section |
| src/processors/FeatureBasedAlignment.py | 83-84 | `zeros((len, 2), float32)` | Match point arrays | Structural, no config needed |
| src/processors/FeatureBasedAlignment.py | 97 | `cv2.RANSAC` in `findHomography` | Robust method for homography | `preProcessors.FeatureBasedAlignment.ransacThreshold` group |

## builtins.py

| File | Line | Value | What it controls | Suggested source |
|------|------|-------|------------------|------------------|
| src/processors/builtins.py | 18, 20 | `return 0`, `return 255` | Levels output clamp | Fixed image range, keep as code constant |
| src/processors/builtins.py | 21 | `1.0 / gamma` | Gamma inversion in Levels | Derived from `gamma` option, no new config |
| src/processors/builtins.py | 22 | `* 255` | Levels output scale | Fixed image range, keep as code constant |
| src/processors/builtins.py | 28-30 | `255 * low/high`, defaults `0, 1, 1.0` | Levels input range and gamma | Already in template options as `low`, `high`, `gamma`, keep |
| src/processors/builtins.py | 32 | `arange(0, 256)` | LUT table size | Fixed image range, no config needed |
| src/processors/builtins.py | 44 | `DEFAULT_MEDIAN_BLUR_KERNEL_SIZE = 5` | Median blur kernel | Already in template options as `kSize`, keep |
| src/processors/builtins.py | 54-55 | `DEFAULT_GAUSSIAN_BLUR_PARAMS = (3, 3), 0` | Gaussian blur kernel and sigma | Already in template options as `kSize`, `sigmaX`, keep |

## Centralized but still fixed

These values were already extracted to `src/constants/image_processing.py` and are referenced by the tables above. They remain hardcodings until read from template options or `config.json`. Full list: `MIN_PAGE_AREA_THRESHOLD`, `MAX_COSINE_THRESHOLD`, `DEFAULT_GAUSSIAN_BLUR_KERNEL`, `PAGE_THRESHOLD_PARAMS`, `CANNY_PARAMS`, `APPROX_POLY_EPSILON_FACTOR`, `QUADRANT_DIVISION`, `MARKER_RECTANGLE_COLOR`, `ERODE_RECT_COLOR`, `NORMAL_RECT_COLOR`, `EROSION_PARAMS`, `DEFAULT_MAX_FEATURES`, `DEFAULT_GOOD_MATCH_PERCENT`, `DEFAULT_MEDIAN_BLUR_KERNEL_SIZE`, `DEFAULT_GAUSSIAN_BLUR_PARAMS`, plus debug draw constants.

## Next steps

1. Add the missing template options named in the Suggested source column to `src/schemas/template_schema.py`.
2. Read them with `options.get(name, CONSTANT)` in each preprocessor so old templates keep working.
3. Keep debug only values under `config.json` outputs or dimensions.
