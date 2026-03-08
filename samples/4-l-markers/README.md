# Sample 4 — L-Shaped Markers

This sample demonstrates OMR sheet alignment using **L-shaped corner markers**.

## What are L-shaped markers?

L-shaped markers are printed symbols at each corner of an OMR sheet that form
an "L" (right angle). The inner corner (turning point) of each L is used as
the control point for perspective correction and cropping.

## When to use vs `FOUR_MARKERS`

| | `FOUR_MARKERS` | `L_MARKERS` |
|---|---|---|
| Marker type | Arbitrary image template | L-shaped printed corner |
| Detection method | Template matching (cross-correlation) | Morphology + contour + convexity defect |
| Reference image needed | Yes (`referenceImage`) | No |
| Robust to | Scale/rotation variation | Noise, fills, xerox artifacts |
| Best for | Custom logo/stamp markers | Standard L-corner printed sheets |

## Template config

The `template.json` in this directory uses:
- `type: "L_MARKERS"` — activates `CropOnLMarkers` processor
- `enable_cropping: true` — crops to the bounding box of the 4 detected corners
- `warp_method: "HOMOGRAPHY"` — supports N control points (not limited to 4)
- Each zone preset covers one quadrant of the page

## Usage

```bash
uv run main.py -i samples/4-l-markers/inputs/
```
