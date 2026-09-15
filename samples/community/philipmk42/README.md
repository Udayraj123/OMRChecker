# CropOnLMarkers — L-shaped corner marker cropping (issue #270)

Samples demonstrating the `CropOnLMarkers` preprocessor, which crops and
deskews a sheet using its L-shaped corner (anchor) markers.

## Approach

A hybrid strategy keeps detection robust on real-world photos:

1. **Page quadrilateral** is detected as a robust anchor (morphology -> Canny
   -> contour -> 4-point polygon), giving four corners even when markers are
   faint.
2. **Per-corner refinement**: within an anchor region near each page corner,
   an adaptive-threshold pass looks for the L-bracket and snaps the corner
   onto it when found.
3. **Graceful fallback**: if a marker is not confidently found in a corner,
   the page-corner position is used. No corner ever fails.
4. The warp reuses `ImageUtils.four_point_transform`.

## Samples

- `synthetic/` — a generated sheet with bold L-markers on a background.
  Fully reproducible; demonstrates a clean crop.
- `real/` — a photographed hotel survey sheet. The source image is the
  sample attached by the maintainer to issue #270 (papersurvey.io template),
  included here to demonstrate behaviour on a skewed, noisy phone photo.

## Known limitation

On a curled/lifted page corner in a phone photo, deskew is not pixel-tight:
the page-quad step can cut the curled corner slightly. The crop is still
correct and usable; sub-pixel corner accuracy on curled pages is future work.
