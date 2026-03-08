/**
 * Utilities for patch-based scanning and point detection.
 *
 * Port of Python: src/processors/image/crop_on_patches/patch_utils.py
 * Task: omr-egt
 *
 * Provides reusable utilities for:
 * - Selecting points from rectangles (corners, centers)
 * - Computing and drawing scan zones
 * - Managing edge contours from zone points
 */

/**
 * Select a specific point from a rectangle based on selector type.
 *
 * Port of Python: select_point_from_rectangle
 *
 * @param rectangle - Array of 4 corner points [tl, tr, br, bl]
 * @param pointsSelector - Selector type string (e.g. "SELECT_CENTER", "L_INNER_CORNER")
 * @returns Selected point as [x, y] or null if selector is invalid
 */
export function selectPointFromRectangle(
  rectangle: number[][],
  pointsSelector: string
): number[] | null {
  const [tl, tr, br, bl] = rectangle;
  if (pointsSelector === 'SELECT_TOP_LEFT') return tl;
  if (pointsSelector === 'SELECT_TOP_RIGHT') return tr;
  if (pointsSelector === 'SELECT_BOTTOM_RIGHT') return br;
  if (pointsSelector === 'SELECT_BOTTOM_LEFT') return bl;
  if (pointsSelector === 'SELECT_CENTER') {
    return [Math.floor((tl[0] + br[0]) / 2), Math.floor((tl[1] + br[1]) / 2)];
  }
  if (pointsSelector === 'L_INNER_CORNER') {
    // For L-marker detection, all 4 corners of the degenerate rectangle are
    // the same point (the detected inner corner), so just return tl.
    return tl;
  }
  return null;
}
