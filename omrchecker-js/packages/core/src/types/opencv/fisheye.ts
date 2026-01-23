import type { InputArray, OutputArray, int, Size } from "./_types";

/**
 * Computes the undistortion and rectification maps for the image transform using remap.
 * If D is empty, zero distortion is used. If R or P is empty, identity matrices are used.
 *
 * @param {InputArray} K - Camera intrinsic matrix.
 * @param {InputArray} D - Input vector of distortion coefficients (k1, k2, k3, k4).
 * @param {InputArray} R - Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3 1-channel or 1x1 3-channel.
 * @param {InputArray} P - New camera intrinsic matrix (3x3) or new projection matrix (3x4).
 * @param {Size} size - Undistorted image size.
 * @param {int} m1type - Type of the first output map that can be CV_32FC1 or CV_16SC2. See convertMaps for details.
 * @param {OutputArray} map1 - The first output map.
 * @param {OutputArray} map2 - The second output map.
 * @return {void}
 */
export declare function fisheye_initUndistortRectifyMap(
  K: InputArray,
  D: InputArray,
  R: InputArray,
  P: InputArray,
  size: Size,
  m1type: int,
  map1: OutputArray,
  map2: OutputArray,
): void;
