import type { InputArray, InputOutputArray } from "./_types";
/*
 * # Drawing Function of Keypoints and Matches
 *
 */
/**
 * For Python API, flags are modified as cv.DRAW_MATCHES_FLAGS_DEFAULT,
 * cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
 * cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
 *
 * @param image Source image.
 *
 * @param keypoints Keypoints from the source image.
 *
 * @param outImage Output image. Its content depends on the flags value defining what is drawn in the
 * output image. See possible flags bit values below.
 *
 * @param color Color of keypoints.
 *
 * @param flags Flags setting drawing features. Possible flags bit values are defined by
 * DrawMatchesFlags. See details above in drawMatches .
 */
export declare function drawKeypoints(
  image: InputArray,
  keypoints: any,
  outImage: InputOutputArray,
  color?: any,
  flags?: DrawMatchesFlags,
): void;

/**
 * This function draws matches of keypoints from two images in the output image. Match is a line
 * connecting two keypoints (circles). See [cv::DrawMatchesFlags].
 *
 * @param img1 First source image.
 *
 * @param keypoints1 Keypoints from the first source image.
 *
 * @param img2 Second source image.
 *
 * @param keypoints2 Keypoints from the second source image.
 *
 * @param matches1to2 Matches from the first image to the second one, which means that keypoints1[i]
 * has a corresponding point in keypoints2[matches[i]] .
 *
 * @param outImg Output image. Its content depends on the flags value defining what is drawn in the
 * output image. See possible flags bit values below.
 *
 * @param matchColor Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1) ,
 * the color is generated randomly.
 *
 * @param singlePointColor Color of single keypoints (circles), which means that keypoints do not have
 * the matches. If singlePointColor==Scalar::all(-1) , the color is generated randomly.
 *
 * @param matchesMask Mask determining which matches are drawn. If the mask is empty, all matches are
 * drawn.
 *
 * @param flags Flags setting drawing features. Possible flags bit values are defined by
 * DrawMatchesFlags.
 */
export declare function drawMatches(
  img1: InputArray,
  keypoints1: any,
  img2: InputArray,
  keypoints2: any,
  matches1to2: any,
  outImg: InputOutputArray,
  matchColor?: any,
  singlePointColor?: any,
  matchesMask?: any,
  flags?: DrawMatchesFlags,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function drawMatches(
  img1: InputArray,
  keypoints1: any,
  img2: InputArray,
  keypoints2: any,
  matches1to2: any,
  outImg: InputOutputArray,
  matchColor?: any,
  singlePointColor?: any,
  matchesMask?: any,
  flags?: DrawMatchesFlags,
): void;

/**
 * Output image matrix will be created ([Mat::create]), i.e. existing memory of output image may be
 * reused. Two source image, matches and single keypoints will be drawn. For each keypoint only the
 * center point will be drawn (without the circle around keypoint with keypoint size and orientation).
 *
 */
export declare const DEFAULT: DrawMatchesFlags; // initializer: = 0

/**
 * Output image matrix will not be created ([Mat::create]). Matches will be drawn on existing content
 * of output image.
 *
 */
export declare const DRAW_OVER_OUTIMG: DrawMatchesFlags; // initializer: = 1

export declare const NOT_DRAW_SINGLE_POINTS: DrawMatchesFlags; // initializer: = 2

/**
 * For each keypoint the circle around keypoint with keypoint size and orientation will be drawn.
 *
 */
export declare const DRAW_RICH_KEYPOINTS: DrawMatchesFlags; // initializer: = 4

export type DrawMatchesFlags = any;
