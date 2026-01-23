import type { Algorithm, KeyPointVector, Mat, OutputArray } from "./_types";

/**
 * https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html
 */
export declare class Feature2D extends Algorithm {
  /**
   * Detects keypoints and computes the descriptors
   * @param img
   * @param mask
   * @param keypoints
   * @param descriptors
   */
  public detectAndCompute(
    img: Mat,
    mask: Mat,
    keypoints: KeyPointVector,
    descriptors: OutputArray,
  ): void;
}
