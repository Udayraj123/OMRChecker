import type { bool, DescriptorMatcher, int, Ptr } from "./_types";

/**
 * For each descriptor in the first set, this matcher finds the closest descriptor in the second set by
 * trying each one. This descriptor matcher supports masking permissible matches of descriptor sets.
 *
 * Source:
 * [opencv2/features2d.hpp](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/features2d.hpp#L1140).
 *
 */
export declare class BFMatcher extends DescriptorMatcher {
  public constructor(normType?: int, crossCheck?: bool);

  /**
   * @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object,
   * that is, copies both parameters and train data. If emptyTrainData is true, the method creates an
   * object copy with the current parameters but with empty train data.
   */
  public clone(emptyTrainData?: bool): Ptr;

  public isMaskSupported(): bool;

  /**
   * @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
   * preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
   * BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
   * description).
   *
   * @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k
   * nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
   * k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
   * matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
   * pairs. Such technique usually produces best results with minimal number of outliers when there are
   * enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
   */
  public static create(normType?: int, crossCheck?: bool): Ptr;
}
