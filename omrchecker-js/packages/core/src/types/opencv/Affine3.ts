import type { float_type, int, Mat, Mat3, Mat4, Vec3 } from "./_types";

/**
 * It represents a 4x4 homogeneous transformation matrix `$T$`
 *
 * `\\[T = \\begin{bmatrix} R & t\\\\ 0 & 1\\\\ \\end{bmatrix} \\]`
 *
 * where `$R$` is a 3x3 rotation matrix and `$t$` is a 3x1 translation vector.
 *
 * You can specify `$R$` either by a 3x3 rotation matrix or by a 3x1 rotation vector, which is
 * converted to a 3x3 rotation matrix by the Rodrigues formula.
 *
 * To construct a matrix `$T$` representing first rotation around the axis `$r$` with rotation angle
 * `$|r|$` in radian (right hand rule) and then translation by the vector `$t$`, you can use
 *
 * ```cpp
 * cv::Vec3f r, t;
 * cv::Affine3f T(r, t);
 * ```
 *
 * If you already have the rotation matrix `$R$`, then you can use
 *
 * ```cpp
 * cv::Matx33f R;
 * cv::Affine3f T(R, t);
 * ```
 *
 * To extract the rotation matrix `$R$` from `$T$`, use
 *
 * ```cpp
 * cv::Matx33f R = T.rotation();
 * ```
 *
 * To extract the translation vector `$t$` from `$T$`, use
 *
 * ```cpp
 * cv::Vec3f t = T.translation();
 * ```
 *
 * To extract the rotation vector `$r$` from `$T$`, use
 *
 * ```cpp
 * cv::Vec3f r = T.rvec();
 * ```
 *
 * Note that since the mapping from rotation vectors to rotation matrices is many to one. The returned
 * rotation vector is not necessarily the one you used before to set the matrix.
 *
 * If you have two transformations `$T = T_1 * T_2$`, use
 *
 * ```cpp
 * cv::Affine3f T, T1, T2;
 * T = T2.concatenate(T1);
 * ```
 *
 * To get the inverse transform of `$T$`, use
 *
 * ```cpp
 * cv::Affine3f T, T_inv;
 * T_inv = T.inv();
 * ```
 *
 * Source:
 * [opencv2/core/affine.hpp](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/core/affine.hpp#L129).
 *
 */
export declare class Affine3 {
  public matrix: Mat4;

  public constructor();

  public constructor(affine: Mat4);

  /**
   *   The resulting 4x4 matrix is
   *
   *   `\\[ \\begin{bmatrix} R & t\\\\ 0 & 1\\\\ \\end{bmatrix} \\]`
   *
   * @param R 3x3 rotation matrix.
   *
   * @param t 3x1 translation vector.
   */
  public constructor(R: Mat3, t?: Vec3);

  /**
   *   Rodrigues vector.
   *
   *   The last row of the current matrix is set to [0,0,0,1].
   *
   * @param rvec 3x1 rotation vector. Its direction indicates the rotation axis and its length
   * indicates the rotation angle in radian (using right hand rule).
   *
   * @param t 3x1 translation vector.
   */
  public constructor(rvec: Vec3, t?: Vec3);

  /**
   *   Combines all constructors above. Supports 4x4, 3x4, 3x3, 1x3, 3x1 sizes of data matrix.
   *
   *   The last row of the current matrix is set to [0,0,0,1] when data is not 4x4.
   *
   * @param data 1-channel matrix. when it is 4x4, it is copied to the current matrix and t is not
   * used. When it is 3x4, it is copied to the upper part 3x4 of the current matrix and t is not used.
   * When it is 3x3, it is copied to the upper left 3x3 part of the current matrix. When it is 3x1 or
   * 1x3, it is treated as a rotation vector and the Rodrigues formula is used to compute a 3x3 rotation
   * matrix.
   *
   * @param t 3x1 translation vector. It is used only when data is neither 4x4 nor 3x4.
   */
  public constructor(data: Mat, t?: Vec3);

  public constructor(vals: float_type);

  public cast(arg401: any): Affine3;

  public concatenate(affine: Affine3): Affine3;

  /**
   *   the inverse of the current matrix.
   */
  public inv(method?: int): Affine3;

  /**
   *   Copy the 3x3 matrix L to the upper left part of the current matrix
   *
   *   It sets the upper left 3x3 part of the matrix. The remaining part is unaffected.
   *
   * @param L 3x3 matrix.
   */
  public linear(L: Mat3): Mat3;

  /**
   *   the upper left 3x3 part
   */
  public linear(): Mat3;

  public rotate(R: Mat3): Affine3;

  public rotate(rvec: Vec3): Affine3;

  /**
   *   Rotation matrix.
   *
   *   Copy the rotation matrix to the upper left 3x3 part of the current matrix. The remaining elements
   * of the current matrix are not changed.
   *
   * @param R 3x3 rotation matrix.
   */
  public rotation(R: Mat3): Mat3;

  /**
   *   Rodrigues vector.
   *
   *   It sets the upper left 3x3 part of the matrix. The remaining part is unaffected.
   *
   * @param rvec 3x1 rotation vector. The direction indicates the rotation axis and its length
   * indicates the rotation angle in radian (using the right thumb convention).
   */
  public rotation(rvec: Vec3): Vec3;

  /**
   *   Combines rotation methods above. Supports 3x3, 1x3, 3x1 sizes of data matrix.
   *
   *   It sets the upper left 3x3 part of the matrix. The remaining part is unaffected.
   *
   * @param data 1-channel matrix. When it is a 3x3 matrix, it sets the upper left 3x3 part of the
   * current matrix. When it is a 1x3 or 3x1 matrix, it is used as a rotation vector. The Rodrigues
   * formula is used to compute the rotation matrix and sets the upper left 3x3 part of the current
   * matrix.
   */
  public rotation(data: Mat): Mat;

  /**
   *   the upper left 3x3 part
   */
  public rotation(): Mat3;

  /**
   *   Rodrigues vector.
   *
   *   a vector representing the upper left 3x3 rotation matrix of the current matrix.
   *
   *   Since the mapping between rotation vectors and rotation matrices is many to one, this function
   * returns only one rotation vector that represents the current rotation matrix, which is not
   * necessarily the same one set by `[rotation(const Vec3& rvec)]`.
   */
  public rvec(): Vec3;

  public translate(t: Vec3): Affine3;

  /**
   *   Copy t to the first three elements of the last column of the current matrix
   *
   *   It sets the upper right 3x1 part of the matrix. The remaining part is unaffected.
   *
   * @param t 3x1 translation vector.
   */
  public translation(t: Vec3): Vec3;

  /**
   *   the upper right 3x1 part
   */
  public translation(): Vec3;

  public static Identity(): Affine3;
}
