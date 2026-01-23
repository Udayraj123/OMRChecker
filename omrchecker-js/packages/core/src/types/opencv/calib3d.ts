import type {
  bool,
  double,
  float,
  InputArray,
  InputArrayOfArrays,
  InputOutputArray,
  int,
  Mat,
  OutputArray,
  OutputArrayOfArrays,
  Point2d,
  Rect,
  Size,
  size_t,
  TermCriteria,
  Vec3d,
} from "./_types";
/*
 * # Camera Calibration and 3D Reconstruction
 * The functions in this section use a so-called pinhole camera model. In this model, a scene view is formed by projecting 3D points into the image plane using a perspective transformation.
 *
 * `\[s \; m' = A [R|t] M'\]`
 *
 * or
 *
 * `\[s \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1 \\ r_{21} & r_{22} & r_{23} & t_2 \\ r_{31} & r_{32} & r_{33} & t_3 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}\]`
 *
 * where:
 *
 *
 *
 *
 *
 *  * `$(X, Y, Z)$` are the coordinates of a 3D point in the world coordinate space
 *  * `$(u, v)$` are the coordinates of the projection point in pixels
 *  * `$A$` is a camera matrix, or a matrix of intrinsic parameters
 *  * `$(cx, cy)$` is a principal point that is usually at the image center
 *  * `$fx, fy$` are the focal lengths expressed in pixel units.
 *
 *
 * Thus, if an image from the camera is scaled by a factor, all of these parameters should be scaled (multiplied/divided, respectively) by the same factor. The matrix of intrinsic parameters does not depend on the scene viewed. So, once estimated, it can be re-used as long as the focal length is fixed (in case of zoom lens). The joint rotation-translation matrix `$[R|t]$` is called a matrix of extrinsic parameters. It is used to describe the camera motion around a static scene, or vice versa, rigid motion of an object in front of a still camera. That is, `$[R|t]$` translates coordinates of a point `$(X, Y, Z)$` to a coordinate system, fixed with respect to the camera. The transformation above is equivalent to the following (when `$z \ne 0$` ):
 *
 * `\[\begin{array}{l} \vecthree{x}{y}{z} = R \vecthree{X}{Y}{Z} + t \\ x' = x/z \\ y' = y/z \\ u = f_x*x' + c_x \\ v = f_y*y' + c_y \end{array}\]`
 *
 * The following figure illustrates the pinhole camera model.
 *
 *
 *  Real lenses usually have some distortion, mostly radial distortion and slight tangential distortion. So, the above model is extended as:
 *
 * `\[\begin{array}{l} \vecthree{x}{y}{z} = R \vecthree{X}{Y}{Z} + t \\ x' = x/z \\ y' = y/z \\ x'' = x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2) + s_1 r^2 + s_2 r^4 \\ y'' = y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\ \text{where} \quad r^2 = x'^2 + y'^2 \\ u = f_x*x'' + c_x \\ v = f_y*y'' + c_y \end{array}\]`
 *
 * `$k_1$`, `$k_2$`, `$k_3$`, `$k_4$`, `$k_5$`, and `$k_6$` are radial distortion coefficients. `$p_1$` and `$p_2$` are tangential distortion coefficients. `$s_1$`, `$s_2$`, `$s_3$`, and `$s_4$`, are the thin prism distortion coefficients. Higher-order coefficients are not considered in OpenCV.
 *
 * The next figures show two common types of radial distortion: barrel distortion (typically `$ k_1 < 0 $`) and pincushion distortion (typically `$ k_1 > 0 $`).
 *
 *
 *
 *
 *
 * In some cases the image sensor may be tilted in order to focus an oblique plane in front of the camera (Scheimpfug condition). This can be useful for particle image velocimetry (PIV) or triangulation with a laser fan. The tilt causes a perspective distortion of `$x''$` and `$y''$`. This distortion can be modelled in the following way, see e.g. Louhichi07.
 *
 * `\[\begin{array}{l} s\vecthree{x'''}{y'''}{1} = \vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}(\tau_x, \tau_y)} {0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)} {0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\ u = f_x*x''' + c_x \\ v = f_y*y''' + c_y \end{array}\]`
 *
 * where the matrix `$R(\tau_x, \tau_y)$` is defined by two rotations with angular parameter `$\tau_x$` and `$\tau_y$`, respectively,
 *
 * `\[ R(\tau_x, \tau_y) = \vecthreethree{\cos(\tau_y)}{0}{-\sin(\tau_y)}{0}{1}{0}{\sin(\tau_y)}{0}{\cos(\tau_y)} \vecthreethree{1}{0}{0}{0}{\cos(\tau_x)}{\sin(\tau_x)}{0}{-\sin(\tau_x)}{\cos(\tau_x)} = \vecthreethree{\cos(\tau_y)}{\sin(\tau_y)\sin(\tau_x)}{-\sin(\tau_y)\cos(\tau_x)} {0}{\cos(\tau_x)}{\sin(\tau_x)} {\sin(\tau_y)}{-\cos(\tau_y)\sin(\tau_x)}{\cos(\tau_y)\cos(\tau_x)}. \]`
 *
 * In the functions below the coefficients are passed or returned as
 *
 * `\[(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\]`
 *
 * vector. That is, if the vector contains four elements, it means that `$k_3=0$` . The distortion coefficients do not depend on the scene viewed. Thus, they also belong to the intrinsic camera parameters. And they remain the same regardless of the captured image resolution. If, for example, a camera has been calibrated on images of 320 x 240 resolution, absolutely the same distortion coefficients can be used for 640 x 480 images from the same camera while `$f_x$`, `$f_y$`, `$c_x$`, and `$c_y$` need to be scaled appropriately.
 *
 * The functions below use the above model to do the following:
 *
 *
 *
 *
 *
 *  * Project 3D points to the image plane given intrinsic and extrinsic parameters.
 *  * Compute extrinsic parameters given intrinsic parameters, a few 3D points, and their projections.
 *  * Estimate intrinsic and extrinsic camera parameters from several views of a known calibration pattern (every view is described by several 3D-2D point correspondences).
 *  * Estimate the relative position and orientation of the stereo camera "heads" and compute the rectification* transformation that makes the camera optical axes parallel.
 *
 *
 *
 *
 *
 *
 *
 *
 *  * A calibration sample for 3 cameras in horizontal position can be found at opencv_source_code/samples/cpp/3calibration.cpp
 *  * A calibration sample based on a sequence of images can be found at opencv_source_code/samples/cpp/calibration.cpp
 *  * A calibration sample in order to do 3D reconstruction can be found at opencv_source_code/samples/cpp/build3dmodel.cpp
 *  * A calibration example on stereo calibration can be found at opencv_source_code/samples/cpp/stereo_calib.cpp
 *  * A calibration example on stereo matching can be found at opencv_source_code/samples/cpp/stereo_match.cpp
 *  * (Python) A camera calibration sample can be found at opencv_source_code/samples/python/calibrate.py
 */
/**
 * the overall RMS re-projection error.
 * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
 * views. The algorithm is based on Zhang2000 and BouguetMCT . The coordinates of 3D object points and
 * their corresponding 2D projections in each view must be specified. That may be achieved by using an
 * object with a known geometry and easily detectable feature points. Such an object is called a
 * calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as a
 * calibration rig (see findChessboardCorners ). Currently, initialization of intrinsic parameters
 * (when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration patterns
 * (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also be used
 * as long as initial cameraMatrix is provided.
 *
 * The algorithm performs the following steps:
 *
 * Compute the initial intrinsic parameters (the option only available for planar calibration patterns)
 * or read them from the input parameters. The distortion coefficients are all set to zeros initially
 * unless some of CALIB_FIX_K? are specified.
 * Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
 * done using solvePnP .
 * Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error, that
 * is, the total sum of squared distances between the observed feature points imagePoints and the
 * projected (using the current estimates for camera parameters and the poses) object points
 * objectPoints. See projectPoints for details.
 *
 * If you use a non-square (=non-NxN) grid and findChessboardCorners for calibration, and
 * calibrateCamera returns bad values (zero distortion coefficients, an image center very far from
 * (w/2-0.5,h/2-0.5), and/or large differences between `$f_x$` and `$f_y$` (ratios of 10:1 or more)),
 * then you have probably used patternSize=cvSize(rows,cols) instead of using
 * patternSize=cvSize(cols,rows) in findChessboardCorners .
 *
 * [calibrateCameraRO], [findChessboardCorners], [solvePnP], [initCameraMatrix2D], [stereoCalibrate],
 * [undistort]
 *
 * @param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
 * the calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer
 * vector contains as many elements as the number of the pattern views. If the same calibration pattern
 * is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
 * possible to use partially occluded patterns, or even different patterns in different views. Then,
 * the vectors will be different. The points are 3D, but since they are in a pattern coordinate system,
 * then, if the rig is planar, it may make sense to put the model to a XY coordinate plane so that
 * Z-coordinate of each input object point is 0. In the old interface all the vectors of object points
 * from different views are concatenated together.
 *
 * @param imagePoints In the new interface it is a vector of vectors of the projections of calibration
 * pattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and
 * objectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i. In
 * the old interface all the vectors of object points from different views are concatenated together.
 *
 * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
 *
 * @param cameraMatrix Output 3x3 floating-point camera matrix $A =
 * \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ . If CV_CALIB_USE_INTRINSIC_GUESS and/or
 * CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be initialized before
 * calling the function.
 *
 * @param distCoeffs Output vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5,
 * k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements.
 *
 * @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view
 * (e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
 * k-th translation vector (see the next output parameter description) brings the calibration pattern
 * from the model coordinate space (in which object points are specified) to the world coordinate
 * space, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. M -1).
 *
 * @param tvecs Output vector of translation vectors estimated for each pattern view.
 *
 * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic
 * parameters. Order of deviations values: $(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6
 * , s_1, s_2, s_3, s_4, \tau_x, \tau_y)$ If one of parameters is not estimated, it's deviation is
 * equals to zero.
 *
 * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic
 * parameters. Order of deviations values: $(R_1, T_1, \dotsc , R_M, T_M)$ where M is number of pattern
 * views, $R_i, T_i$ are concatenated 1x3 vectors.
 *
 * @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
 *
 * @param flags Different flags that may be zero or a combination of the following values:
 * CALIB_USE_INTRINSIC_GUESS cameraMatrix contains valid initial values of fx, fy, cx, cy that are
 * optimized further. Otherwise, (cx, cy) is initially set to the image center ( imageSize is used),
 * and focal distances are computed in a least-squares fashion. Note, that if intrinsic parameters are
 * known, there is no need to use this function just to estimate extrinsic parameters. Use solvePnP
 * instead.CALIB_FIX_PRINCIPAL_POINT The principal point is not changed during the global optimization.
 * It stays at the center or at a different location specified when CALIB_USE_INTRINSIC_GUESS is set
 * too.CALIB_FIX_ASPECT_RATIO The functions considers only fy as a free parameter. The ratio fx/fy
 * stays the same as in the input cameraMatrix . When CALIB_USE_INTRINSIC_GUESS is not set, the actual
 * input values of fx and fy are ignored, only their ratio is computed and used
 * further.CALIB_ZERO_TANGENT_DIST Tangential distortion coefficients $(p_1, p_2)$ are set to zeros and
 * stay zero.CALIB_FIX_K1,...,CALIB_FIX_K6 The corresponding radial distortion coefficient is not
 * changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
 * supplied distCoeffs matrix is used. Otherwise, it is set to 0.CALIB_RATIONAL_MODEL Coefficients k4,
 * k5, and k6 are enabled. To provide the backward compatibility, this extra flag should be explicitly
 * specified to make the calibration function use the rational model and return 8 coefficients. If the
 * flag is not set, the function computes and returns only 5 distortion
 * coefficients.CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the
 * backward compatibility, this extra flag should be explicitly specified to make the calibration
 * function use the thin prism model and return 12 coefficients. If the flag is not set, the function
 * computes and returns only 5 distortion coefficients.CALIB_FIX_S1_S2_S3_S4 The thin prism distortion
 * coefficients are not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the
 * coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to
 * 0.CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the backward compatibility,
 * this extra flag should be explicitly specified to make the calibration function use the tilted
 * sensor model and return 14 coefficients. If the flag is not set, the function computes and returns
 * only 5 distortion coefficients.CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are
 * not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
 * supplied distCoeffs matrix is used. Otherwise, it is set to 0.
 *
 * @param criteria Termination criteria for the iterative optimization algorithm.
 */
export declare function calibrateCamera(
  objectPoints: InputArrayOfArrays,
  imagePoints: InputArrayOfArrays,
  imageSize: Size,
  cameraMatrix: InputOutputArray,
  distCoeffs: InputOutputArray,
  rvecs: OutputArrayOfArrays,
  tvecs: OutputArrayOfArrays,
  stdDeviationsIntrinsics: OutputArray,
  stdDeviationsExtrinsics: OutputArray,
  perViewErrors: OutputArray,
  flags?: int,
  criteria?: TermCriteria,
): double;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function calibrateCamera(
  objectPoints: InputArrayOfArrays,
  imagePoints: InputArrayOfArrays,
  imageSize: Size,
  cameraMatrix: InputOutputArray,
  distCoeffs: InputOutputArray,
  rvecs: OutputArrayOfArrays,
  tvecs: OutputArrayOfArrays,
  flags?: int,
  criteria?: TermCriteria,
): double;

/**
 * This function is an extension of [calibrateCamera()] with the method of releasing object which was
 * proposed in strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar targets
 * (calibration plates), this method can dramatically improve the precision of the estimated camera
 * parameters. Both the object-releasing method and standard method are supported by this function. Use
 * the parameter **iFixedPoint** for method selection. In the internal implementation,
 * [calibrateCamera()] is a wrapper for this function.
 *
 * the overall RMS re-projection error.
 * The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
 * views. The algorithm is based on Zhang2000, BouguetMCT and strobl2011iccv. See [calibrateCamera()]
 * for other detailed explanations.
 *
 * [calibrateCamera], [findChessboardCorners], [solvePnP], [initCameraMatrix2D], [stereoCalibrate],
 * [undistort]
 *
 * @param objectPoints Vector of vectors of calibration pattern points in the calibration pattern
 * coordinate space. See calibrateCamera() for details. If the method of releasing object to be used,
 * the identical calibration board must be used in each view and it must be fully visible, and all
 * objectPoints[i] must be the same and all points should be roughly close to a plane. The calibration
 * target has to be rigid, or at least static if the camera (rather than the calibration target) is
 * shifted for grabbing images.
 *
 * @param imagePoints Vector of vectors of the projections of calibration pattern points. See
 * calibrateCamera() for details.
 *
 * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
 *
 * @param iFixedPoint The index of the 3D object point in objectPoints[0] to be fixed. It also acts as
 * a switch for calibration method selection. If object-releasing method to be used, pass in the
 * parameter in the range of [1, objectPoints[0].size()-2], otherwise a value out of this range will
 * make standard calibration method selected. Usually the top-right corner point of the calibration
 * board grid is recommended to be fixed when object-releasing method being utilized. According to
 * strobl2011iccv, two other points are also fixed. In this implementation, objectPoints[0].front and
 * objectPoints[0].back.z are used. With object-releasing method, accurate rvecs, tvecs and
 * newObjPoints are only possible if coordinates of these three fixed points are accurate enough.
 *
 * @param cameraMatrix Output 3x3 floating-point camera matrix. See calibrateCamera() for details.
 *
 * @param distCoeffs Output vector of distortion coefficients. See calibrateCamera() for details.
 *
 * @param rvecs Output vector of rotation vectors estimated for each pattern view. See
 * calibrateCamera() for details.
 *
 * @param tvecs Output vector of translation vectors estimated for each pattern view.
 *
 * @param newObjPoints The updated output vector of calibration pattern points. The coordinates might
 * be scaled based on three fixed points. The returned coordinates are accurate only if the above
 * mentioned three fixed points are accurate. If not needed, noArray() can be passed in. This parameter
 * is ignored with standard calibration method.
 *
 * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic
 * parameters. See calibrateCamera() for details.
 *
 * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic
 * parameters. See calibrateCamera() for details.
 *
 * @param stdDeviationsObjPoints Output vector of standard deviations estimated for refined coordinates
 * of calibration pattern points. It has the same size and order as objectPoints[0] vector. This
 * parameter is ignored with standard calibration method.
 *
 * @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
 *
 * @param flags Different flags that may be zero or a combination of some predefined values. See
 * calibrateCamera() for details. If the method of releasing object is used, the calibration time may
 * be much longer. CALIB_USE_QR or CALIB_USE_LU could be used for faster calibration with potentially
 * less precise and less stable in some rare cases.
 *
 * @param criteria Termination criteria for the iterative optimization algorithm.
 */
export declare function calibrateCameraRO(
  objectPoints: InputArrayOfArrays,
  imagePoints: InputArrayOfArrays,
  imageSize: Size,
  iFixedPoint: int,
  cameraMatrix: InputOutputArray,
  distCoeffs: InputOutputArray,
  rvecs: OutputArrayOfArrays,
  tvecs: OutputArrayOfArrays,
  newObjPoints: OutputArray,
  stdDeviationsIntrinsics: OutputArray,
  stdDeviationsExtrinsics: OutputArray,
  stdDeviationsObjPoints: OutputArray,
  perViewErrors: OutputArray,
  flags?: int,
  criteria?: TermCriteria,
): double;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function calibrateCameraRO(
  objectPoints: InputArrayOfArrays,
  imagePoints: InputArrayOfArrays,
  imageSize: Size,
  iFixedPoint: int,
  cameraMatrix: InputOutputArray,
  distCoeffs: InputOutputArray,
  rvecs: OutputArrayOfArrays,
  tvecs: OutputArrayOfArrays,
  newObjPoints: OutputArray,
  flags?: int,
  criteria?: TermCriteria,
): double;

/**
 * The function performs the Hand-Eye calibration using various methods. One approach consists in
 * estimating the rotation then the translation (separable solutions) and the following methods are
 * implemented:
 *
 * R. Tsai, R. Lenz A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/EyeCalibration
 * Tsai89
 * F. Park, B. Martin Robot Sensor Calibration: Solving AX = XB on the Euclidean Group Park94
 * R. Horaud, F. Dornaika Hand-Eye Calibration Horaud95
 *
 * Another approach consists in estimating simultaneously the rotation and the translation
 * (simultaneous solutions), with the following implemented method:
 *
 * N. Andreff, R. Horaud, B. Espiau On-line Hand-Eye Calibration Andreff99
 * K. Daniilidis Hand-Eye Calibration Using Dual Quaternions Daniilidis98
 *
 * The following picture describes the Hand-Eye calibration problem where the transformation between a
 * camera ("eye") mounted on a robot gripper ("hand") has to be estimated.
 *
 * The calibration procedure is the following:
 *
 * a static calibration pattern is used to estimate the transformation between the target frame and the
 * camera frame
 * the robot gripper is moved in order to acquire several poses
 * for each pose, the homogeneous transformation between the gripper frame and the robot base frame is
 * recorded using for instance the robot kinematics `\\[ \\begin{bmatrix} X_b\\\\ Y_b\\\\ Z_b\\\\ 1
 * \\end{bmatrix} = \\begin{bmatrix} _{}^{b}\\textrm{R}_g & _{}^{b}\\textrm{t}_g \\\\ 0_{1 \\times 3} &
 * 1 \\end{bmatrix} \\begin{bmatrix} X_g\\\\ Y_g\\\\ Z_g\\\\ 1 \\end{bmatrix} \\]`
 * for each pose, the homogeneous transformation between the calibration target frame and the camera
 * frame is recorded using for instance a pose estimation method (PnP) from 2D-3D point correspondences
 * `\\[ \\begin{bmatrix} X_c\\\\ Y_c\\\\ Z_c\\\\ 1 \\end{bmatrix} = \\begin{bmatrix}
 * _{}^{c}\\textrm{R}_t & _{}^{c}\\textrm{t}_t \\\\ 0_{1 \\times 3} & 1 \\end{bmatrix} \\begin{bmatrix}
 * X_t\\\\ Y_t\\\\ Z_t\\\\ 1 \\end{bmatrix} \\]`
 *
 * The Hand-Eye calibration procedure returns the following homogeneous transformation `\\[
 * \\begin{bmatrix} X_g\\\\ Y_g\\\\ Z_g\\\\ 1 \\end{bmatrix} = \\begin{bmatrix} _{}^{g}\\textrm{R}_c &
 * _{}^{g}\\textrm{t}_c \\\\ 0_{1 \\times 3} & 1 \\end{bmatrix} \\begin{bmatrix} X_c\\\\ Y_c\\\\
 * Z_c\\\\ 1 \\end{bmatrix} \\]`
 *
 * This problem is also known as solving the `$\\mathbf{A}\\mathbf{X}=\\mathbf{X}\\mathbf{B}$`
 * equation: `\\[ \\begin{align*} ^{b}{\\textrm{T}_g}^{(1)} \\hspace{0.2em} ^{g}\\textrm{T}_c
 * \\hspace{0.2em} ^{c}{\\textrm{T}_t}^{(1)} &= \\hspace{0.1em} ^{b}{\\textrm{T}_g}^{(2)}
 * \\hspace{0.2em} ^{g}\\textrm{T}_c \\hspace{0.2em} ^{c}{\\textrm{T}_t}^{(2)} \\\\
 * (^{b}{\\textrm{T}_g}^{(2)})^{-1} \\hspace{0.2em} ^{b}{\\textrm{T}_g}^{(1)} \\hspace{0.2em}
 * ^{g}\\textrm{T}_c &= \\hspace{0.1em} ^{g}\\textrm{T}_c \\hspace{0.2em} ^{c}{\\textrm{T}_t}^{(2)}
 * (^{c}{\\textrm{T}_t}^{(1)})^{-1} \\\\ \\textrm{A}_i \\textrm{X} &= \\textrm{X} \\textrm{B}_i \\\\
 * \\end{align*} \\]`
 *
 * Additional information can be found on this .
 *
 * A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye
 * transformation. So at least 3 different poses are required, but it is strongly recommended to use
 * many more poses.
 *
 * @param R_gripper2base Rotation part extracted from the homogeneous matrix that transforms a point
 * expressed in the gripper frame to the robot base frame ( $_{}^{b}\textrm{T}_g$). This is a vector
 * (vector<Mat>) that contains the rotation matrices for all the transformations from gripper frame to
 * robot base frame.
 *
 * @param t_gripper2base Translation part extracted from the homogeneous matrix that transforms a point
 * expressed in the gripper frame to the robot base frame ( $_{}^{b}\textrm{T}_g$). This is a vector
 * (vector<Mat>) that contains the translation vectors for all the transformations from gripper frame
 * to robot base frame.
 *
 * @param R_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
 * expressed in the target frame to the camera frame ( $_{}^{c}\textrm{T}_t$). This is a vector
 * (vector<Mat>) that contains the rotation matrices for all the transformations from calibration
 * target frame to camera frame.
 *
 * @param t_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
 * expressed in the target frame to the camera frame ( $_{}^{c}\textrm{T}_t$). This is a vector
 * (vector<Mat>) that contains the translation vectors for all the transformations from calibration
 * target frame to camera frame.
 *
 * @param R_cam2gripper Estimated rotation part extracted from the homogeneous matrix that transforms a
 * point expressed in the camera frame to the gripper frame ( $_{}^{g}\textrm{T}_c$).
 *
 * @param t_cam2gripper Estimated translation part extracted from the homogeneous matrix that
 * transforms a point expressed in the camera frame to the gripper frame ( $_{}^{g}\textrm{T}_c$).
 *
 * @param method One of the implemented Hand-Eye calibration method, see cv::HandEyeCalibrationMethod
 */
export declare function calibrateHandEye(
  R_gripper2base: InputArrayOfArrays,
  t_gripper2base: InputArrayOfArrays,
  R_target2cam: InputArrayOfArrays,
  t_target2cam: InputArrayOfArrays,
  R_cam2gripper: OutputArray,
  t_cam2gripper: OutputArray,
  method?: HandEyeCalibrationMethod,
): void;

/**
 * The function computes various useful camera characteristics from the previously estimated camera
 * matrix.
 *
 * Do keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for the
 * chessboard pitch (it can thus be any value).
 *
 * @param cameraMatrix Input camera matrix that can be estimated by calibrateCamera or stereoCalibrate
 * .
 *
 * @param imageSize Input image size in pixels.
 *
 * @param apertureWidth Physical width in mm of the sensor.
 *
 * @param apertureHeight Physical height in mm of the sensor.
 *
 * @param fovx Output field of view in degrees along the horizontal sensor axis.
 *
 * @param fovy Output field of view in degrees along the vertical sensor axis.
 *
 * @param focalLength Focal length of the lens in mm.
 *
 * @param principalPoint Principal point in mm.
 *
 * @param aspectRatio $f_y/f_x$
 */
export declare function calibrationMatrixValues(
  cameraMatrix: InputArray,
  imageSize: Size,
  apertureWidth: double,
  apertureHeight: double,
  fovx: any,
  fovy: any,
  focalLength: any,
  principalPoint: any,
  aspectRatio: any,
): void;

export declare function checkChessboard(img: InputArray, size: Size): bool;

/**
 * The functions compute:
 *
 * `\\[\\begin{array}{l} \\texttt{rvec3} = \\mathrm{rodrigues} ^{-1} \\left ( \\mathrm{rodrigues} (
 * \\texttt{rvec2} ) \\cdot \\mathrm{rodrigues} ( \\texttt{rvec1} ) \\right ) \\\\ \\texttt{tvec3} =
 * \\mathrm{rodrigues} ( \\texttt{rvec2} ) \\cdot \\texttt{tvec1} + \\texttt{tvec2} \\end{array} ,\\]`
 *
 * where `$\\mathrm{rodrigues}$` denotes a rotation vector to a rotation matrix transformation, and
 * `$\\mathrm{rodrigues}^{-1}$` denotes the inverse transformation. See Rodrigues for details.
 *
 * Also, the functions can compute the derivatives of the output vectors with regards to the input
 * vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
 * your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
 * function that contains a matrix multiplication.
 *
 * @param rvec1 First rotation vector.
 *
 * @param tvec1 First translation vector.
 *
 * @param rvec2 Second rotation vector.
 *
 * @param tvec2 Second translation vector.
 *
 * @param rvec3 Output rotation vector of the superposition.
 *
 * @param tvec3 Output translation vector of the superposition.
 *
 * @param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
 *
 * @param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
 *
 * @param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
 *
 * @param dr3dt2 Optional output derivative of rvec3 with regard to tvec2
 *
 * @param dt3dr1 Optional output derivative of tvec3 with regard to rvec1
 *
 * @param dt3dt1 Optional output derivative of tvec3 with regard to tvec1
 *
 * @param dt3dr2 Optional output derivative of tvec3 with regard to rvec2
 *
 * @param dt3dt2 Optional output derivative of tvec3 with regard to tvec2
 */
export declare function composeRT(
  rvec1: InputArray,
  tvec1: InputArray,
  rvec2: InputArray,
  tvec2: InputArray,
  rvec3: OutputArray,
  tvec3: OutputArray,
  dr3dr1?: OutputArray,
  dr3dt1?: OutputArray,
  dr3dr2?: OutputArray,
  dr3dt2?: OutputArray,
  dt3dr1?: OutputArray,
  dt3dt1?: OutputArray,
  dt3dr2?: OutputArray,
  dt3dt2?: OutputArray,
): void;

/**
 * For every point in one of the two images of a stereo pair, the function finds the equation of the
 * corresponding epipolar line in the other image.
 *
 * From the fundamental matrix definition (see findFundamentalMat ), line `$l^{(2)}_i$` in the second
 * image for the point `$p^{(1)}_i$` in the first image (when whichImage=1 ) is computed as:
 *
 * `\\[l^{(2)}_i = F p^{(1)}_i\\]`
 *
 * And vice versa, when whichImage=2, `$l^{(1)}_i$` is computed from `$p^{(2)}_i$` as:
 *
 * `\\[l^{(1)}_i = F^T p^{(2)}_i\\]`
 *
 * Line coefficients are defined up to a scale. They are normalized so that `$a_i^2+b_i^2=1$` .
 *
 * @param points Input points. $N \times 1$ or $1 \times N$ matrix of type CV_32FC2 or vector<Point2f>
 * .
 *
 * @param whichImage Index of the image (1 or 2) that contains the points .
 *
 * @param F Fundamental matrix that can be estimated using findFundamentalMat or stereoRectify .
 *
 * @param lines Output vector of the epipolar lines corresponding to the points in the other image.
 * Each line $ax + by + c=0$ is encoded by 3 numbers $(a, b, c)$ .
 */
export declare function computeCorrespondEpilines(
  points: InputArray,
  whichImage: int,
  F: InputArray,
  lines: OutputArray,
): void;

/**
 * The function converts points homogeneous to Euclidean space using perspective projection. That is,
 * each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the
 * output point coordinates will be (0,0,0,...).
 *
 * @param src Input vector of N-dimensional points.
 *
 * @param dst Output vector of N-1-dimensional points.
 */
export declare function convertPointsFromHomogeneous(
  src: InputArray,
  dst: OutputArray,
): void;

/**
 * The function converts 2D or 3D points from/to homogeneous coordinates by calling either
 * convertPointsToHomogeneous or convertPointsFromHomogeneous.
 *
 * The function is obsolete. Use one of the previous two functions instead.
 *
 * @param src Input array or vector of 2D, 3D, or 4D points.
 *
 * @param dst Output vector of 2D, 3D, or 4D points.
 */
export declare function convertPointsHomogeneous(
  src: InputArray,
  dst: OutputArray,
): void;

/**
 * The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of
 * point coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).
 *
 * @param src Input vector of N-dimensional points.
 *
 * @param dst Output vector of N+1-dimensional points.
 */
export declare function convertPointsToHomogeneous(
  src: InputArray,
  dst: OutputArray,
): void;

/**
 * The function implements the Optimal Triangulation Method (see Multiple View Geometry for details).
 * For each given point correspondence points1[i] <-> points2[i], and a fundamental matrix F, it
 * computes the corrected correspondences newPoints1[i] <-> newPoints2[i] that minimize the geometric
 * error `$d(points1[i], newPoints1[i])^2 + d(points2[i],newPoints2[i])^2$` (where `$d(a,b)$` is the
 * geometric distance between points `$a$` and `$b$` ) subject to the epipolar constraint
 * `$newPoints2^T * F * newPoints1 = 0$` .
 *
 * @param F 3x3 fundamental matrix.
 *
 * @param points1 1xN array containing the first set of points.
 *
 * @param points2 1xN array containing the second set of points.
 *
 * @param newPoints1 The optimized points1.
 *
 * @param newPoints2 The optimized points2.
 */
export declare function correctMatches(
  F: InputArray,
  points1: InputArray,
  points2: InputArray,
  newPoints1: OutputArray,
  newPoints2: OutputArray,
): void;

/**
 * This function decompose an essential matrix E using svd decomposition HartleyZ00 . Generally 4
 * possible poses exists for a given E. They are `$[R_1, t]$`, `$[R_1, -t]$`, `$[R_2, t]$`, `$[R_2,
 * -t]$`. By decomposing E, you can only get the direction of the translation, so the function returns
 * unit t.
 *
 * @param E The input essential matrix.
 *
 * @param R1 One possible rotation matrix.
 *
 * @param R2 Another possible rotation matrix.
 *
 * @param t One possible translation.
 */
export declare function decomposeEssentialMat(
  E: InputArray,
  R1: OutputArray,
  R2: OutputArray,
  t: OutputArray,
): void;

/**
 * This function extracts relative camera motion between two views observing a planar object from the
 * homography H induced by the plane. The intrinsic camera matrix K must also be provided. The function
 * may return up to four mathematical solution sets. At least two of the solutions may further be
 * invalidated if point correspondences are available by applying positive depth constraint (all points
 * must be in front of the camera). The decomposition method is described in detail in Malis .
 *
 * @param H The input homography matrix between two images.
 *
 * @param K The input intrinsic camera calibration matrix.
 *
 * @param rotations Array of rotation matrices.
 *
 * @param translations Array of translation matrices.
 *
 * @param normals Array of plane normal matrices.
 */
export declare function decomposeHomographyMat(
  H: InputArray,
  K: InputArray,
  rotations: OutputArrayOfArrays,
  translations: OutputArrayOfArrays,
  normals: OutputArrayOfArrays,
): int;

/**
 * The function computes a decomposition of a projection matrix into a calibration and a rotation
 * matrix and the position of a camera.
 *
 * It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
 * be used in OpenGL. Note, there is always more than one sequence of rotations about the three
 * principal axes that results in the same orientation of an object, e.g. see Slabaugh . Returned tree
 * rotation matrices and corresponding three Euler angles are only one of the possible solutions.
 *
 * The function is based on RQDecomp3x3 .
 *
 * @param projMatrix 3x4 input projection matrix P.
 *
 * @param cameraMatrix Output 3x3 camera matrix K.
 *
 * @param rotMatrix Output 3x3 external rotation matrix R.
 *
 * @param transVect Output 4x1 translation vector T.
 *
 * @param rotMatrixX Optional 3x3 rotation matrix around x-axis.
 *
 * @param rotMatrixY Optional 3x3 rotation matrix around y-axis.
 *
 * @param rotMatrixZ Optional 3x3 rotation matrix around z-axis.
 *
 * @param eulerAngles Optional three-element vector containing three Euler angles of rotation in
 * degrees.
 */
export declare function decomposeProjectionMatrix(
  projMatrix: InputArray,
  cameraMatrix: OutputArray,
  rotMatrix: OutputArray,
  transVect: OutputArray,
  rotMatrixX?: OutputArray,
  rotMatrixY?: OutputArray,
  rotMatrixZ?: OutputArray,
  eulerAngles?: OutputArray,
): void;

/**
 * The function draws individual chessboard corners detected either as red circles if the board was not
 * found, or as colored corners connected with lines if the board was found.
 *
 * @param image Destination image. It must be an 8-bit color image.
 *
 * @param patternSize Number of inner corners per a chessboard row and column (patternSize =
 * cv::Size(points_per_row,points_per_column)).
 *
 * @param corners Array of detected corners, the output of findChessboardCorners.
 *
 * @param patternWasFound Parameter indicating whether the complete board was found or not. The return
 * value of findChessboardCorners should be passed here.
 */
export declare function drawChessboardCorners(
  image: InputOutputArray,
  patternSize: Size,
  corners: InputArray,
  patternWasFound: bool,
): void;

/**
 * [solvePnP]
 *
 * This function draws the axes of the world/object coordinate system w.r.t. to the camera frame. OX is
 * drawn in red, OY in green and OZ in blue.
 *
 * @param image Input/output image. It must have 1 or 3 channels. The number of channels is not
 * altered.
 *
 * @param cameraMatrix Input 3x3 floating-point matrix of camera intrinsic parameters. $A =
 * \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is empty,
 * the zero distortion coefficients are assumed.
 *
 * @param rvec Rotation vector (see Rodrigues ) that, together with tvec, brings points from the model
 * coordinate system to the camera coordinate system.
 *
 * @param tvec Translation vector.
 *
 * @param length Length of the painted axes in the same unit than tvec (usually in meters).
 *
 * @param thickness Line thickness of the painted axes.
 */
export declare function drawFrameAxes(
  image: InputOutputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  rvec: InputArray,
  tvec: InputArray,
  length: float,
  thickness?: int,
): void;

/**
 * It computes `\\[ \\begin{bmatrix} x\\\\ y\\\\ \\end{bmatrix} = \\begin{bmatrix} a_{11} & a_{12}\\\\
 * a_{21} & a_{22}\\\\ \\end{bmatrix} \\begin{bmatrix} X\\\\ Y\\\\ \\end{bmatrix} + \\begin{bmatrix}
 * b_1\\\\ b_2\\\\ \\end{bmatrix} \\]`
 *
 * Output 2D affine transformation matrix `$2 \\times 3$` or empty matrix if transformation could not
 * be estimated. The returned matrix has the following form: `\\[ \\begin{bmatrix} a_{11} & a_{12} &
 * b_1\\\\ a_{21} & a_{22} & b_2\\\\ \\end{bmatrix} \\]`
 * The function estimates an optimal 2D affine transformation between two 2D point sets using the
 * selected robust algorithm.
 *
 * The computed transformation is then refined further (using only inliers) with the
 * Levenberg-Marquardt method to reduce the re-projection error even more.
 *
 * The RANSAC method can handle practically any ratio of outliers but needs a threshold to distinguish
 * inliers from outliers. The method LMeDS does not need any threshold but it works correctly only when
 * there are more than 50% of inliers.
 *
 * [estimateAffinePartial2D], [getAffineTransform]
 *
 * @param from First input 2D point set containing $(X,Y)$.
 *
 * @param to Second input 2D point set containing $(x,y)$.
 *
 * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
 *
 * @param method Robust method used to compute transformation. The following methods are possible:
 * cv::RANSAC - RANSAC-based robust methodcv::LMEDS - Least-Median robust method RANSAC is the default
 * method.
 *
 * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider a point
 * as an inlier. Applies only to RANSAC.
 *
 * @param maxIters The maximum number of robust method iterations.
 *
 * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
 * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
 * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
 *
 * @param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt). Passing
 * 0 will disable refining, so the output matrix will be output of robust method.
 */
export declare function estimateAffine2D(
  from: InputArray,
  to: InputArray,
  inliers?: OutputArray,
  method?: int,
  ransacReprojThreshold?: double,
  maxIters?: size_t,
  confidence?: double,
  refineIters?: size_t,
): any;

/**
 * It computes `\\[ \\begin{bmatrix} x\\\\ y\\\\ z\\\\ \\end{bmatrix} = \\begin{bmatrix} a_{11} &
 * a_{12} & a_{13}\\\\ a_{21} & a_{22} & a_{23}\\\\ a_{31} & a_{32} & a_{33}\\\\ \\end{bmatrix}
 * \\begin{bmatrix} X\\\\ Y\\\\ Z\\\\ \\end{bmatrix} + \\begin{bmatrix} b_1\\\\ b_2\\\\ b_3\\\\
 * \\end{bmatrix} \\]`
 *
 * The function estimates an optimal 3D affine transformation between two 3D point sets using the
 * RANSAC algorithm.
 *
 * @param src First input 3D point set containing $(X,Y,Z)$.
 *
 * @param dst Second input 3D point set containing $(x,y,z)$.
 *
 * @param out Output 3D affine transformation matrix $3 \times 4$ of the form \[ \begin{bmatrix} a_{11}
 * & a_{12} & a_{13} & b_1\\ a_{21} & a_{22} & a_{23} & b_2\\ a_{31} & a_{32} & a_{33} & b_3\\
 * \end{bmatrix} \]
 *
 * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
 *
 * @param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as an
 * inlier.
 *
 * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
 * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
 * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
 */
export declare function estimateAffine3D(
  src: InputArray,
  dst: InputArray,
  out: OutputArray,
  inliers: OutputArray,
  ransacThreshold?: double,
  confidence?: double,
): int;

/**
 * Output 2D affine transformation (4 degrees of freedom) matrix `$2 \\times 3$` or empty matrix if
 * transformation could not be estimated.
 * The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
 * combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
 * estimation.
 *
 * The computed transformation is then refined further (using only inliers) with the
 * Levenberg-Marquardt method to reduce the re-projection error even more.
 *
 * Estimated transformation matrix is: `\\[ \\begin{bmatrix} \\cos(\\theta) \\cdot s & -\\sin(\\theta)
 * \\cdot s & t_x \\\\ \\sin(\\theta) \\cdot s & \\cos(\\theta) \\cdot s & t_y \\end{bmatrix} \\]`
 * Where `$ \\theta $` is the rotation angle, `$ s $` the scaling factor and `$ t_x, t_y $` are
 * translations in `$ x, y $` axes respectively.
 *
 * The RANSAC method can handle practically any ratio of outliers but need a threshold to distinguish
 * inliers from outliers. The method LMeDS does not need any threshold but it works correctly only when
 * there are more than 50% of inliers.
 *
 * [estimateAffine2D], [getAffineTransform]
 *
 * @param from First input 2D point set.
 *
 * @param to Second input 2D point set.
 *
 * @param inliers Output vector indicating which points are inliers.
 *
 * @param method Robust method used to compute transformation. The following methods are possible:
 * cv::RANSAC - RANSAC-based robust methodcv::LMEDS - Least-Median robust method RANSAC is the default
 * method.
 *
 * @param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider a point
 * as an inlier. Applies only to RANSAC.
 *
 * @param maxIters The maximum number of robust method iterations.
 *
 * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
 * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
 * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
 *
 * @param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt). Passing
 * 0 will disable refining, so the output matrix will be output of robust method.
 */
export declare function estimateAffinePartial2D(
  from: InputArray,
  to: InputArray,
  inliers?: OutputArray,
  method?: int,
  ransacReprojThreshold?: double,
  maxIters?: size_t,
  confidence?: double,
  refineIters?: size_t,
): any;

/**
 * This function is intended to filter the output of the decomposeHomographyMat based on additional
 * information as described in Malis . The summary of the method: the decomposeHomographyMat function
 * returns 2 unique solutions and their "opposites" for a total of 4 solutions. If we have access to
 * the sets of points visible in the camera frame before and after the homography transformation is
 * applied, we can determine which are the true potential solutions and which are the opposites by
 * verifying which homographies are consistent with all visible reference points being in front of the
 * camera. The inputs are left unchanged; the filtered solution set is returned as indices into the
 * existing one.
 *
 * @param rotations Vector of rotation matrices.
 *
 * @param normals Vector of plane normal matrices.
 *
 * @param beforePoints Vector of (rectified) visible reference points before the homography is applied
 *
 * @param afterPoints Vector of (rectified) visible reference points after the homography is applied
 *
 * @param possibleSolutions Vector of int indices representing the viable solution set after filtering
 *
 * @param pointsMask optional Mat/Vector of 8u type representing the mask for the inliers as given by
 * the findHomography function
 */
export declare function filterHomographyDecompByVisibleRefpoints(
  rotations: InputArrayOfArrays,
  normals: InputArrayOfArrays,
  beforePoints: InputArray,
  afterPoints: InputArray,
  possibleSolutions: OutputArray,
  pointsMask?: InputArray,
): void;

/**
 * @param img The input 16-bit signed disparity image
 *
 * @param newVal The disparity value used to paint-off the speckles
 *
 * @param maxSpeckleSize The maximum speckle size to consider it a speckle. Larger blobs are not
 * affected by the algorithm
 *
 * @param maxDiff Maximum difference between neighbor disparity pixels to put them into the same blob.
 * Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point disparity map,
 * where disparity values are multiplied by 16, this scale factor should be taken into account when
 * specifying this parameter value.
 *
 * @param buf The optional temporary buffer to avoid memory allocation within the function.
 */
export declare function filterSpeckles(
  img: InputOutputArray,
  newVal: double,
  maxSpeckleSize: int,
  maxDiff: double,
  buf?: InputOutputArray,
): void;

export declare function find4QuadCornerSubpix(
  img: InputArray,
  corners: InputOutputArray,
  region_size: Size,
): bool;

/**
 * The function attempts to determine whether the input image is a view of the chessboard pattern and
 * locate the internal chessboard corners. The function returns a non-zero value if all of the corners
 * are found and they are placed in a certain order (row by row, left to right in every row).
 * Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,
 * a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black
 * squares touch each other. The detected coordinates are approximate, and to determine their positions
 * more accurately, the function calls cornerSubPix. You also may use the function cornerSubPix with
 * different parameters if returned coordinates are not accurate enough.
 *
 * Sample usage of detecting and drawing chessboard corners: :
 *
 * ```cpp
 * Size patternsize(8,6); //interior number of corners
 * Mat gray = ....; //source image
 * vector<Point2f> corners; //this will be filled by the detected corners
 *
 * //CALIB_CB_FAST_CHECK saves a lot of time on images
 * //that do not contain any chessboard corners
 * bool patternfound = findChessboardCorners(gray, patternsize, corners,
 *         CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
 *         + CALIB_CB_FAST_CHECK);
 *
 * if(patternfound)
 *   cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
 *     TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
 *
 * drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
 * ```
 *
 * The function requires white space (like a square-thick border, the wider the better) around the
 * board to make the detection more robust in various environments. Otherwise, if there is no border
 * and the background is dark, the outer black squares cannot be segmented properly and so the square
 * grouping and ordering algorithm fails.
 *
 * @param image Source chessboard view. It must be an 8-bit grayscale or color image.
 *
 * @param patternSize Number of inner corners per a chessboard row and column ( patternSize =
 * cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
 *
 * @param corners Output array of detected corners.
 *
 * @param flags Various operation flags that can be zero or a combination of the following values:
 * CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black and white, rather
 * than a fixed threshold level (computed from the average image brightness).CALIB_CB_NORMALIZE_IMAGE
 * Normalize the image gamma with equalizeHist before applying fixed or adaptive
 * thresholding.CALIB_CB_FILTER_QUADS Use additional criteria (like contour area, perimeter,
 * square-like shape) to filter out false quads extracted at the contour retrieval
 * stage.CALIB_CB_FAST_CHECK Run a fast check on the image that looks for chessboard corners, and
 * shortcut the call if none is found. This can drastically speed up the call in the degenerate
 * condition when no chessboard is observed.
 */
export declare function findChessboardCorners(
  image: InputArray,
  patternSize: Size,
  corners: OutputArray,
  flags?: int,
): bool;

/**
 * The function is analog to findchessboardCorners but uses a localized radon transformation
 * approximated by box filters being more robust to all sort of noise, faster on larger images and is
 * able to directly return the sub-pixel position of the internal chessboard corners. The Method is
 * based on the paper duda2018 "Accurate Detection and Localization of Checkerboard Corners for
 * Calibration" demonstrating that the returned sub-pixel positions are more accurate than the one
 * returned by cornerSubPix allowing a precise camera calibration for demanding applications.
 *
 * The function requires a white boarder with roughly the same width as one of the checkerboard fields
 * around the whole board to improve the detection in various environments. In addition, because of the
 * localized radon transformation it is beneficial to use round corners for the field corners which are
 * located on the outside of the board. The following figure illustrates a sample checkerboard
 * optimized for the detection. However, any other checkerboard can be used as well.
 *
 * @param image Source chessboard view. It must be an 8-bit grayscale or color image.
 *
 * @param patternSize Number of inner corners per a chessboard row and column ( patternSize =
 * cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
 *
 * @param corners Output array of detected corners.
 *
 * @param flags Various operation flags that can be zero or a combination of the following values:
 * CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before
 * detection.CALIB_CB_EXHAUSTIVE Run an exhaustive search to improve detection rate.CALIB_CB_ACCURACY
 * Up sample input image to improve sub-pixel accuracy due to aliasing effects. This should be used if
 * an accurate camera calibration is required.
 */
export declare function findChessboardCornersSB(
  image: InputArray,
  patternSize: Size,
  corners: OutputArray,
  flags?: int,
): bool;

/**
 * The function attempts to determine whether the input image contains a grid of circles. If it is, the
 * function locates centers of the circles. The function returns a non-zero value if all of the centers
 * have been found and they have been placed in a certain order (row by row, left to right in every
 * row). Otherwise, if the function fails to find all the corners or reorder them, it returns 0.
 *
 * Sample usage of detecting and drawing the centers of circles: :
 *
 * ```cpp
 * Size patternsize(7,7); //number of centers
 * Mat gray = ....; //source image
 * vector<Point2f> centers; //this will be filled by the detected centers
 *
 * bool patternfound = findCirclesGrid(gray, patternsize, centers);
 *
 * drawChessboardCorners(img, patternsize, Mat(centers), patternfound);
 * ```
 *
 * The function requires white space (like a square-thick border, the wider the better) around the
 * board to make the detection more robust in various environments.
 *
 * @param image grid view of input circles; it must be an 8-bit grayscale or color image.
 *
 * @param patternSize number of circles per row and column ( patternSize = Size(points_per_row,
 * points_per_colum) ).
 *
 * @param centers output array of detected centers.
 *
 * @param flags various operation flags that can be one of the following values:
 * CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.CALIB_CB_ASYMMETRIC_GRID uses asymmetric
 * pattern of circles.CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more
 * robust to perspective distortions but much more sensitive to background clutter.
 *
 * @param blobDetector feature detector that finds blobs like dark circles on light background.
 *
 * @param parameters struct for finding circles in a grid pattern.
 */
export declare function findCirclesGrid(
  image: InputArray,
  patternSize: Size,
  centers: OutputArray,
  flags: int,
  blobDetector: any,
  parameters: any,
): bool;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function findCirclesGrid(
  image: InputArray,
  patternSize: Size,
  centers: OutputArray,
  flags?: int,
  blobDetector?: any,
): bool;

/**
 * This function estimates essential matrix based on the five-point algorithm solver in Nister03 .
 * SteweniusCFS is also a related. The epipolar geometry is described by the following equation:
 *
 * `\\[[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\\]`
 *
 * where `$E$` is an essential matrix, `$p_1$` and `$p_2$` are corresponding points in the first and
 * the second images, respectively. The result of this function may be passed further to
 * decomposeEssentialMat or recoverPose to recover the relative pose between cameras.
 *
 * @param points1 Array of N (N >= 5) 2D points from the first image. The point coordinates should be
 * floating-point (single or double precision).
 *
 * @param points2 Array of the second image points of the same size and format as points1 .
 *
 * @param cameraMatrix Camera matrix $K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ . Note
 * that this function assumes that points1 and points2 are feature points from cameras with the same
 * camera matrix.
 *
 * @param method Method for computing an essential matrix.
 * RANSAC for the RANSAC algorithm.LMEDS for the LMedS algorithm.
 *
 * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
 * confidence (probability) that the estimated matrix is correct.
 *
 * @param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
 * line in pixels, beyond which the point is considered an outlier and is not used for computing the
 * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
 * point localization, image resolution, and the image noise.
 *
 * @param mask Output array of N elements, every element of which is set to 0 for outliers and to 1 for
 * the other points. The array is computed only in the RANSAC and LMedS methods.
 */
export declare function findEssentialMat(
  points1: InputArray,
  points2: InputArray,
  cameraMatrix: InputArray,
  method?: int,
  prob?: double,
  threshold?: double,
  mask?: OutputArray,
): Mat;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 * This function differs from the one above that it computes camera matrix from focal length and
 * principal point:
 *
 * `\\[K = \\begin{bmatrix} f & 0 & x_{pp} \\\\ 0 & f & y_{pp} \\\\ 0 & 0 & 1 \\end{bmatrix}\\]`
 *
 * @param points1 Array of N (N >= 5) 2D points from the first image. The point coordinates should be
 * floating-point (single or double precision).
 *
 * @param points2 Array of the second image points of the same size and format as points1 .
 *
 * @param focal focal length of the camera. Note that this function assumes that points1 and points2
 * are feature points from cameras with same focal length and principal point.
 *
 * @param pp principal point of the camera.
 *
 * @param method Method for computing a fundamental matrix.
 * RANSAC for the RANSAC algorithm.LMEDS for the LMedS algorithm.
 *
 * @param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
 * confidence (probability) that the estimated matrix is correct.
 *
 * @param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
 * line in pixels, beyond which the point is considered an outlier and is not used for computing the
 * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
 * point localization, image resolution, and the image noise.
 *
 * @param mask Output array of N elements, every element of which is set to 0 for outliers and to 1 for
 * the other points. The array is computed only in the RANSAC and LMedS methods.
 */
export declare function findEssentialMat(
  points1: InputArray,
  points2: InputArray,
  focal?: double,
  pp?: Point2d,
  method?: int,
  prob?: double,
  threshold?: double,
  mask?: OutputArray,
): Mat;

/**
 * `\\[[p_2; 1]^T F [p_1; 1] = 0\\]`
 *
 * where `$F$` is a fundamental matrix, `$p_1$` and `$p_2$` are corresponding points in the first and
 * the second images, respectively.
 *
 * The function calculates the fundamental matrix using one of four methods listed above and returns
 * the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point
 * algorithm, the function may return up to 3 solutions ( `$9 \\times 3$` matrix that stores all 3
 * matrices sequentially).
 *
 * The calculated fundamental matrix may be passed further to computeCorrespondEpilines that finds the
 * epipolar lines corresponding to the specified points. It can also be passed to
 * stereoRectifyUncalibrated to compute the rectification transformation. :
 *
 * ```cpp
 * // Example. Estimation of fundamental matrix using the RANSAC algorithm
 * int point_count = 100;
 * vector<Point2f> points1(point_count);
 * vector<Point2f> points2(point_count);
 *
 * // initialize the points here ...
 * for( int i = 0; i < point_count; i++ )
 * {
 *     points1[i] = ...;
 *     points2[i] = ...;
 * }
 *
 * Mat fundamental_matrix =
 *  findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
 * ```
 *
 * @param points1 Array of N points from the first image. The point coordinates should be
 * floating-point (single or double precision).
 *
 * @param points2 Array of the second image points of the same size and format as points1 .
 *
 * @param method Method for computing a fundamental matrix.
 * CV_FM_7POINT for a 7-point algorithm. $N = 7$CV_FM_8POINT for an 8-point algorithm. $N \ge
 * 8$CV_FM_RANSAC for the RANSAC algorithm. $N \ge 8$CV_FM_LMEDS for the LMedS algorithm. $N \ge 8$
 *
 * @param ransacReprojThreshold Parameter used only for RANSAC. It is the maximum distance from a point
 * to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for
 * computing the final fundamental matrix. It can be set to something like 1-3, depending on the
 * accuracy of the point localization, image resolution, and the image noise.
 *
 * @param confidence Parameter used for the RANSAC and LMedS methods only. It specifies a desirable
 * level of confidence (probability) that the estimated matrix is correct.
 *
 * @param mask The epipolar geometry is described by the following equation:
 */
export declare function findFundamentalMat(
  points1: InputArray,
  points2: InputArray,
  method?: int,
  ransacReprojThreshold?: double,
  confidence?: double,
  mask?: OutputArray,
): Mat;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function findFundamentalMat(
  points1: InputArray,
  points2: InputArray,
  mask: OutputArray,
  method?: int,
  ransacReprojThreshold?: double,
  confidence?: double,
): Mat;

/**
 * The function finds and returns the perspective transformation `$H$` between the source and the
 * destination planes:
 *
 * `\\[s_i \\vecthree{x'_i}{y'_i}{1} \\sim H \\vecthree{x_i}{y_i}{1}\\]`
 *
 * so that the back-projection error
 *
 * `\\[\\sum _i \\left ( x'_i- \\frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i +
 * h_{33}} \\right )^2+ \\left ( y'_i- \\frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i
 * + h_{33}} \\right )^2\\]`
 *
 * is minimized. If the parameter method is set to the default value 0, the function uses all the point
 * pairs to compute an initial homography estimate with a simple least-squares scheme.
 *
 * However, if not all of the point pairs ( `$srcPoints_i$`, `$dstPoints_i$` ) fit the rigid
 * perspective transformation (that is, there are some outliers), this initial estimate will be poor.
 * In this case, you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try
 * many different random subsets of the corresponding point pairs (of four pairs each, collinear pairs
 * are discarded), estimate the homography matrix using this subset and a simple least-squares
 * algorithm, and then compute the quality/goodness of the computed homography (which is the number of
 * inliers for RANSAC or the least median re-projection error for LMeDS). The best subset is then used
 * to produce the initial estimate of the homography matrix and the mask of inliers/outliers.
 *
 * Regardless of the method, robust or not, the computed homography matrix is refined further (using
 * inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
 * re-projection error even more.
 *
 * The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
 * distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
 * correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
 * noise is rather small, use the default method (method=0).
 *
 * The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
 * determined up to a scale. Thus, it is normalized so that `$h_{33}=1$`. Note that whenever an `$H$`
 * matrix cannot be estimated, an empty one will be returned.
 *
 * [getAffineTransform], [estimateAffine2D], [estimateAffinePartial2D], [getPerspectiveTransform],
 * [warpPerspective], [perspectiveTransform]
 *
 * @param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2 or
 * vector<Point2f> .
 *
 * @param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or a
 * vector<Point2f> .
 *
 * @param method Method used to compute a homography matrix. The following methods are possible:
 * 0 - a regular method using all the points, i.e., the least squares methodRANSAC - RANSAC-based
 * robust methodLMEDS - Least-Median robust methodRHO - PROSAC-based robust method
 *
 * @param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
 * (used in the RANSAC and RHO methods only). That is, if \[\| \texttt{dstPoints} _i -
 * \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2 >
 * \texttt{ransacReprojThreshold}\] then the point $i$ is considered as an outlier. If srcPoints and
 * dstPoints are measured in pixels, it usually makes sense to set this parameter somewhere in the
 * range of 1 to 10.
 *
 * @param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input
 * mask values are ignored.
 *
 * @param maxIters The maximum number of RANSAC iterations.
 *
 * @param confidence Confidence level, between 0 and 1.
 */
export declare function findHomography(
  srcPoints: InputArray,
  dstPoints: InputArray,
  method?: int,
  ransacReprojThreshold?: double,
  mask?: OutputArray,
  maxIters?: any,
  confidence?: any,
): Mat;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function findHomography(
  srcPoints: InputArray,
  dstPoints: InputArray,
  mask: OutputArray,
  method?: int,
  ransacReprojThreshold?: double,
): Mat;

/**
 * The function returns the camera matrix that is either an exact copy of the input cameraMatrix (when
 * centerPrinicipalPoint=false ), or the modified one (when centerPrincipalPoint=true).
 *
 * In the latter case, the new camera matrix will be:
 *
 * `\\[\\begin{bmatrix} f_x && 0 && ( \\texttt{imgSize.width} -1)*0.5 \\\\ 0 && f_y && (
 * \\texttt{imgSize.height} -1)*0.5 \\\\ 0 && 0 && 1 \\end{bmatrix} ,\\]`
 *
 * where `$f_x$` and `$f_y$` are `$(0,0)$` and `$(1,1)$` elements of cameraMatrix, respectively.
 *
 * By default, the undistortion functions in OpenCV (see [initUndistortRectifyMap], [undistort]) do not
 * move the principal point. However, when you work with stereo, it is important to move the principal
 * points in both views to the same y-coordinate (which is required by most of stereo correspondence
 * algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for
 * each view where the principal points are located at the center.
 *
 * @param cameraMatrix Input camera matrix.
 *
 * @param imgsize Camera view image size in pixels.
 *
 * @param centerPrincipalPoint Location of the principal point in the new camera matrix. The parameter
 * indicates whether this location should be at the image center or not.
 */
export declare function getDefaultNewCameraMatrix(
  cameraMatrix: InputArray,
  imgsize?: Size,
  centerPrincipalPoint?: bool,
): Mat;

/**
 * new_camera_matrix Output new camera matrix.
 * The function computes and returns the optimal new camera matrix based on the free scaling parameter.
 * By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original
 * image pixels if there is valuable information in the corners alpha=1 , or get something in between.
 * When alpha>0 , the undistorted result is likely to have some black pixels corresponding to "virtual"
 * pixels outside of the captured distorted image. The original camera matrix, distortion coefficients,
 * the computed new camera matrix, and newImageSize should be passed to initUndistortRectifyMap to
 * produce the maps for remap .
 *
 * @param cameraMatrix Input camera matrix.
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param imageSize Original image size.
 *
 * @param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
 * valid) and 1 (when all the source image pixels are retained in the undistorted image). See
 * stereoRectify for details.
 *
 * @param newImgSize Image size after rectification. By default, it is set to imageSize .
 *
 * @param validPixROI Optional output rectangle that outlines all-good-pixels region in the undistorted
 * image. See roi1, roi2 description in stereoRectify .
 *
 * @param centerPrincipalPoint Optional flag that indicates whether in the new camera matrix the
 * principal point should be at the image center or not. By default, the principal point is chosen to
 * best fit a subset of the source image (determined by alpha) to the corrected image.
 */
export declare function getOptimalNewCameraMatrix(
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  imageSize: Size,
  alpha: double,
  newImgSize?: Size,
  validPixROI?: any,
  centerPrincipalPoint?: bool,
): Mat;

export declare function getValidDisparityROI(
  roi1: Rect,
  roi2: Rect,
  minDisparity: int,
  numberOfDisparities: int,
  SADWindowSize: int,
): Rect;

/**
 * The function estimates and returns an initial camera matrix for the camera calibration process.
 * Currently, the function only supports planar calibration patterns, which are patterns where each
 * object point has z-coordinate =0.
 *
 * @param objectPoints Vector of vectors of the calibration pattern points in the calibration pattern
 * coordinate space. In the old interface all the per-view vectors are concatenated. See
 * calibrateCamera for details.
 *
 * @param imagePoints Vector of vectors of the projections of the calibration pattern points. In the
 * old interface all the per-view vectors are concatenated.
 *
 * @param imageSize Image size in pixels used to initialize the principal point.
 *
 * @param aspectRatio If it is zero or negative, both $f_x$ and $f_y$ are estimated independently.
 * Otherwise, $f_x = f_y * \texttt{aspectRatio}$ .
 */
export declare function initCameraMatrix2D(
  objectPoints: InputArrayOfArrays,
  imagePoints: InputArrayOfArrays,
  imageSize: Size,
  aspectRatio?: double,
): Mat;

/**
 * The function computes the joint undistortion and rectification transformation and represents the
 * result in the form of maps for remap. The undistorted image looks like original, as if it is
 * captured with a camera using the camera matrix =newCameraMatrix and zero distortion. In case of a
 * monocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by
 * [getOptimalNewCameraMatrix] for a better control over scaling. In case of a stereo camera,
 * newCameraMatrix is normally set to P1 or P2 computed by [stereoRectify] .
 *
 * Also, this new camera is oriented differently in the coordinate space, according to R. That, for
 * example, helps to align two heads of a stereo camera so that the epipolar lines on both images
 * become horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera).
 *
 * The function actually builds the maps for the inverse mapping algorithm that is used by remap. That
 * is, for each pixel `$(u, v)$` in the destination (corrected and rectified) image, the function
 * computes the corresponding coordinates in the source image (that is, in the original image from
 * camera). The following process is applied: `\\[ \\begin{array}{l} x \\leftarrow (u - {c'}_x)/{f'}_x
 * \\\\ y \\leftarrow (v - {c'}_y)/{f'}_y \\\\ {[X\\,Y\\,W]} ^T \\leftarrow R^{-1}*[x \\, y \\, 1]^T
 * \\\\ x' \\leftarrow X/W \\\\ y' \\leftarrow Y/W \\\\ r^2 \\leftarrow x'^2 + y'^2 \\\\ x''
 * \\leftarrow x' \\frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2p_1 x' y'
 * + p_2(r^2 + 2 x'^2) + s_1 r^2 + s_2 r^4\\\\ y'' \\leftarrow y' \\frac{1 + k_1 r^2 + k_2 r^4 + k_3
 * r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\\\
 * s\\vecthree{x'''}{y'''}{1} = \\vecthreethree{R_{33}(\\tau_x, \\tau_y)}{0}{-R_{13}((\\tau_x,
 * \\tau_y)} {0}{R_{33}(\\tau_x, \\tau_y)}{-R_{23}(\\tau_x, \\tau_y)} {0}{0}{1} R(\\tau_x, \\tau_y)
 * \\vecthree{x''}{y''}{1}\\\\ map_x(u,v) \\leftarrow x''' f_x + c_x \\\\ map_y(u,v) \\leftarrow y'''
 * f_y + c_y \\end{array} \\]` where `$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[,
 * \\tau_x, \\tau_y]]]])$` are the distortion coefficients.
 *
 * In case of a stereo camera, this function is called twice: once for each camera head, after
 * stereoRectify, which in its turn is called after [stereoCalibrate]. But if the stereo camera was not
 * calibrated, it is still possible to compute the rectification transformations directly from the
 * fundamental matrix using [stereoRectifyUncalibrated]. For each camera, the function computes
 * homography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D
 * space. R can be computed from H as `\\[\\texttt{R} = \\texttt{cameraMatrix} ^{-1} \\cdot \\texttt{H}
 * \\cdot \\texttt{cameraMatrix}\\]` where cameraMatrix can be chosen arbitrarily.
 *
 * @param cameraMatrix Input camera matrix $A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5,
 * k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param R Optional rectification transformation in the object space (3x3 matrix). R1 or R2 , computed
 * by stereoRectify can be passed here. If the matrix is empty, the identity transformation is assumed.
 * In cvInitUndistortMap R assumed to be an identity matrix.
 *
 * @param newCameraMatrix New camera matrix $A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}$.
 *
 * @param size Undistorted image size.
 *
 * @param m1type Type of the first output map that can be CV_32FC1, CV_32FC2 or CV_16SC2, see
 * convertMaps
 *
 * @param map1 The first output map.
 *
 * @param map2 The second output map.
 */
export declare function initUndistortRectifyMap(
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  R: InputArray,
  newCameraMatrix: InputArray,
  size: Size,
  m1type: int,
  map1: OutputArray,
  map2: OutputArray,
): void;

export declare function initWideAngleProjMap(
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  imageSize: Size,
  destImageWidth: int,
  m1type: int,
  map1: OutputArray,
  map2: OutputArray,
  projType?: any,
  alpha?: double,
): float;

export declare function initWideAngleProjMap(
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  imageSize: Size,
  destImageWidth: int,
  m1type: int,
  map1: OutputArray,
  map2: OutputArray,
  projType: int,
  alpha?: double,
): float;

/**
 * The function computes partial derivatives of the elements of the matrix product `$A*B$` with regard
 * to the elements of each of the two input matrices. The function is used to compute the Jacobian
 * matrices in stereoCalibrate but can also be used in any other similar optimization function.
 *
 * @param A First multiplied matrix.
 *
 * @param B Second multiplied matrix.
 *
 * @param dABdA First output derivative matrix d(A*B)/dA of size $\texttt{A.rows*B.cols} \times
 * {A.rows*A.cols}$ .
 *
 * @param dABdB Second output derivative matrix d(A*B)/dB of size $\texttt{A.rows*B.cols} \times
 * {B.rows*B.cols}$ .
 */
export declare function matMulDeriv(
  A: InputArray,
  B: InputArray,
  dABdA: OutputArray,
  dABdB: OutputArray,
): void;

/**
 * The function computes projections of 3D points to the image plane given intrinsic and extrinsic
 * camera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of
 * image points coordinates (as functions of all the input parameters) with respect to the particular
 * parameters, intrinsic and/or extrinsic. The Jacobians are used during the global optimization in
 * calibrateCamera, solvePnP, and stereoCalibrate . The function itself can also be used to compute a
 * re-projection error given the current intrinsic and extrinsic parameters.
 *
 * By setting rvec=tvec=(0,0,0) or by setting cameraMatrix to a 3x3 identity matrix, or by passing zero
 * distortion coefficients, you can get various useful partial cases of the function. This means that
 * you can compute the distorted coordinates for a sparse set of points or apply a perspective
 * transformation (and also compute the derivatives) in the ideal zero-distortion setup.
 *
 * @param objectPoints Array of object points, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel (or
 * vector<Point3f> ), where N is the number of points in the view.
 *
 * @param rvec Rotation vector. See Rodrigues for details.
 *
 * @param tvec Translation vector.
 *
 * @param cameraMatrix Camera matrix $A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is empty,
 * the zero distortion coefficients are assumed.
 *
 * @param imagePoints Output array of image points, 1xN/Nx1 2-channel, or vector<Point2f> .
 *
 * @param jacobian Optional output 2Nx(10+<numDistCoeffs>) jacobian matrix of derivatives of image
 * points with respect to components of the rotation vector, translation vector, focal lengths,
 * coordinates of the principal point and the distortion coefficients. In the old interface different
 * components of the jacobian are returned via different output parameters.
 *
 * @param aspectRatio Optional "fixed aspect ratio" parameter. If the parameter is not 0, the function
 * assumes that the aspect ratio (fx/fy) is fixed and correspondingly adjusts the jacobian matrix.
 */
export declare function projectPoints(
  objectPoints: InputArray,
  rvec: InputArray,
  tvec: InputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  imagePoints: OutputArray,
  jacobian?: OutputArray,
  aspectRatio?: double,
): void;

/**
 * This function can be used to process output E and mask from findEssentialMat. In this scenario,
 * points1 and points2 are the same input for findEssentialMat. :
 *
 * ```cpp
 * // Example. Estimation of fundamental matrix using the RANSAC algorithm
 * int point_count = 100;
 * vector<Point2f> points1(point_count);
 * vector<Point2f> points2(point_count);
 *
 * // initialize the points here ...
 * for( int i = 0; i < point_count; i++ )
 * {
 *     points1[i] = ...;
 *     points2[i] = ...;
 * }
 *
 * // cametra matrix with both focal lengths = 1, and principal point = (0, 0)
 * Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
 *
 * Mat E, R, t, mask;
 *
 * E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
 * recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
 * ```
 *
 * @param E The input essential matrix.
 *
 * @param points1 Array of N 2D points from the first image. The point coordinates should be
 * floating-point (single or double precision).
 *
 * @param points2 Array of the second image points of the same size and format as points1 .
 *
 * @param cameraMatrix Camera matrix $K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ . Note
 * that this function assumes that points1 and points2 are feature points from cameras with the same
 * camera matrix.
 *
 * @param R Recovered relative rotation.
 *
 * @param t Recovered relative translation.
 *
 * @param mask Input/output mask for inliers in points1 and points2. : If it is not empty, then it
 * marks inliers in points1 and points2 for then given essential matrix E. Only these inliers will be
 * used to recover pose. In the output mask only inliers which pass the cheirality check. This function
 * decomposes an essential matrix using decomposeEssentialMat and then verifies possible pose
 * hypotheses by doing cheirality check. The cheirality check basically means that the triangulated 3D
 * points should have positive depth. Some details can be found in Nister03 .
 */
export declare function recoverPose(
  E: InputArray,
  points1: InputArray,
  points2: InputArray,
  cameraMatrix: InputArray,
  R: OutputArray,
  t: OutputArray,
  mask?: InputOutputArray,
): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 * This function differs from the one above that it computes camera matrix from focal length and
 * principal point:
 *
 * `\\[K = \\begin{bmatrix} f & 0 & x_{pp} \\\\ 0 & f & y_{pp} \\\\ 0 & 0 & 1 \\end{bmatrix}\\]`
 *
 * @param E The input essential matrix.
 *
 * @param points1 Array of N 2D points from the first image. The point coordinates should be
 * floating-point (single or double precision).
 *
 * @param points2 Array of the second image points of the same size and format as points1 .
 *
 * @param R Recovered relative rotation.
 *
 * @param t Recovered relative translation.
 *
 * @param focal Focal length of the camera. Note that this function assumes that points1 and points2
 * are feature points from cameras with same focal length and principal point.
 *
 * @param pp principal point of the camera.
 *
 * @param mask Input/output mask for inliers in points1 and points2. : If it is not empty, then it
 * marks inliers in points1 and points2 for then given essential matrix E. Only these inliers will be
 * used to recover pose. In the output mask only inliers which pass the cheirality check.
 */
export declare function recoverPose(
  E: InputArray,
  points1: InputArray,
  points2: InputArray,
  R: OutputArray,
  t: OutputArray,
  focal?: double,
  pp?: Point2d,
  mask?: InputOutputArray,
): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param E The input essential matrix.
 *
 * @param points1 Array of N 2D points from the first image. The point coordinates should be
 * floating-point (single or double precision).
 *
 * @param points2 Array of the second image points of the same size and format as points1.
 *
 * @param cameraMatrix Camera matrix $K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ . Note
 * that this function assumes that points1 and points2 are feature points from cameras with the same
 * camera matrix.
 *
 * @param R Recovered relative rotation.
 *
 * @param t Recovered relative translation.
 *
 * @param distanceThresh threshold distance which is used to filter out far away points (i.e. infinite
 * points).
 *
 * @param mask Input/output mask for inliers in points1 and points2. : If it is not empty, then it
 * marks inliers in points1 and points2 for then given essential matrix E. Only these inliers will be
 * used to recover pose. In the output mask only inliers which pass the cheirality check.
 *
 * @param triangulatedPoints 3d points which were reconstructed by triangulation.
 */
export declare function recoverPose(
  E: InputArray,
  points1: InputArray,
  points2: InputArray,
  cameraMatrix: InputArray,
  R: OutputArray,
  t: OutputArray,
  distanceThresh: double,
  mask?: InputOutputArray,
  triangulatedPoints?: OutputArray,
): int;

export declare function rectify3Collinear(
  cameraMatrix1: InputArray,
  distCoeffs1: InputArray,
  cameraMatrix2: InputArray,
  distCoeffs2: InputArray,
  cameraMatrix3: InputArray,
  distCoeffs3: InputArray,
  imgpt1: InputArrayOfArrays,
  imgpt3: InputArrayOfArrays,
  imageSize: Size,
  R12: InputArray,
  T12: InputArray,
  R13: InputArray,
  T13: InputArray,
  R1: OutputArray,
  R2: OutputArray,
  R3: OutputArray,
  P1: OutputArray,
  P2: OutputArray,
  P3: OutputArray,
  Q: OutputArray,
  alpha: double,
  newImgSize: Size,
  roi1: any,
  roi2: any,
  flags: int,
): float;

/**
 * The function transforms a single-channel disparity map to a 3-channel image representing a 3D
 * surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it
 * computes:
 *
 * `\\[\\begin{array}{l} [X \\; Y \\; Z \\; W]^T = \\texttt{Q} *[x \\; y \\; \\texttt{disparity} (x,y)
 * \\; 1]^T \\\\ \\texttt{\\_3dImage} (x,y) = (X/W, \\; Y/W, \\; Z/W) \\end{array}\\]`
 *
 * The matrix Q can be an arbitrary `$4 \\times 4$` matrix (for example, the one computed by
 * stereoRectify). To reproject a sparse set of points {(x,y,d),...} to 3D space, use
 * perspectiveTransform .
 *
 * @param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
 * floating-point disparity image. If 16-bit signed format is used, the values are assumed to have no
 * fractional bits.
 *
 * @param _3dImage Output 3-channel floating-point image of the same size as disparity . Each element
 * of _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map.
 *
 * @param Q $4 \times 4$ perspective transformation matrix that can be obtained with stereoRectify.
 *
 * @param handleMissingValues Indicates, whether the function should handle missing values (i.e. points
 * where the disparity was not computed). If handleMissingValues=true, then pixels with the minimal
 * disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed to 3D
 * points with a very large Z value (currently set to 10000).
 *
 * @param ddepth The optional output array depth. If it is -1, the output image will have CV_32F depth.
 * ddepth can also be set to CV_16S, CV_32S or CV_32F.
 */
export declare function reprojectImageTo3D(
  disparity: InputArray,
  _3dImage: OutputArray,
  Q: InputArray,
  handleMissingValues?: bool,
  ddepth?: int,
): void;

/**
 * `\\[\\begin{array}{l} \\theta \\leftarrow norm(r) \\\\ r \\leftarrow r/ \\theta \\\\ R =
 * \\cos{\\theta} I + (1- \\cos{\\theta} ) r r^T + \\sin{\\theta}
 * \\vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \\end{array}\\]`
 *
 * Inverse transformation can be also done easily, since
 *
 * `\\[\\sin ( \\theta ) \\vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \\frac{R -
 * R^T}{2}\\]`
 *
 * A rotation vector is a convenient and most compact representation of a rotation matrix (since any
 * rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry
 * optimization procedures like calibrateCamera, stereoCalibrate, or solvePnP .
 *
 * @param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
 *
 * @param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
 *
 * @param jacobian Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial
 * derivatives of the output array components with respect to the input array components.
 */
export declare function Rodrigues(
  src: InputArray,
  dst: OutputArray,
  jacobian?: OutputArray,
): void;

/**
 * The function computes a RQ decomposition using the given rotations. This function is used in
 * decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
 * and a rotation matrix.
 *
 * It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
 * degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
 * sequence of rotations about the three principal axes that results in the same orientation of an
 * object, e.g. see Slabaugh . Returned tree rotation matrices and corresponding three Euler angles are
 * only one of the possible solutions.
 *
 * @param src 3x3 input matrix.
 *
 * @param mtxR Output 3x3 upper-triangular matrix.
 *
 * @param mtxQ Output 3x3 orthogonal matrix.
 *
 * @param Qx Optional output 3x3 rotation matrix around x-axis.
 *
 * @param Qy Optional output 3x3 rotation matrix around y-axis.
 *
 * @param Qz Optional output 3x3 rotation matrix around z-axis.
 */
export declare function RQDecomp3x3(
  src: InputArray,
  mtxR: OutputArray,
  mtxQ: OutputArray,
  Qx?: OutputArray,
  Qy?: OutputArray,
  Qz?: OutputArray,
): Vec3d;

/**
 * The function [cv::sampsonDistance] calculates and returns the first order approximation of the
 * geometric error as: `\\[ sd( \\texttt{pt1} , \\texttt{pt2} )= \\frac{(\\texttt{pt2}^t \\cdot
 * \\texttt{F} \\cdot \\texttt{pt1})^2} {((\\texttt{F} \\cdot \\texttt{pt1})(0))^2 + ((\\texttt{F}
 * \\cdot \\texttt{pt1})(1))^2 + ((\\texttt{F}^t \\cdot \\texttt{pt2})(0))^2 + ((\\texttt{F}^t \\cdot
 * \\texttt{pt2})(1))^2} \\]` The fundamental matrix may be calculated using the
 * [cv::findFundamentalMat] function. See HartleyZ00 11.4.3 for details.
 *
 * The computed Sampson distance.
 *
 * @param pt1 first homogeneous 2d point
 *
 * @param pt2 second homogeneous 2d point
 *
 * @param F fundamental matrix
 */
export declare function sampsonDistance(
  pt1: InputArray,
  pt2: InputArray,
  F: InputArray,
): double;

/**
 * The function estimates the object pose given 3 object points, their corresponding image projections,
 * as well as the camera matrix and the distortion coefficients.
 *
 * The solutions are sorted by reprojection errors (lowest to highest).
 *
 * @param objectPoints Array of object points in the object coordinate space, 3x3 1-channel or 1x3/3x1
 * 3-channel. vector<Point3f> can be also passed here.
 *
 * @param imagePoints Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.
 * vector<Point2f> can be also passed here.
 *
 * @param cameraMatrix Input camera matrix $A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param rvecs Output rotation vectors (see Rodrigues ) that, together with tvecs, brings points from
 * the model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions.
 *
 * @param tvecs Output translation vectors.
 *
 * @param flags Method for solving a P3P problem:
 * SOLVEPNP_P3P Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang "Complete
 * Solution Classification for the Perspective-Three-Point Problem" (gao2003complete).SOLVEPNP_AP3P
 * Method is based on the paper of T. Ke and S. Roumeliotis. "An Efficient Algebraic Solution to the
 * Perspective-Three-Point Problem" (Ke17).
 */
export declare function solveP3P(
  objectPoints: InputArray,
  imagePoints: InputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  rvecs: OutputArrayOfArrays,
  tvecs: OutputArrayOfArrays,
  flags: int,
): int;

/**
 * P3P methods ([SOLVEPNP_P3P], [SOLVEPNP_AP3P]): need 4 input points to return a unique solution.
 * [SOLVEPNP_IPPE] Input points must be >= 4 and object points must be coplanar.
 * [SOLVEPNP_IPPE_SQUARE] Special case suitable for marker pose estimation. Number of input points must
 * be 4. Object points must be defined in the following order:
 *
 * point 0: [-squareLength / 2, squareLength / 2, 0]
 * point 1: [ squareLength / 2, squareLength / 2, 0]
 * point 2: [ squareLength / 2, -squareLength / 2, 0]
 * point 3: [-squareLength / 2, -squareLength / 2, 0]
 *
 * for all the other flags, number of input points must be >= 4 and object points can be in any
 * configuration.
 *
 * The function estimates the object pose given a set of object points, their corresponding image
 * projections, as well as the camera matrix and the distortion coefficients, see the figure below
 * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward and
 * the Z-axis forward).
 *
 * Points expressed in the world frame `$ \\bf{X}_w $` are projected into the image plane `$ \\left[ u,
 * v \\right] $` using the perspective projection model `$ \\Pi $` and the camera intrinsic parameters
 * matrix `$ \\bf{A} $`:
 *
 * `\\[ \\begin{align*} \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} &= \\bf{A} \\hspace{0.1em} \\Pi
 * \\hspace{0.2em} ^{c}\\bf{M}_w \\begin{bmatrix} X_{w} \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix}
 * \\\\ \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} &= \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y
 * & c_y \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 1
 * & 0 \\end{bmatrix} \\begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\\\ r_{21} & r_{22} & r_{23} &
 * t_y \\\\ r_{31} & r_{32} & r_{33} & t_z \\\\ 0 & 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} X_{w}
 * \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix} \\end{align*} \\]`
 *
 * The estimated pose is thus the rotation (`rvec`) and the translation (`tvec`) vectors that allow
 * transforming a 3D point expressed in the world frame into the camera frame:
 *
 * `\\[ \\begin{align*} \\begin{bmatrix} X_c \\\\ Y_c \\\\ Z_c \\\\ 1 \\end{bmatrix} &= \\hspace{0.2em}
 * ^{c}\\bf{M}_w \\begin{bmatrix} X_{w} \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix} \\\\
 * \\begin{bmatrix} X_c \\\\ Y_c \\\\ Z_c \\\\ 1 \\end{bmatrix} &= \\begin{bmatrix} r_{11} & r_{12} &
 * r_{13} & t_x \\\\ r_{21} & r_{22} & r_{23} & t_y \\\\ r_{31} & r_{32} & r_{33} & t_z \\\\ 0 & 0 & 0
 * & 1 \\end{bmatrix} \\begin{bmatrix} X_{w} \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix} \\end{align*}
 * \\]`
 *
 * An example of how to use solvePnP for planar augmented reality can be found at
 * opencv_source_code/samples/python/plane_ar.py
 * If you are using Python:
 *
 * Numpy array slices won't work as input because solvePnP requires contiguous arrays (enforced by the
 * assertion using [cv::Mat::checkVector()] around line 55 of modules/calib3d/src/solvepnp.cpp version
 * 2.4.9)
 * The P3P algorithm requires image points to be in an array of shape (N,1,2) due to its calling of
 * [cv::undistortPoints] (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9) which
 * requires 2-channel information.
 * Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of it as,
 * e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
 * np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
 *
 * The methods **SOLVEPNP_DLS** and **SOLVEPNP_UPNP** cannot be used as the current implementations are
 * unstable and sometimes give completely wrong results. If you pass one of these two flags,
 * **SOLVEPNP_EPNP** method will be used instead.
 * The minimum number of points is 4 in the general case. In the case of **SOLVEPNP_P3P** and
 * **SOLVEPNP_AP3P** methods, it is required to use exactly 4 points (the first 3 points are used to
 * estimate all the solutions of the P3P problem, the last one is used to retain the best solution that
 * minimizes the reprojection error).
 * With **SOLVEPNP_ITERATIVE** method and `useExtrinsicGuess=true`, the minimum number of points is 3
 * (3 points are sufficient to compute a pose but there are up to 4 solutions). The initial solution
 * should be close to the global solution to converge.
 * With **SOLVEPNP_IPPE** input points must be >= 4 and object points must be coplanar.
 * With **SOLVEPNP_IPPE_SQUARE** this is a special case suitable for marker pose estimation. Number of
 * input points must be 4. Object points must be defined in the following order:
 *
 * point 0: [-squareLength / 2, squareLength / 2, 0]
 * point 1: [ squareLength / 2, squareLength / 2, 0]
 * point 2: [ squareLength / 2, -squareLength / 2, 0]
 * point 3: [-squareLength / 2, -squareLength / 2, 0]
 *
 * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1
 * 3-channel, where N is the number of points. vector<Point3f> can be also passed here.
 *
 * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel, where N
 * is the number of points. vector<Point2f> can be also passed here.
 *
 * @param cameraMatrix Input camera matrix $A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param rvec Output rotation vector (see Rodrigues ) that, together with tvec, brings points from the
 * model coordinate system to the camera coordinate system.
 *
 * @param tvec Output translation vector.
 *
 * @param useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses the
 * provided rvec and tvec values as initial approximations of the rotation and translation vectors,
 * respectively, and further optimizes them.
 *
 * @param flags Method for solving a PnP problem:
 * SOLVEPNP_ITERATIVE Iterative method is based on a Levenberg-Marquardt optimization. In this case the
 * function finds such a pose that minimizes reprojection error, that is the sum of squared distances
 * between the observed projections imagePoints and the projected (using projectPoints ) objectPoints
 * .SOLVEPNP_P3P Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang "Complete
 * Solution Classification for the Perspective-Three-Point Problem" (gao2003complete). In this case the
 * function requires exactly four object and image points.SOLVEPNP_AP3P Method is based on the paper of
 * T. Ke, S. Roumeliotis "An Efficient Algebraic Solution to the Perspective-Three-Point Problem"
 * (Ke17). In this case the function requires exactly four object and image points.SOLVEPNP_EPNP Method
 * has been introduced by F. Moreno-Noguer, V. Lepetit and P. Fua in the paper "EPnP: Efficient
 * Perspective-n-Point Camera Pose Estimation" (lepetit2009epnp).SOLVEPNP_DLS Method is based on the
 * paper of J. Hesch and S. Roumeliotis. "A Direct Least-Squares (DLS) Method for PnP"
 * (hesch2011direct).SOLVEPNP_UPNP Method is based on the paper of A. Penate-Sanchez, J. Andrade-Cetto,
 * F. Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
 * Estimation" (penate2013exhaustive). In this case the function also estimates the parameters $f_x$
 * and $f_y$ assuming that both have the same value. Then the cameraMatrix is updated with the
 * estimated focal length.SOLVEPNP_IPPE Method is based on the paper of T. Collins and A. Bartoli.
 * "Infinitesimal Plane-Based Pose Estimation" (Collins14). This method requires coplanar object
 * points.SOLVEPNP_IPPE_SQUARE Method is based on the paper of Toby Collins and Adrien Bartoli.
 * "Infinitesimal Plane-Based Pose Estimation" (Collins14). This method is suitable for marker pose
 * estimation. It requires 4 coplanar object points defined in the following order:
 * point 0: [-squareLength / 2, squareLength / 2, 0]point 1: [ squareLength / 2, squareLength / 2,
 * 0]point 2: [ squareLength / 2, -squareLength / 2, 0]point 3: [-squareLength / 2, -squareLength / 2,
 * 0]
 */
export declare function solvePnP(
  objectPoints: InputArray,
  imagePoints: InputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  rvec: OutputArray,
  tvec: OutputArray,
  useExtrinsicGuess?: bool,
  flags?: int,
): bool;

/**
 * P3P methods ([SOLVEPNP_P3P], [SOLVEPNP_AP3P]): 3 or 4 input points. Number of returned solutions can
 * be between 0 and 4 with 3 input points.
 * [SOLVEPNP_IPPE] Input points must be >= 4 and object points must be coplanar. Returns 2 solutions.
 * [SOLVEPNP_IPPE_SQUARE] Special case suitable for marker pose estimation. Number of input points must
 * be 4 and 2 solutions are returned. Object points must be defined in the following order:
 *
 * point 0: [-squareLength / 2, squareLength / 2, 0]
 * point 1: [ squareLength / 2, squareLength / 2, 0]
 * point 2: [ squareLength / 2, -squareLength / 2, 0]
 * point 3: [-squareLength / 2, -squareLength / 2, 0]
 *
 * for all the other flags, number of input points must be >= 4 and object points can be in any
 * configuration. Only 1 solution is returned.
 *
 * The function estimates the object pose given a set of object points, their corresponding image
 * projections, as well as the camera matrix and the distortion coefficients, see the figure below
 * (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward and
 * the Z-axis forward).
 *
 * Points expressed in the world frame `$ \\bf{X}_w $` are projected into the image plane `$ \\left[ u,
 * v \\right] $` using the perspective projection model `$ \\Pi $` and the camera intrinsic parameters
 * matrix `$ \\bf{A} $`:
 *
 * `\\[ \\begin{align*} \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} &= \\bf{A} \\hspace{0.1em} \\Pi
 * \\hspace{0.2em} ^{c}\\bf{M}_w \\begin{bmatrix} X_{w} \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix}
 * \\\\ \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} &= \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y
 * & c_y \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 1
 * & 0 \\end{bmatrix} \\begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\\\ r_{21} & r_{22} & r_{23} &
 * t_y \\\\ r_{31} & r_{32} & r_{33} & t_z \\\\ 0 & 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} X_{w}
 * \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix} \\end{align*} \\]`
 *
 * The estimated pose is thus the rotation (`rvec`) and the translation (`tvec`) vectors that allow
 * transforming a 3D point expressed in the world frame into the camera frame:
 *
 * `\\[ \\begin{align*} \\begin{bmatrix} X_c \\\\ Y_c \\\\ Z_c \\\\ 1 \\end{bmatrix} &= \\hspace{0.2em}
 * ^{c}\\bf{M}_w \\begin{bmatrix} X_{w} \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix} \\\\
 * \\begin{bmatrix} X_c \\\\ Y_c \\\\ Z_c \\\\ 1 \\end{bmatrix} &= \\begin{bmatrix} r_{11} & r_{12} &
 * r_{13} & t_x \\\\ r_{21} & r_{22} & r_{23} & t_y \\\\ r_{31} & r_{32} & r_{33} & t_z \\\\ 0 & 0 & 0
 * & 1 \\end{bmatrix} \\begin{bmatrix} X_{w} \\\\ Y_{w} \\\\ Z_{w} \\\\ 1 \\end{bmatrix} \\end{align*}
 * \\]`
 *
 * An example of how to use solvePnP for planar augmented reality can be found at
 * opencv_source_code/samples/python/plane_ar.py
 * If you are using Python:
 *
 * Numpy array slices won't work as input because solvePnP requires contiguous arrays (enforced by the
 * assertion using [cv::Mat::checkVector()] around line 55 of modules/calib3d/src/solvepnp.cpp version
 * 2.4.9)
 * The P3P algorithm requires image points to be in an array of shape (N,1,2) due to its calling of
 * [cv::undistortPoints] (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9) which
 * requires 2-channel information.
 * Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of it as,
 * e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
 * np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
 *
 * The methods **SOLVEPNP_DLS** and **SOLVEPNP_UPNP** cannot be used as the current implementations are
 * unstable and sometimes give completely wrong results. If you pass one of these two flags,
 * **SOLVEPNP_EPNP** method will be used instead.
 * The minimum number of points is 4 in the general case. In the case of **SOLVEPNP_P3P** and
 * **SOLVEPNP_AP3P** methods, it is required to use exactly 4 points (the first 3 points are used to
 * estimate all the solutions of the P3P problem, the last one is used to retain the best solution that
 * minimizes the reprojection error).
 * With **SOLVEPNP_ITERATIVE** method and `useExtrinsicGuess=true`, the minimum number of points is 3
 * (3 points are sufficient to compute a pose but there are up to 4 solutions). The initial solution
 * should be close to the global solution to converge.
 * With **SOLVEPNP_IPPE** input points must be >= 4 and object points must be coplanar.
 * With **SOLVEPNP_IPPE_SQUARE** this is a special case suitable for marker pose estimation. Number of
 * input points must be 4. Object points must be defined in the following order:
 *
 * point 0: [-squareLength / 2, squareLength / 2, 0]
 * point 1: [ squareLength / 2, squareLength / 2, 0]
 * point 2: [ squareLength / 2, -squareLength / 2, 0]
 * point 3: [-squareLength / 2, -squareLength / 2, 0]
 *
 * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1
 * 3-channel, where N is the number of points. vector<Point3f> can be also passed here.
 *
 * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel, where N
 * is the number of points. vector<Point2f> can be also passed here.
 *
 * @param cameraMatrix Input camera matrix $A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param rvecs Vector of output rotation vectors (see Rodrigues ) that, together with tvecs, brings
 * points from the model coordinate system to the camera coordinate system.
 *
 * @param tvecs Vector of output translation vectors.
 *
 * @param useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses the
 * provided rvec and tvec values as initial approximations of the rotation and translation vectors,
 * respectively, and further optimizes them.
 *
 * @param flags Method for solving a PnP problem:
 * SOLVEPNP_ITERATIVE Iterative method is based on a Levenberg-Marquardt optimization. In this case the
 * function finds such a pose that minimizes reprojection error, that is the sum of squared distances
 * between the observed projections imagePoints and the projected (using projectPoints ) objectPoints
 * .SOLVEPNP_P3P Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang "Complete
 * Solution Classification for the Perspective-Three-Point Problem" (gao2003complete). In this case the
 * function requires exactly four object and image points.SOLVEPNP_AP3P Method is based on the paper of
 * T. Ke, S. Roumeliotis "An Efficient Algebraic Solution to the Perspective-Three-Point Problem"
 * (Ke17). In this case the function requires exactly four object and image points.SOLVEPNP_EPNP Method
 * has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the paper "EPnP: Efficient
 * Perspective-n-Point Camera Pose Estimation" (lepetit2009epnp).SOLVEPNP_DLS Method is based on the
 * paper of Joel A. Hesch and Stergios I. Roumeliotis. "A Direct Least-Squares (DLS) Method for PnP"
 * (hesch2011direct).SOLVEPNP_UPNP Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
 * F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
 * Estimation" (penate2013exhaustive). In this case the function also estimates the parameters $f_x$
 * and $f_y$ assuming that both have the same value. Then the cameraMatrix is updated with the
 * estimated focal length.SOLVEPNP_IPPE Method is based on the paper of T. Collins and A. Bartoli.
 * "Infinitesimal Plane-Based Pose Estimation" (Collins14). This method requires coplanar object
 * points.SOLVEPNP_IPPE_SQUARE Method is based on the paper of Toby Collins and Adrien Bartoli.
 * "Infinitesimal Plane-Based Pose Estimation" (Collins14). This method is suitable for marker pose
 * estimation. It requires 4 coplanar object points defined in the following order:
 * point 0: [-squareLength / 2, squareLength / 2, 0]point 1: [ squareLength / 2, squareLength / 2,
 * 0]point 2: [ squareLength / 2, -squareLength / 2, 0]point 3: [-squareLength / 2, -squareLength / 2,
 * 0]
 *
 * @param rvec Rotation vector used to initialize an iterative PnP refinement algorithm, when flag is
 * SOLVEPNP_ITERATIVE and useExtrinsicGuess is set to true.
 *
 * @param tvec Translation vector used to initialize an iterative PnP refinement algorithm, when flag
 * is SOLVEPNP_ITERATIVE and useExtrinsicGuess is set to true.
 *
 * @param reprojectionError Optional vector of reprojection error, that is the RMS error ( $
 * \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} $) between the input
 * image points and the 3D object points projected with the estimated pose.
 */
export declare function solvePnPGeneric(
  objectPoints: InputArray,
  imagePoints: InputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  rvecs: OutputArrayOfArrays,
  tvecs: OutputArrayOfArrays,
  useExtrinsicGuess?: bool,
  flags?: SolvePnPMethod,
  rvec?: InputArray,
  tvec?: InputArray,
  reprojectionError?: OutputArray,
): int;

/**
 * The function estimates an object pose given a set of object points, their corresponding image
 * projections, as well as the camera matrix and the distortion coefficients. This function finds such
 * a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
 * projections imagePoints and the projected (using [projectPoints] ) objectPoints. The use of RANSAC
 * makes the function resistant to outliers.
 *
 * An example of how to use solvePNPRansac for object detection can be found at
 * opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
 * The default method used to estimate the camera pose for the Minimal Sample Sets step is
 * [SOLVEPNP_EPNP]. Exceptions are:
 *
 * if you choose [SOLVEPNP_P3P] or [SOLVEPNP_AP3P], these methods will be used.
 * if the number of input points is equal to 4, [SOLVEPNP_P3P] is used.
 *
 * The method used to estimate the camera pose using all the inliers is defined by the flags parameters
 * unless it is equal to [SOLVEPNP_P3P] or [SOLVEPNP_AP3P]. In this case, the method [SOLVEPNP_EPNP]
 * will be used instead.
 *
 * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1
 * 3-channel, where N is the number of points. vector<Point3f> can be also passed here.
 *
 * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel, where N
 * is the number of points. vector<Point2f> can be also passed here.
 *
 * @param cameraMatrix Input camera matrix $A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param rvec Output rotation vector (see Rodrigues ) that, together with tvec, brings points from the
 * model coordinate system to the camera coordinate system.
 *
 * @param tvec Output translation vector.
 *
 * @param useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses the
 * provided rvec and tvec values as initial approximations of the rotation and translation vectors,
 * respectively, and further optimizes them.
 *
 * @param iterationsCount Number of iterations.
 *
 * @param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value is
 * the maximum allowed distance between the observed and computed point projections to consider it an
 * inlier.
 *
 * @param confidence The probability that the algorithm produces a useful result.
 *
 * @param inliers Output vector that contains indices of inliers in objectPoints and imagePoints .
 *
 * @param flags Method for solving a PnP problem (see solvePnP ).
 */
export declare function solvePnPRansac(
  objectPoints: InputArray,
  imagePoints: InputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  rvec: OutputArray,
  tvec: OutputArray,
  useExtrinsicGuess?: bool,
  iterationsCount?: int,
  reprojectionError?: float,
  confidence?: double,
  inliers?: OutputArray,
  flags?: int,
): bool;

/**
 * The function refines the object pose given at least 3 object points, their corresponding image
 * projections, an initial solution for the rotation and translation vector, as well as the camera
 * matrix and the distortion coefficients. The function minimizes the projection error with respect to
 * the rotation and the translation vectors, according to a Levenberg-Marquardt iterative minimization
 * Madsen04 Eade13 process.
 *
 * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1
 * 3-channel, where N is the number of points. vector<Point3f> can also be passed here.
 *
 * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel, where N
 * is the number of points. vector<Point2f> can also be passed here.
 *
 * @param cameraMatrix Input camera matrix $A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param rvec Input/Output rotation vector (see Rodrigues ) that, together with tvec, brings points
 * from the model coordinate system to the camera coordinate system. Input values are used as an
 * initial solution.
 *
 * @param tvec Input/Output translation vector. Input values are used as an initial solution.
 *
 * @param criteria Criteria when to stop the Levenberg-Marquard iterative algorithm.
 */
export declare function solvePnPRefineLM(
  objectPoints: InputArray,
  imagePoints: InputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  rvec: InputOutputArray,
  tvec: InputOutputArray,
  criteria?: TermCriteria,
): void;

/**
 * The function refines the object pose given at least 3 object points, their corresponding image
 * projections, an initial solution for the rotation and translation vector, as well as the camera
 * matrix and the distortion coefficients. The function minimizes the projection error with respect to
 * the rotation and the translation vectors, using a virtual visual servoing (VVS) Chaumette06
 * Marchand16 scheme.
 *
 * @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1
 * 3-channel, where N is the number of points. vector<Point3f> can also be passed here.
 *
 * @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel, where N
 * is the number of points. vector<Point2f> can also be passed here.
 *
 * @param cameraMatrix Input camera matrix $A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6
 * [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param rvec Input/Output rotation vector (see Rodrigues ) that, together with tvec, brings points
 * from the model coordinate system to the camera coordinate system. Input values are used as an
 * initial solution.
 *
 * @param tvec Input/Output translation vector. Input values are used as an initial solution.
 *
 * @param criteria Criteria when to stop the Levenberg-Marquard iterative algorithm.
 *
 * @param VVSlambda Gain for the virtual visual servoing control law, equivalent to the $\alpha$ gain
 * in the Damped Gauss-Newton formulation.
 */
export declare function solvePnPRefineVVS(
  objectPoints: InputArray,
  imagePoints: InputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  rvec: InputOutputArray,
  tvec: InputOutputArray,
  criteria?: TermCriteria,
  VVSlambda?: double,
): void;

/**
 * The function estimates transformation between two cameras making a stereo pair. If you have a stereo
 * camera where the relative position and orientation of two cameras is fixed, and if you computed
 * poses of an object relative to the first camera and to the second camera, (R1, T1) and (R2, T2),
 * respectively (this can be done with solvePnP ), then those poses definitely relate to each other.
 * This means that, given ( `$R_1$`, `$T_1$` ), it should be possible to compute ( `$R_2$`, `$T_2$` ).
 * You only need to know the position and orientation of the second camera relative to the first
 * camera. This is what the described function does. It computes ( `$R$`, `$T$` ) so that:
 *
 * `\\[R_2=R*R_1\\]` `\\[T_2=R*T_1 + T,\\]`
 *
 * Optionally, it computes the essential matrix E:
 *
 * `\\[E= \\vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} *R\\]`
 *
 * where `$T_i$` are components of the translation vector `$T$` : `$T=[T_0, T_1, T_2]^T$` . And the
 * function can also compute the fundamental matrix F:
 *
 * `\\[F = cameraMatrix2^{-T} E cameraMatrix1^{-1}\\]`
 *
 * Besides the stereo-related information, the function can also perform a full calibration of each of
 * two cameras. However, due to the high dimensionality of the parameter space and noise in the input
 * data, the function can diverge from the correct solution. If the intrinsic parameters can be
 * estimated with high accuracy for each of the cameras individually (for example, using
 * calibrateCamera ), you are recommended to do so and then pass CALIB_FIX_INTRINSIC flag to the
 * function along with the computed intrinsic parameters. Otherwise, if all the parameters are
 * estimated at once, it makes sense to restrict some parameters, for example, pass
 * CALIB_SAME_FOCAL_LENGTH and CALIB_ZERO_TANGENT_DIST flags, which is usually a reasonable assumption.
 *
 * Similarly to calibrateCamera , the function minimizes the total re-projection error for all the
 * points in all the available views from both cameras. The function returns the final value of the
 * re-projection error.
 *
 * @param objectPoints Vector of vectors of the calibration pattern points.
 *
 * @param imagePoints1 Vector of vectors of the projections of the calibration pattern points, observed
 * by the first camera.
 *
 * @param imagePoints2 Vector of vectors of the projections of the calibration pattern points, observed
 * by the second camera.
 *
 * @param cameraMatrix1 Input/output first camera matrix:
 * $\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}$ , $j = 0,\, 1$ . If any
 * of CALIB_USE_INTRINSIC_GUESS , CALIB_FIX_ASPECT_RATIO , CALIB_FIX_INTRINSIC , or
 * CALIB_FIX_FOCAL_LENGTH are specified, some or all of the matrix components must be initialized. See
 * the flags description for details.
 *
 * @param distCoeffs1 Input/output vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4,
 * k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. The output
 * vector length depends on the flags.
 *
 * @param cameraMatrix2 Input/output second camera matrix. The parameter is similar to cameraMatrix1
 *
 * @param distCoeffs2 Input/output lens distortion coefficients for the second camera. The parameter is
 * similar to distCoeffs1 .
 *
 * @param imageSize Size of the image used only to initialize intrinsic camera matrix.
 *
 * @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
 *
 * @param T Output translation vector between the coordinate systems of the cameras.
 *
 * @param E Output essential matrix.
 *
 * @param F Output fundamental matrix.
 *
 * @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
 *
 * @param flags Different flags that may be zero or a combination of the following values:
 * CALIB_FIX_INTRINSIC Fix cameraMatrix? and distCoeffs? so that only R, T, E , and F matrices are
 * estimated.CALIB_USE_INTRINSIC_GUESS Optimize some or all of the intrinsic parameters according to
 * the specified flags. Initial values are provided by the user.CALIB_USE_EXTRINSIC_GUESS R, T contain
 * valid initial values that are optimized further. Otherwise R, T are initialized to the median value
 * of the pattern views (each dimension separately).CALIB_FIX_PRINCIPAL_POINT Fix the principal points
 * during the optimization.CALIB_FIX_FOCAL_LENGTH Fix $f^{(j)}_x$ and $f^{(j)}_y$
 * .CALIB_FIX_ASPECT_RATIO Optimize $f^{(j)}_y$ . Fix the ratio $f^{(j)}_x/f^{(j)}_y$
 *
 * CALIB_SAME_FOCAL_LENGTH Enforce $f^{(0)}_x=f^{(1)}_x$ and $f^{(0)}_y=f^{(1)}_y$
 * .CALIB_ZERO_TANGENT_DIST Set tangential distortion coefficients for each camera to zeros and fix
 * there.CALIB_FIX_K1,...,CALIB_FIX_K6 Do not change the corresponding radial distortion coefficient
 * during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied
 * distCoeffs matrix is used. Otherwise, it is set to 0.CALIB_RATIONAL_MODEL Enable coefficients k4,
 * k5, and k6. To provide the backward compatibility, this extra flag should be explicitly specified to
 * make the calibration function use the rational model and return 8 coefficients. If the flag is not
 * set, the function computes and returns only 5 distortion coefficients.CALIB_THIN_PRISM_MODEL
 * Coefficients s1, s2, s3 and s4 are enabled. To provide the backward compatibility, this extra flag
 * should be explicitly specified to make the calibration function use the thin prism model and return
 * 12 coefficients. If the flag is not set, the function computes and returns only 5 distortion
 * coefficients.CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during the
 * optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs
 * matrix is used. Otherwise, it is set to 0.CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled.
 * To provide the backward compatibility, this extra flag should be explicitly specified to make the
 * calibration function use the tilted sensor model and return 14 coefficients. If the flag is not set,
 * the function computes and returns only 5 distortion coefficients.CALIB_FIX_TAUX_TAUY The
 * coefficients of the tilted sensor model are not changed during the optimization. If
 * CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used.
 * Otherwise, it is set to 0.
 *
 * @param criteria Termination criteria for the iterative optimization algorithm.
 */
export declare function stereoCalibrate(
  objectPoints: InputArrayOfArrays,
  imagePoints1: InputArrayOfArrays,
  imagePoints2: InputArrayOfArrays,
  cameraMatrix1: InputOutputArray,
  distCoeffs1: InputOutputArray,
  cameraMatrix2: InputOutputArray,
  distCoeffs2: InputOutputArray,
  imageSize: Size,
  R: InputOutputArray,
  T: InputOutputArray,
  E: OutputArray,
  F: OutputArray,
  perViewErrors: OutputArray,
  flags?: int,
  criteria?: TermCriteria,
): double;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function stereoCalibrate(
  objectPoints: InputArrayOfArrays,
  imagePoints1: InputArrayOfArrays,
  imagePoints2: InputArrayOfArrays,
  cameraMatrix1: InputOutputArray,
  distCoeffs1: InputOutputArray,
  cameraMatrix2: InputOutputArray,
  distCoeffs2: InputOutputArray,
  imageSize: Size,
  R: OutputArray,
  T: OutputArray,
  E: OutputArray,
  F: OutputArray,
  flags?: int,
  criteria?: TermCriteria,
): double;

/**
 * The function computes the rotation matrices for each camera that (virtually) make both camera image
 * planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
 * the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
 * as input. As output, it provides two rotation matrices and also two projection matrices in the new
 * coordinates. The function distinguishes the following two cases:
 *
 * **Horizontal stereo**: the first and the second camera views are shifted relative to each other
 * mainly along the x axis (with possible small vertical shift). In the rectified images, the
 * corresponding epipolar lines in the left and right cameras are horizontal and have the same
 * y-coordinate. P1 and P2 look like:`\\[\\texttt{P1} = \\begin{bmatrix} f & 0 & cx_1 & 0 \\\\ 0 & f &
 * cy & 0 \\\\ 0 & 0 & 1 & 0 \\end{bmatrix}\\]``\\[\\texttt{P2} = \\begin{bmatrix} f & 0 & cx_2 & T_x*f
 * \\\\ 0 & f & cy & 0 \\\\ 0 & 0 & 1 & 0 \\end{bmatrix} ,\\]`where `$T_x$` is a horizontal shift
 * between the cameras and `$cx_1=cx_2$` if CALIB_ZERO_DISPARITY is set.
 * **Vertical stereo**: the first and the second camera views are shifted relative to each other mainly
 * in vertical direction (and probably a bit in the horizontal direction too). The epipolar lines in
 * the rectified images are vertical and have the same x-coordinate. P1 and P2 look
 * like:`\\[\\texttt{P1} = \\begin{bmatrix} f & 0 & cx & 0 \\\\ 0 & f & cy_1 & 0 \\\\ 0 & 0 & 1 & 0
 * \\end{bmatrix}\\]``\\[\\texttt{P2} = \\begin{bmatrix} f & 0 & cx & 0 \\\\ 0 & f & cy_2 & T_y*f \\\\
 * 0 & 0 & 1 & 0 \\end{bmatrix} ,\\]`where `$T_y$` is a vertical shift between the cameras and
 * `$cy_1=cy_2$` if CALIB_ZERO_DISPARITY is set.
 *
 * As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
 * matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
 * initialize the rectification map for each camera.
 *
 * See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
 * the corresponding image regions. This means that the images are well rectified, which is what most
 * stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
 * their interiors are all valid pixels.
 *
 * @param cameraMatrix1 First camera matrix.
 *
 * @param distCoeffs1 First camera distortion parameters.
 *
 * @param cameraMatrix2 Second camera matrix.
 *
 * @param distCoeffs2 Second camera distortion parameters.
 *
 * @param imageSize Size of the image used for stereo calibration.
 *
 * @param R Rotation matrix between the coordinate systems of the first and the second cameras.
 *
 * @param T Translation vector between coordinate systems of the cameras.
 *
 * @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
 *
 * @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
 *
 * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
 * camera.
 *
 * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
 * camera.
 *
 * @param Q Output $4 \times 4$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).
 *
 * @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set, the
 * function makes the principal points of each camera have the same pixel coordinates in the rectified
 * views. And if the flag is not set, the function may still shift the images in the horizontal or
 * vertical direction (depending on the orientation of epipolar lines) to maximize the useful image
 * area.
 *
 * @param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
 * scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images
 * are zoomed and shifted so that only valid pixels are visible (no black areas after rectification).
 * alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the
 * original images from the cameras are retained in the rectified images (no source image pixels are
 * lost). Obviously, any intermediate value yields an intermediate result between those two extreme
 * cases.
 *
 * @param newImageSize New image resolution after rectification. The same size should be passed to
 * initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0) is
 * passed (default), it is set to the original imageSize . Setting it to larger value can help you
 * preserve details in the original image, especially when there is a big radial distortion.
 *
 * @param validPixROI1 Optional output rectangles inside the rectified images where all the pixels are
 * valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see
 * the picture below).
 *
 * @param validPixROI2 Optional output rectangles inside the rectified images where all the pixels are
 * valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see
 * the picture below).
 */
export declare function stereoRectify(
  cameraMatrix1: InputArray,
  distCoeffs1: InputArray,
  cameraMatrix2: InputArray,
  distCoeffs2: InputArray,
  imageSize: Size,
  R: InputArray,
  T: InputArray,
  R1: OutputArray,
  R2: OutputArray,
  P1: OutputArray,
  P2: OutputArray,
  Q: OutputArray,
  flags?: int,
  alpha?: double,
  newImageSize?: Size,
  validPixROI1?: any,
  validPixROI2?: any,
): void;

/**
 * The function computes the rectification transformations without knowing intrinsic parameters of the
 * cameras and their relative position in the space, which explains the suffix "uncalibrated". Another
 * related difference from stereoRectify is that the function outputs not the rectification
 * transformations in the object (3D) space, but the planar perspective transformations encoded by the
 * homography matrices H1 and H2 . The function implements the algorithm Hartley99 .
 *
 * While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily
 * depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion, it
 * would be better to correct it before computing the fundamental matrix and calling this function. For
 * example, distortion coefficients can be estimated for each head of stereo camera separately by using
 * calibrateCamera . Then, the images can be corrected using undistort , or just the point coordinates
 * can be corrected with undistortPoints .
 *
 * @param points1 Array of feature points in the first image.
 *
 * @param points2 The corresponding points in the second image. The same formats as in
 * findFundamentalMat are supported.
 *
 * @param F Input fundamental matrix. It can be computed from the same set of point pairs using
 * findFundamentalMat .
 *
 * @param imgSize Size of the image.
 *
 * @param H1 Output rectification homography matrix for the first image.
 *
 * @param H2 Output rectification homography matrix for the second image.
 *
 * @param threshold Optional threshold used to filter out the outliers. If the parameter is greater
 * than zero, all the point pairs that do not comply with the epipolar geometry (that is, the points
 * for which $|\texttt{points2[i]}^T*\texttt{F}*\texttt{points1[i]}|>\texttt{threshold}$ ) are rejected
 * prior to computing the homographies. Otherwise, all the points are considered inliers.
 */
export declare function stereoRectifyUncalibrated(
  points1: InputArray,
  points2: InputArray,
  F: InputArray,
  imgSize: Size,
  H1: OutputArray,
  H2: OutputArray,
  threshold?: double,
): bool;

/**
 * The function reconstructs 3-dimensional points (in homogeneous coordinates) by using their
 * observations with a stereo camera. Projections matrices can be obtained from stereoRectify.
 *
 * Keep in mind that all input data should be of float type in order for this function to work.
 *
 * [reprojectImageTo3D]
 *
 * @param projMatr1 3x4 projection matrix of the first camera.
 *
 * @param projMatr2 3x4 projection matrix of the second camera.
 *
 * @param projPoints1 2xN array of feature points in the first image. In case of c++ version it can be
 * also a vector of feature points or two-channel matrix of size 1xN or Nx1.
 *
 * @param projPoints2 2xN array of corresponding points in the second image. In case of c++ version it
 * can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
 *
 * @param points4D 4xN array of reconstructed points in homogeneous coordinates.
 */
export declare function triangulatePoints(
  projMatr1: InputArray,
  projMatr2: InputArray,
  projPoints1: InputArray,
  projPoints2: InputArray,
  points4D: OutputArray,
): void;

/**
 * The function transforms an image to compensate radial and tangential lens distortion.
 *
 * The function is simply a combination of [initUndistortRectifyMap] (with unity R ) and [remap] (with
 * bilinear interpolation). See the former function for details of the transformation being performed.
 *
 * Those pixels in the destination image, for which there is no correspondent pixels in the source
 * image, are filled with zeros (black color).
 *
 * A particular subset of the source image that will be visible in the corrected image can be regulated
 * by newCameraMatrix. You can use [getOptimalNewCameraMatrix] to compute the appropriate
 * newCameraMatrix depending on your requirements.
 *
 * The camera matrix and the distortion parameters can be determined using [calibrateCamera]. If the
 * resolution of images is different from the resolution used at the calibration stage, `$f_x, f_y,
 * c_x$` and `$c_y$` need to be scaled accordingly, while the distortion coefficients remain the same.
 *
 * @param src Input (distorted) image.
 *
 * @param dst Output (corrected) image that has the same size and type as src .
 *
 * @param cameraMatrix Input camera matrix $A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5,
 * k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param newCameraMatrix Camera matrix of the distorted image. By default, it is the same as
 * cameraMatrix but you may additionally scale and shift the result by using a different matrix.
 */
export declare function undistort(
  src: InputArray,
  dst: OutputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  newCameraMatrix?: InputArray,
): void;

/**
 * The function is similar to [undistort] and [initUndistortRectifyMap] but it operates on a sparse set
 * of points instead of a raster image. Also the function performs a reverse transformation to
 * projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a planar
 * object, it does, up to a translation vector, if the proper R is specified.
 *
 * For each observed point coordinate `$(u, v)$` the function computes: `\\[ \\begin{array}{l} x^{"}
 * \\leftarrow (u - c_x)/f_x \\\\ y^{"} \\leftarrow (v - c_y)/f_y \\\\ (x',y') = undistort(x^{"},y^{"},
 * \\texttt{distCoeffs}) \\\\ {[X\\,Y\\,W]} ^T \\leftarrow R*[x' \\, y' \\, 1]^T \\\\ x \\leftarrow X/W
 * \\\\ y \\leftarrow Y/W \\\\ \\text{only performed if P is specified:} \\\\ u' \\leftarrow x {f'}_x +
 * {c'}_x \\\\ v' \\leftarrow y {f'}_y + {c'}_y \\end{array} \\]`
 *
 * where *undistort* is an approximate iterative algorithm that estimates the normalized original point
 * coordinates out of the normalized distorted point coordinates ("normalized" means that the
 * coordinates do not depend on the camera matrix).
 *
 * The function can be used for both a stereo camera head or a monocular camera (when R is empty).
 *
 * @param src Observed point coordinates, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2)
 * (or vector<Point2f> ).
 *
 * @param dst Output ideal point coordinates (1xN/Nx1 2-channel or vector<Point2f> ) after undistortion
 * and reverse perspective transformation. If matrix P is identity or omitted, dst will contain
 * normalized point coordinates.
 *
 * @param cameraMatrix Camera matrix $\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
 *
 * @param distCoeffs Input vector of distortion coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5,
 * k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is
 * NULL/empty, the zero distortion coefficients are assumed.
 *
 * @param R Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by
 * stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.
 *
 * @param P New camera matrix (3x3) or new projection matrix (3x4) $\begin{bmatrix} {f'}_x & 0 & {c'}_x
 * & t_x \\ 0 & {f'}_y & {c'}_y & t_y \\ 0 & 0 & 1 & t_z \end{bmatrix}$. P1 or P2 computed by
 * stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.
 */
export declare function undistortPoints(
  src: InputArray,
  dst: OutputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  R?: InputArray,
  P?: InputArray,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * Default version of [undistortPoints] does 5 iterations to compute undistorted points.
 */
export declare function undistortPoints(
  src: InputArray,
  dst: OutputArray,
  cameraMatrix: InputArray,
  distCoeffs: InputArray,
  R: InputArray,
  P: InputArray,
  criteria: TermCriteria,
): void;

export declare function validateDisparity(
  disparity: InputOutputArray,
  cost: InputArray,
  minDisparity: int,
  numberOfDisparities: int,
  disp12MaxDisp?: int,
): void;

export declare const LMEDS: any; // initializer: = 4

export declare const RANSAC: any; // initializer: = 8

export declare const RHO: any; // initializer: = 16

export declare const CALIB_CB_ADAPTIVE_THRESH: any; // initializer: = 1

export declare const CALIB_CB_NORMALIZE_IMAGE: any; // initializer: = 2

export declare const CALIB_CB_FILTER_QUADS: any; // initializer: = 4

export declare const CALIB_CB_FAST_CHECK: any; // initializer: = 8

export declare const CALIB_CB_EXHAUSTIVE: any; // initializer: = 16

export declare const CALIB_CB_ACCURACY: any; // initializer: = 32

export declare const CALIB_CB_SYMMETRIC_GRID: any; // initializer: = 1

export declare const CALIB_CB_ASYMMETRIC_GRID: any; // initializer: = 2

export declare const CALIB_CB_CLUSTERING: any; // initializer: = 4

export declare const CALIB_NINTRINSIC: any; // initializer: = 18

export declare const CALIB_USE_INTRINSIC_GUESS: any; // initializer: = 0x00001

export declare const CALIB_FIX_ASPECT_RATIO: any; // initializer: = 0x00002

export declare const CALIB_FIX_PRINCIPAL_POINT: any; // initializer: = 0x00004

export declare const CALIB_ZERO_TANGENT_DIST: any; // initializer: = 0x00008

export declare const CALIB_FIX_FOCAL_LENGTH: any; // initializer: = 0x00010

export declare const CALIB_FIX_K1: any; // initializer: = 0x00020

export declare const CALIB_FIX_K2: any; // initializer: = 0x00040

export declare const CALIB_FIX_K3: any; // initializer: = 0x00080

export declare const CALIB_FIX_K4: any; // initializer: = 0x00800

export declare const CALIB_FIX_K5: any; // initializer: = 0x01000

export declare const CALIB_FIX_K6: any; // initializer: = 0x02000

export declare const CALIB_RATIONAL_MODEL: any; // initializer: = 0x04000

export declare const CALIB_THIN_PRISM_MODEL: any; // initializer: = 0x08000

export declare const CALIB_FIX_S1_S2_S3_S4: any; // initializer: = 0x10000

export declare const CALIB_TILTED_MODEL: any; // initializer: = 0x40000

export declare const CALIB_FIX_TAUX_TAUY: any; // initializer: = 0x80000

export declare const CALIB_USE_QR: any; // initializer: = 0x100000

export declare const CALIB_FIX_TANGENT_DIST: any; // initializer: = 0x200000

export declare const CALIB_FIX_INTRINSIC: any; // initializer: = 0x00100

export declare const CALIB_SAME_FOCAL_LENGTH: any; // initializer: = 0x00200

export declare const CALIB_ZERO_DISPARITY: any; // initializer: = 0x00400

export declare const CALIB_USE_LU: any; // initializer: = (1 << 17)

export declare const CALIB_USE_EXTRINSIC_GUESS: any; // initializer: = (1 << 22)

export declare const FM_7POINT: any; // initializer: = 1

export declare const FM_8POINT: any; // initializer: = 2

export declare const FM_LMEDS: any; // initializer: = 4

export declare const FM_RANSAC: any; // initializer: = 8

export declare const CALIB_HAND_EYE_TSAI: HandEyeCalibrationMethod; // initializer: = 0

export declare const CALIB_HAND_EYE_PARK: HandEyeCalibrationMethod; // initializer: = 1

export declare const CALIB_HAND_EYE_HORAUD: HandEyeCalibrationMethod; // initializer: = 2

export declare const CALIB_HAND_EYE_ANDREFF: HandEyeCalibrationMethod; // initializer: = 3

export declare const CALIB_HAND_EYE_DANIILIDIS: HandEyeCalibrationMethod; // initializer: = 4

export declare const SOLVEPNP_ITERATIVE: SolvePnPMethod; // initializer: = 0

export declare const SOLVEPNP_EPNP: SolvePnPMethod; // initializer: = 1

export declare const SOLVEPNP_P3P: SolvePnPMethod; // initializer: = 2

export declare const SOLVEPNP_DLS: SolvePnPMethod; // initializer: = 3

export declare const SOLVEPNP_UPNP: SolvePnPMethod; // initializer: = 4

export declare const SOLVEPNP_AP3P: SolvePnPMethod; // initializer: = 5

/**
 * Infinitesimal Plane-Based Pose Estimation Collins14
 *  Object points must be coplanar.
 *
 */
export declare const SOLVEPNP_IPPE: SolvePnPMethod; // initializer: = 6

/**
 * Infinitesimal Plane-Based Pose Estimation Collins14
 *  This is a special case suitable for marker pose estimation.
 *  4 coplanar object points must be defined in the following order:
 *
 * point 0: [-squareLength / 2, squareLength / 2, 0]
 * point 1: [ squareLength / 2, squareLength / 2, 0]
 * point 2: [ squareLength / 2, -squareLength / 2, 0]
 * point 3: [-squareLength / 2, -squareLength / 2, 0]
 *
 */
export declare const SOLVEPNP_IPPE_SQUARE: SolvePnPMethod; // initializer: = 7

export declare const PROJ_SPHERICAL_ORTHO: UndistortTypes; // initializer: = 0

export declare const PROJ_SPHERICAL_EQRECT: UndistortTypes; // initializer: = 1

export type HandEyeCalibrationMethod = any;

export type SolvePnPMethod = any;

export type UndistortTypes = any;
