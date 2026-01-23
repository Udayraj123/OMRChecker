import type {
  bool,
  double,
  InputArray,
  InputOutputArray,
  int,
  OutputArray,
  Size,
  TermCriteria,
} from "./_types";
/*
 * # Feature Detection
 *
 */
/**
 * The function finds edges in the input image and marks them in the output map edges using the Canny
 * algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
 * largest value is used to find initial segments of strong edges. See
 *
 * @param image 8-bit input image.
 *
 * @param edges output edge map; single channels 8-bit image, which has the same size as image .
 *
 * @param threshold1 first threshold for the hysteresis procedure.
 *
 * @param threshold2 second threshold for the hysteresis procedure.
 *
 * @param apertureSize aperture size for the Sobel operator.
 *
 * @param L2gradient a flag, indicating whether a more accurate $L_2$ norm $=\sqrt{(dI/dx)^2 +
 * (dI/dy)^2}$ should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether
 * the default $L_1$ norm $=|dI/dx|+|dI/dy|$ is enough ( L2gradient=false ).
 */
export declare function Canny(
  image: InputArray,
  edges: OutputArray,
  threshold1: double,
  threshold2: double,
  apertureSize?: int,
  L2gradient?: bool,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * Finds edges in an image using the Canny algorithm with custom image gradient.
 *
 * @param dx 16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
 *
 * @param dy 16-bit y derivative of input image (same type as dx).
 *
 * @param edges output edge map; single channels 8-bit image, which has the same size as image .
 *
 * @param threshold1 first threshold for the hysteresis procedure.
 *
 * @param threshold2 second threshold for the hysteresis procedure.
 *
 * @param L2gradient a flag, indicating whether a more accurate $L_2$ norm $=\sqrt{(dI/dx)^2 +
 * (dI/dy)^2}$ should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether
 * the default $L_1$ norm $=|dI/dx|+|dI/dy|$ is enough ( L2gradient=false ).
 */
export declare function Canny(
  dx: InputArray,
  dy: InputArray,
  edges: OutputArray,
  threshold1: double,
  threshold2: double,
  L2gradient?: bool,
): void;

/**
 * For every pixel `$p$` , the function cornerEigenValsAndVecs considers a blockSize `$\\times$`
 * blockSize neighborhood `$S(p)$` . It calculates the covariation matrix of derivatives over the
 * neighborhood as:
 *
 * `\\[M = \\begin{bmatrix} \\sum _{S(p)}(dI/dx)^2 & \\sum _{S(p)}dI/dx dI/dy \\\\ \\sum _{S(p)}dI/dx
 * dI/dy & \\sum _{S(p)}(dI/dy)^2 \\end{bmatrix}\\]`
 *
 * where the derivatives are computed using the Sobel operator.
 *
 * After that, it finds eigenvectors and eigenvalues of `$M$` and stores them in the destination image
 * as `$(\\lambda_1, \\lambda_2, x_1, y_1, x_2, y_2)$` where
 *
 * `$\\lambda_1, \\lambda_2$` are the non-sorted eigenvalues of `$M$`
 * `$x_1, y_1$` are the eigenvectors corresponding to `$\\lambda_1$`
 * `$x_2, y_2$` are the eigenvectors corresponding to `$\\lambda_2$`
 *
 * The output of the function can be used for robust edge or corner detection.
 *
 * [cornerMinEigenVal], [cornerHarris], [preCornerDetect]
 *
 * @param src Input single-channel 8-bit or floating-point image.
 *
 * @param dst Image to store the results. It has the same size as src and the type CV_32FC(6) .
 *
 * @param blockSize Neighborhood size (see details below).
 *
 * @param ksize Aperture parameter for the Sobel operator.
 *
 * @param borderType Pixel extrapolation method. See BorderTypes.
 */
export declare function cornerEigenValsAndVecs(
  src: InputArray,
  dst: OutputArray,
  blockSize: int,
  ksize: int,
  borderType?: int,
): void;

/**
 * The function runs the Harris corner detector on the image. Similarly to cornerMinEigenVal and
 * cornerEigenValsAndVecs , for each pixel `$(x, y)$` it calculates a `$2\\times2$` gradient covariance
 * matrix `$M^{(x,y)}$` over a `$\\texttt{blockSize} \\times \\texttt{blockSize}$` neighborhood. Then,
 * it computes the following characteristic:
 *
 * `\\[\\texttt{dst} (x,y) = \\mathrm{det} M^{(x,y)} - k \\cdot \\left ( \\mathrm{tr} M^{(x,y)} \\right
 * )^2\\]`
 *
 * Corners in the image can be found as the local maxima of this response map.
 *
 * @param src Input single-channel 8-bit or floating-point image.
 *
 * @param dst Image to store the Harris detector responses. It has the type CV_32FC1 and the same size
 * as src .
 *
 * @param blockSize Neighborhood size (see the details on cornerEigenValsAndVecs ).
 *
 * @param ksize Aperture parameter for the Sobel operator.
 *
 * @param k Harris detector free parameter. See the formula above.
 *
 * @param borderType Pixel extrapolation method. See BorderTypes.
 */
export declare function cornerHarris(
  src: InputArray,
  dst: OutputArray,
  blockSize: int,
  ksize: int,
  k: double,
  borderType?: int,
): void;

/**
 * The function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal
 * eigenvalue of the covariance matrix of derivatives, that is, `$\\min(\\lambda_1, \\lambda_2)$` in
 * terms of the formulae in the cornerEigenValsAndVecs description.
 *
 * @param src Input single-channel 8-bit or floating-point image.
 *
 * @param dst Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as src
 * .
 *
 * @param blockSize Neighborhood size (see the details on cornerEigenValsAndVecs ).
 *
 * @param ksize Aperture parameter for the Sobel operator.
 *
 * @param borderType Pixel extrapolation method. See BorderTypes.
 */
export declare function cornerMinEigenVal(
  src: InputArray,
  dst: OutputArray,
  blockSize: int,
  ksize?: int,
  borderType?: int,
): void;

/**
 * The function iterates to find the sub-pixel accurate location of corners or radial saddle points, as
 * shown on the figure below.
 *
 *  Sub-pixel accurate corner locator is based on the observation that every vector from the center
 * `$q$` to a point `$p$` located within a neighborhood of `$q$` is orthogonal to the image gradient at
 * `$p$` subject to image and measurement noise. Consider the expression:
 *
 * `\\[\\epsilon _i = {DI_{p_i}}^T \\cdot (q - p_i)\\]`
 *
 * where `${DI_{p_i}}$` is an image gradient at one of the points `$p_i$` in a neighborhood of `$q$` .
 * The value of `$q$` is to be found so that `$\\epsilon_i$` is minimized. A system of equations may be
 * set up with `$\\epsilon_i$` set to zero:
 *
 * `\\[\\sum _i(DI_{p_i} \\cdot {DI_{p_i}}^T) \\cdot q - \\sum _i(DI_{p_i} \\cdot {DI_{p_i}}^T \\cdot
 * p_i)\\]`
 *
 * where the gradients are summed within a neighborhood ("search window") of `$q$` . Calling the first
 * gradient term `$G$` and the second gradient term `$b$` gives:
 *
 * `\\[q = G^{-1} \\cdot b\\]`
 *
 * The algorithm sets the center of the neighborhood window at this new center `$q$` and then iterates
 * until the center stays within a set threshold.
 *
 * @param image Input single-channel, 8-bit or float image.
 *
 * @param corners Initial coordinates of the input corners and refined coordinates provided for output.
 *
 * @param winSize Half of the side length of the search window. For example, if winSize=Size(5,5) ,
 * then a $(5*2+1) \times (5*2+1) = 11 \times 11$ search window is used.
 *
 * @param zeroZone Half of the size of the dead region in the middle of the search zone over which the
 * summation in the formula below is not done. It is used sometimes to avoid possible singularities of
 * the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
 *
 * @param criteria Criteria for termination of the iterative process of corner refinement. That is, the
 * process of corner position refinement stops either after criteria.maxCount iterations or when the
 * corner position moves by less than criteria.epsilon on some iteration.
 */
export declare function cornerSubPix(
  image: InputArray,
  corners: InputOutputArray,
  winSize: Size,
  zeroZone: Size,
  criteria: TermCriteria,
): void;

/**
 * The [LineSegmentDetector] algorithm is defined using the standard values. Only advanced users may
 * want to edit those, as to tailor it for their own application.
 *
 * Implementation has been removed due original code license conflict
 *
 * @param _refine The way found lines will be refined, see LineSegmentDetectorModes
 *
 * @param _scale The scale of the image that will be used to find the lines. Range (0..1].
 *
 * @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
 *
 * @param _quant Bound to the quantization error on the gradient norm.
 *
 * @param _ang_th Gradient angle tolerance in degrees.
 *
 * @param _log_eps Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is
 * chosen.
 *
 * @param _density_th Minimal density of aligned region points in the enclosing rectangle.
 *
 * @param _n_bins Number of bins in pseudo-ordering of gradient modulus.
 */
export declare function createLineSegmentDetector(
  _refine?: int,
  _scale?: double,
  _sigma_scale?: double,
  _quant?: double,
  _ang_th?: double,
  _log_eps?: double,
  _density_th?: double,
  _n_bins?: int,
): any;

/**
 * The function finds the most prominent corners in the image or in the specified image region, as
 * described in Shi94
 *
 * Function calculates the corner quality measure at every source image pixel using the
 * [cornerMinEigenVal] or [cornerHarris] .
 * Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
 * retained).
 * The corners with the minimal eigenvalue less than `$\\texttt{qualityLevel} \\cdot \\max_{x,y}
 * qualityMeasureMap(x,y)$` are rejected.
 * The remaining corners are sorted by the quality measure in the descending order.
 * Function throws away each corner for which there is a stronger corner at a distance less than
 * maxDistance.
 *
 * The function can be used to initialize a point-based tracker of an object.
 *
 * If the function is called with different values A and B of the parameter qualityLevel , and A > B,
 * the vector of returned corners with qualityLevel=A will be the prefix of the output vector with
 * qualityLevel=B .
 *
 * [cornerMinEigenVal], [cornerHarris], [calcOpticalFlowPyrLK], [estimateRigidTransform],
 *
 * @param image Input 8-bit or floating-point 32-bit, single-channel image.
 *
 * @param corners Output vector of detected corners.
 *
 * @param maxCorners Maximum number of corners to return. If there are more corners than are found, the
 * strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all
 * detected corners are returned.
 *
 * @param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
 * parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
 * (see cornerMinEigenVal ) or the Harris function response (see cornerHarris ). The corners with the
 * quality measure less than the product are rejected. For example, if the best corner has the quality
 * measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than
 * 15 are rejected.
 *
 * @param minDistance Minimum possible Euclidean distance between the returned corners.
 *
 * @param mask Optional region of interest. If the image is not empty (it needs to have the type
 * CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
 *
 * @param blockSize Size of an average block for computing a derivative covariation matrix over each
 * pixel neighborhood. See cornerEigenValsAndVecs .
 *
 * @param useHarrisDetector Parameter indicating whether to use a Harris detector (see cornerHarris) or
 * cornerMinEigenVal.
 *
 * @param k Free parameter of the Harris detector.
 */
export declare function goodFeaturesToTrack(
  image: InputArray,
  corners: OutputArray,
  maxCorners: int,
  qualityLevel: double,
  minDistance: double,
  mask?: InputArray,
  blockSize?: int,
  useHarrisDetector?: bool,
  k?: double,
): void;

export declare function goodFeaturesToTrack(
  image: InputArray,
  corners: OutputArray,
  maxCorners: int,
  qualityLevel: double,
  minDistance: double,
  mask: InputArray,
  blockSize: int,
  gradientSize: int,
  useHarrisDetector?: bool,
  k?: double,
): void;

/**
 * The function finds circles in a grayscale image using a modification of the Hough transform.
 *
 * Example: :
 *
 * ```cpp
 * #include <opencv2/imgproc.hpp>
 * #include <opencv2/highgui.hpp>
 * #include <math.h>
 *
 * using namespace cv;
 * using namespace std;
 *
 * int main(int argc, char** argv)
 * {
 *     Mat img, gray;
 *     if( argc != 2 || !(img=imread(argv[1], 1)).data)
 *         return -1;
 *     cvtColor(img, gray, COLOR_BGR2GRAY);
 *     // smooth it, otherwise a lot of false circles may be detected
 *     GaussianBlur( gray, gray, Size(9, 9), 2, 2 );
 *     vector<Vec3f> circles;
 *     HoughCircles(gray, circles, HOUGH_GRADIENT,
 *                  2, gray.rows/4, 200, 100 );
 *     for( size_t i = 0; i < circles.size(); i++ )
 *     {
 *          Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
 *          int radius = cvRound(circles[i][2]);
 *          // draw the circle center
 *          circle( img, center, 3, Scalar(0,255,0), -1, 8, 0 );
 *          // draw the circle outline
 *          circle( img, center, radius, Scalar(0,0,255), 3, 8, 0 );
 *     }
 *     namedWindow( "circles", 1 );
 *     imshow( "circles", img );
 *
 *     waitKey(0);
 *     return 0;
 * }
 * ```
 *
 * Usually the function detects the centers of circles well. However, it may fail to find correct
 * radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
 * you know it. Or, you may set maxRadius to a negative number to return centers only without radius
 * search, and find the correct radius using an additional procedure.
 *
 * [fitEllipse], [minEnclosingCircle]
 *
 * @param image 8-bit, single-channel, grayscale input image.
 *
 * @param circles Output vector of found circles. Each vector is encoded as 3 or 4 element
 * floating-point vector $(x, y, radius)$ or $(x, y, radius, votes)$ .
 *
 * @param method Detection method, see HoughModes. Currently, the only implemented method is
 * HOUGH_GRADIENT
 *
 * @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1
 * , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as
 * big width and height.
 *
 * @param minDist Minimum distance between the centers of the detected circles. If the parameter is too
 * small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too
 * large, some circles may be missed.
 *
 * @param param1 First method-specific parameter. In case of HOUGH_GRADIENT , it is the higher
 * threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
 *
 * @param param2 Second method-specific parameter. In case of HOUGH_GRADIENT , it is the accumulator
 * threshold for the circle centers at the detection stage. The smaller it is, the more false circles
 * may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
 *
 * @param minRadius Minimum circle radius.
 *
 * @param maxRadius Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, returns
 * centers without finding the radius.
 */
export declare function HoughCircles(
  image: InputArray,
  circles: OutputArray,
  method: int,
  dp: double,
  minDist: double,
  param1?: double,
  param2?: double,
  minRadius?: int,
  maxRadius?: int,
): void;

/**
 * The function implements the standard or standard multi-scale Hough transform algorithm for line
 * detection. See  for a good explanation of Hough transform.
 *
 * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
 *
 * @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector $(\rho,
 * \theta)$ or $(\rho, \theta, \textrm{votes})$ . $\rho$ is the distance from the coordinate origin
 * $(0,0)$ (top-left corner of the image). $\theta$ is the line rotation angle in radians ( $0 \sim
 * \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}$ ). $\textrm{votes}$ is the value of
 * accumulator.
 *
 * @param rho Distance resolution of the accumulator in pixels.
 *
 * @param theta Angle resolution of the accumulator in radians.
 *
 * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
 * votes ( $>\texttt{threshold}$ ).
 *
 * @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .
 * The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn
 * . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these parameters
 * should be positive.
 *
 * @param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
 *
 * @param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines.
 * Must fall between 0 and max_theta.
 *
 * @param max_theta For standard and multi-scale Hough transform, maximum angle to check for lines.
 * Must fall between min_theta and CV_PI.
 */
export declare function HoughLines(
  image: InputArray,
  lines: OutputArray,
  rho: double,
  theta: double,
  threshold: int,
  srn?: double,
  stn?: double,
  min_theta?: double,
  max_theta?: double,
): void;

/**
 * The function implements the probabilistic Hough transform algorithm for line detection, described in
 * Matas00
 *
 * See the line detection example below:
 *
 * ```cpp
 * #include <opencv2/imgproc.hpp>
 * #include <opencv2/highgui.hpp>
 *
 * using namespace cv;
 * using namespace std;
 *
 * int main(int argc, char** argv)
 * {
 *     Mat src, dst, color_dst;
 *     if( argc != 2 || !(src=imread(argv[1], 0)).data)
 *         return -1;
 *
 *     Canny( src, dst, 50, 200, 3 );
 *     cvtColor( dst, color_dst, COLOR_GRAY2BGR );
 *
 *     vector<Vec4i> lines;
 *     HoughLinesP( dst, lines, 1, CV_PI/180, 80, 30, 10 );
 *     for( size_t i = 0; i < lines.size(); i++ )
 *     {
 *         line( color_dst, Point(lines[i][0], lines[i][1]),
 *         Point( lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
 *     }
 *     namedWindow( "Source", 1 );
 *     imshow( "Source", src );
 *
 *     namedWindow( "Detected Lines", 1 );
 *     imshow( "Detected Lines", color_dst );
 *
 *     waitKey(0);
 *     return 0;
 * }
 * ```
 *
 *  This is a sample picture the function parameters have been tuned for:
 *
 *  And this is the output of the above program in case of the probabilistic Hough transform:
 *
 * [LineSegmentDetector]
 *
 * @param image 8-bit, single-channel binary source image. The image may be modified by the function.
 *
 * @param lines Output vector of lines. Each line is represented by a 4-element vector $(x_1, y_1, x_2,
 * y_2)$ , where $(x_1,y_1)$ and $(x_2, y_2)$ are the ending points of each detected line segment.
 *
 * @param rho Distance resolution of the accumulator in pixels.
 *
 * @param theta Angle resolution of the accumulator in radians.
 *
 * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
 * votes ( $>\texttt{threshold}$ ).
 *
 * @param minLineLength Minimum line length. Line segments shorter than that are rejected.
 *
 * @param maxLineGap Maximum allowed gap between points on the same line to link them.
 */
export declare function HoughLinesP(
  image: InputArray,
  lines: OutputArray,
  rho: double,
  theta: double,
  threshold: int,
  minLineLength?: double,
  maxLineGap?: double,
): void;

/**
 * The function finds lines in a set of points using a modification of the Hough transform.
 *
 * ```cpp
 * #include <opencv2/core.hpp>
 * #include <opencv2/imgproc.hpp>
 *
 * using namespace cv;
 * using namespace std;
 *
 * int main()
 * {
 *     Mat lines;
 *     vector<Vec3d> line3d;
 *     vector<Point2f> point;
 *     const static float Points[20][2] = {
 *     { 0.0f,   369.0f }, { 10.0f,  364.0f }, { 20.0f,  358.0f }, { 30.0f,  352.0f },
 *     { 40.0f,  346.0f }, { 50.0f,  341.0f }, { 60.0f,  335.0f }, { 70.0f,  329.0f },
 *     { 80.0f,  323.0f }, { 90.0f,  318.0f }, { 100.0f, 312.0f }, { 110.0f, 306.0f },
 *     { 120.0f, 300.0f }, { 130.0f, 295.0f }, { 140.0f, 289.0f }, { 150.0f, 284.0f },
 *     { 160.0f, 277.0f }, { 170.0f, 271.0f }, { 180.0f, 266.0f }, { 190.0f, 260.0f }
 *     };
 *
 *     for (int i = 0; i < 20; i++)
 *     {
 *         point.push_back(Point2f(Points[i][0],Points[i][1]));
 *     }
 *
 *     double rhoMin = 0.0f, rhoMax = 360.0f, rhoStep = 1;
 *     double thetaMin = 0.0f, thetaMax = CV_PI / 2.0f, thetaStep = CV_PI / 180.0f;
 *
 *     HoughLinesPointSet(point, lines, 20, 1,
 *                        rhoMin, rhoMax, rhoStep,
 *                        thetaMin, thetaMax, thetaStep);
 *
 *     lines.copyTo(line3d);
 *     printf("votes:%d, rho:%.7f, theta:%.7f\\n",(int)line3d.at(0).val[0], line3d.at(0).val[1],
 * line3d.at(0).val[2]);
 * }
 * ```
 *
 * @param _point Input vector of points. Each vector must be encoded as a Point vector $(x,y)$. Type
 * must be CV_32FC2 or CV_32SC2.
 *
 * @param _lines Output vector of found lines. Each vector is encoded as a vector<Vec3d> $(votes, rho,
 * theta)$. The larger the value of 'votes', the higher the reliability of the Hough line.
 *
 * @param lines_max Max count of hough lines.
 *
 * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough
 * votes ( $>\texttt{threshold}$ )
 *
 * @param min_rho Minimum Distance value of the accumulator in pixels.
 *
 * @param max_rho Maximum Distance value of the accumulator in pixels.
 *
 * @param rho_step Distance resolution of the accumulator in pixels.
 *
 * @param min_theta Minimum angle value of the accumulator in radians.
 *
 * @param max_theta Maximum angle value of the accumulator in radians.
 *
 * @param theta_step Angle resolution of the accumulator in radians.
 */
export declare function HoughLinesPointSet(
  _point: InputArray,
  _lines: OutputArray,
  lines_max: int,
  threshold: int,
  min_rho: double,
  max_rho: double,
  rho_step: double,
  min_theta: double,
  max_theta: double,
  theta_step: double,
): void;

/**
 * The function calculates the complex spatial derivative-based function of the source image
 *
 * `\\[\\texttt{dst} = (D_x \\texttt{src} )^2 \\cdot D_{yy} \\texttt{src} + (D_y \\texttt{src} )^2
 * \\cdot D_{xx} \\texttt{src} - 2 D_x \\texttt{src} \\cdot D_y \\texttt{src} \\cdot D_{xy}
 * \\texttt{src}\\]`
 *
 * where `$D_x$`, `$D_y$` are the first image derivatives, `$D_{xx}$`, `$D_{yy}$` are the second image
 * derivatives, and `$D_{xy}$` is the mixed derivative.
 *
 * The corners can be found as local maximums of the functions, as shown below:
 *
 * ```cpp
 * Mat corners, dilated_corners;
 * preCornerDetect(image, corners, 3);
 * // dilation with 3x3 rectangular structuring element
 * dilate(corners, dilated_corners, Mat(), 1);
 * Mat corner_mask = corners == dilated_corners;
 * ```
 *
 * @param src Source single-channel 8-bit of floating-point image.
 *
 * @param dst Output image that has the type CV_32F and the same size as src .
 *
 * @param ksize Aperture size of the Sobel .
 *
 * @param borderType Pixel extrapolation method. See BorderTypes.
 */
export declare function preCornerDetect(
  src: InputArray,
  dst: OutputArray,
  ksize: int,
  borderType?: int,
): void;

/**
 * classical or standard Hough transform. Every line is represented by two floating-point numbers
 * `$(\\rho, \\theta)$` , where `$\\rho$` is a distance between (0,0) point and the line, and
 * `$\\theta$` is the angle between x-axis and the normal to the line. Thus, the matrix must be (the
 * created sequence will be) of CV_32FC2 type
 *
 */
export declare const HOUGH_STANDARD: HoughModes; // initializer: = 0

/**
 * probabilistic Hough transform (more efficient in case if the picture contains a few long linear
 * segments). It returns line segments rather than the whole line. Each segment is represented by
 * starting and ending points, and the matrix must be (the created sequence will be) of the CV_32SC4
 * type.
 *
 */
export declare const HOUGH_PROBABILISTIC: HoughModes; // initializer: = 1

/**
 * multi-scale variant of the classical Hough transform. The lines are encoded the same way as
 * HOUGH_STANDARD.
 *
 */
export declare const HOUGH_MULTI_SCALE: HoughModes; // initializer: = 2

export declare const HOUGH_GRADIENT: HoughModes; // initializer: = 3

export declare const LSD_REFINE_NONE: LineSegmentDetectorModes; // initializer: = 0

export declare const LSD_REFINE_STD: LineSegmentDetectorModes; // initializer: = 1

/**
 * Advanced refinement. Number of false alarms is calculated, lines are refined through increase of
 * precision, decrement in size, etc.
 *
 */
export declare const LSD_REFINE_ADV: LineSegmentDetectorModes; // initializer: = 2

export type HoughModes = any;

export type LineSegmentDetectorModes = any;
