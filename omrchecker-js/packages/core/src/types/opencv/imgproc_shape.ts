import type {
  bool,
  Circle,
  double,
  float,
  InputArray,
  int,
  Moments,
  OutputArray,
  OutputArrayOfArrays,
  Point,
  Point2f,
  Rect,
  RotatedRect,
} from "./_types";

/*
 * # Structural Analysis and Shape Descriptors
 *
 */
/**
 * The function [cv::approxPolyDP] approximates a curve or a polygon with another curve/polygon with
 * less vertices so that the distance between them is less or equal to the specified precision. It uses
 * the Douglas-Peucker algorithm
 *
 * @param curve Input vector of a 2D point stored in std::vector or Mat
 *
 * @param approxCurve Result of the approximation. The type should match the type of the input curve.
 *
 * @param epsilon Parameter specifying the approximation accuracy. This is the maximum distance between
 * the original curve and its approximation.
 *
 * @param closed If true, the approximated curve is closed (its first and last vertices are connected).
 * Otherwise, it is not closed.
 */
export declare function approxPolyDP(
  curve: InputArray,
  approxCurve: OutputArray,
  epsilon: double,
  closed: bool,
): void;

/**
 * The function computes a curve length or a closed contour perimeter.
 *
 * @param curve Input vector of 2D points, stored in std::vector or Mat.
 *
 * @param closed Flag indicating whether the curve is closed or not.
 */
export declare function arcLength(curve: InputArray, closed: bool): double;

/**
 * The function calculates and returns the minimal up-right bounding rectangle for the specified point
 * set or non-zero pixels of gray-scale image.
 *
 * @param array Input gray-scale image or 2D point set, stored in std::vector or Mat.
 */
export declare function boundingRect(array: InputArray): Rect;

/**
 * The function finds the four vertices of a rotated rectangle. This function is useful to draw the
 * rectangle. In C++, instead of using this function, you can directly use [RotatedRect::points]
 * method. Please visit the [tutorial on Creating Bounding rotated boxes and ellipses for contours] for
 * more information.
 *
 * @param box The input rotated rectangle. It may be the output of
 *
 * @returns An array of four vertices of the rectangle (Point2f[])
 */
export declare function boxPoints(box: RotatedRect): Point2f[];

/**
 * image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0
 * represents the background label. ltype specifies the output label image type, an important
 * consideration based on the total number of labels or alternatively the total number of pixels in the
 * source image. ccltype specifies the connected components labeling algorithm to use, currently Grana
 * (BBDT) and Wu's (SAUF) algorithms are supported, see the [ConnectedComponentsAlgorithmsTypes] for
 * details. Note that SAUF algorithm forces a row major ordering of labels while BBDT does not. This
 * function uses parallel version of both Grana and Wu's algorithms if at least one allowed parallel
 * framework is enabled and if the rows of the image are at least twice the number returned by
 * [getNumberOfCPUs].
 *
 * @param image the 8-bit single-channel image to be labeled
 *
 * @param labels destination labeled image
 *
 * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
 *
 * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
 *
 * @param ccltype connected components algorithm type (see the ConnectedComponentsAlgorithmsTypes).
 */
export declare function connectedComponents(
  image: InputArray,
  labels: OutputArray,
  connectivity: int,
  ltype: int,
  ccltype: int,
): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param image the 8-bit single-channel image to be labeled
 *
 * @param labels destination labeled image
 *
 * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
 *
 * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
 */
export declare function connectedComponents(
  image: InputArray,
  labels: OutputArray,
  connectivity?: int,
  ltype?: int,
): int;

/**
 * image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0
 * represents the background label. ltype specifies the output label image type, an important
 * consideration based on the total number of labels or alternatively the total number of pixels in the
 * source image. ccltype specifies the connected components labeling algorithm to use, currently
 * Grana's (BBDT) and Wu's (SAUF) algorithms are supported, see the
 * [ConnectedComponentsAlgorithmsTypes] for details. Note that SAUF algorithm forces a row major
 * ordering of labels while BBDT does not. This function uses parallel version of both Grana and Wu's
 * algorithms (statistics included) if at least one allowed parallel framework is enabled and if the
 * rows of the image are at least twice the number returned by [getNumberOfCPUs].
 *
 * @param image the 8-bit single-channel image to be labeled
 *
 * @param labels destination labeled image
 *
 * @param stats statistics output for each label, including the background label, see below for
 * available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
 * ConnectedComponentsTypes. The data type is CV_32S.
 *
 * @param centroids centroid output for each label, including the background label. Centroids are
 * accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
 *
 * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
 *
 * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
 *
 * @param ccltype connected components algorithm type (see ConnectedComponentsAlgorithmsTypes).
 */
export declare function connectedComponentsWithStats(
  image: InputArray,
  labels: OutputArray,
  stats: OutputArray,
  centroids: OutputArray,
  connectivity: int,
  ltype: int,
  ccltype: int,
): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param image the 8-bit single-channel image to be labeled
 *
 * @param labels destination labeled image
 *
 * @param stats statistics output for each label, including the background label, see below for
 * available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
 * ConnectedComponentsTypes. The data type is CV_32S.
 *
 * @param centroids centroid output for each label, including the background label. Centroids are
 * accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
 *
 * @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
 *
 * @param ltype output image label type. Currently CV_32S and CV_16U are supported.
 */
export declare function connectedComponentsWithStats(
  image: InputArray,
  labels: OutputArray,
  stats: OutputArray,
  centroids: OutputArray,
  connectivity?: int,
  ltype?: int,
): int;

/**
 * The function computes a contour area. Similarly to moments , the area is computed using the Green
 * formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using
 * [drawContours] or [fillPoly] , can be different. Also, the function will most certainly give a wrong
 * results for contours with self-intersections.
 *
 * Example:
 *
 * ```cpp
 * vector<Point> contour;
 * contour.push_back(Point2f(0, 0));
 * contour.push_back(Point2f(10, 0));
 * contour.push_back(Point2f(10, 10));
 * contour.push_back(Point2f(5, 4));
 *
 * double area0 = contourArea(contour);
 * vector<Point> approx;
 * approxPolyDP(contour, approx, 5, true);
 * double area1 = contourArea(approx);
 *
 * cout << "area0 =" << area0 << endl <<
 *         "area1 =" << area1 << endl <<
 *         "approx poly vertices" << approx.size() << endl;
 * ```
 *
 * @param contour Input vector of 2D points (contour vertices), stored in std::vector or Mat.
 *
 * @param oriented Oriented area flag. If it is true, the function returns a signed area value,
 * depending on the contour orientation (clockwise or counter-clockwise). Using this feature you can
 * determine orientation of a contour by taking the sign of an area. By default, the parameter is
 * false, which means that the absolute value is returned.
 */
export declare function contourArea(
  contour: InputArray,
  oriented?: bool,
): double;

/**
 * The function [cv::convexHull] finds the convex hull of a 2D point set using the Sklansky's algorithm
 * Sklansky82 that has *O(N logN)* complexity in the current implementation.
 *
 * `points` and `hull` should be different arrays, inplace processing isn't supported.
 * Check [the corresponding tutorial] for more details.
 *
 * useful links:
 *
 * @param points Input 2D point set, stored in std::vector or Mat.
 *
 * @param hull Output convex hull. It is either an integer vector of indices or vector of points. In
 * the first case, the hull elements are 0-based indices of the convex hull points in the original
 * array (since the set of convex hull points is a subset of the original point set). In the second
 * case, hull elements are the convex hull points themselves.
 *
 * @param clockwise Orientation flag. If it is true, the output convex hull is oriented clockwise.
 * Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing
 * to the right, and its Y axis pointing upwards.
 *
 * @param returnPoints Operation flag. In case of a matrix, when the flag is true, the function returns
 * convex hull points. Otherwise, it returns indices of the convex hull points. When the output array
 * is std::vector, the flag is ignored, and the output depends on the type of the vector:
 * std::vector<int> implies returnPoints=false, std::vector<Point> implies returnPoints=true.
 */
export declare function convexHull(
  points: InputArray,
  hull: OutputArray,
  clockwise?: bool,
  returnPoints?: bool,
): void;

/**
 * The figure below displays convexity defects of a hand contour:
 *
 * @param contour Input contour.
 *
 * @param convexhull Convex hull obtained using convexHull that should contain indices of the contour
 * points that make the hull.
 *
 * @param convexityDefects The output vector of convexity defects. In C++ and the new Python/Java
 * interface each convexity defect is represented as 4-element integer vector (a.k.a. Vec4i):
 * (start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices in the
 * original contour of the convexity defect beginning, end and the farthest point, and fixpt_depth is
 * fixed-point approximation (with 8 fractional bits) of the distance between the farthest contour
 * point and the hull. That is, to get the floating-point value of the depth will be fixpt_depth/256.0.
 */
export declare function convexityDefects(
  contour: InputArray,
  convexhull: InputArray,
  convexityDefects: OutputArray,
): void;

export declare function createGeneralizedHoughBallard(): any;

export declare function createGeneralizedHoughGuil(): any;

/**
 * The function retrieves contours from the binary image using the algorithm Suzuki85 . The contours
 * are a useful tool for shape analysis and object detection and recognition. See squares.cpp in the
 * OpenCV sample directory.
 *
 * Since opencv 3.2 source image is not modified by this function.
 *
 * @param image Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels
 * remain 0's, so the image is treated as binary . You can use compare, inRange, threshold ,
 * adaptiveThreshold, Canny, and others to create a binary image out of a grayscale or color one. If
 * mode equals to RETR_CCOMP or RETR_FLOODFILL, the input can also be a 32-bit integer image of labels
 * (CV_32SC1).
 *
 * @param contours Detected contours. Each contour is stored as a vector of points (e.g.
 * std::vector<std::vector<cv::Point> >).
 *
 * @param hierarchy Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about
 * the image topology. It has as many elements as the number of contours. For each i-th contour
 * contours[i], the elements hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3]
 * are set to 0-based indices in contours of the next and previous contours at the same hierarchical
 * level, the first child contour and the parent contour, respectively. If for the contour i there are
 * no next, previous, parent, or nested contours, the corresponding elements of hierarchy[i] will be
 * negative.
 *
 * @param mode Contour retrieval mode, see RetrievalModes
 *
 * @param method Contour approximation method, see ContourApproximationModes
 *
 * @param offset Optional offset by which every contour point is shifted. This is useful if the
 * contours are extracted from the image ROI and then they should be analyzed in the whole image
 * context.
 */
export declare function findContours(
  image: InputArray,
  contours: OutputArrayOfArrays,
  hierarchy: OutputArray,
  mode: int,
  method: int,
  offset?: Point,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function findContours(
  image: InputArray,
  contours: OutputArrayOfArrays,
  mode: int,
  method: int,
  offset?: Point,
): void;

/**
 * The function calculates the ellipse that fits (in a least-squares sense) a set of 2D points best of
 * all. It returns the rotated rectangle in which the ellipse is inscribed. The first algorithm
 * described by Fitzgibbon95 is used. Developer should keep in mind that it is possible that the
 * returned ellipse/rotatedRect data contains negative indices, due to the data points being close to
 * the border of the containing [Mat] element.
 *
 * @param points Input 2D point set, stored in std::vector<> or Mat
 */
export declare function fitEllipse(points: InputArray): RotatedRect;

/**
 * The function calculates the ellipse that fits a set of 2D points. It returns the rotated rectangle
 * in which the ellipse is inscribed. The Approximate Mean Square (AMS) proposed by Taubin1991 is used.
 *
 * For an ellipse, this basis set is `$ \\chi= \\left(x^2, x y, y^2, x, y, 1\\right) $`, which is a set
 * of six free coefficients `$
 * A^T=\\left\\{A_{\\text{xx}},A_{\\text{xy}},A_{\\text{yy}},A_x,A_y,A_0\\right\\} $`. However, to
 * specify an ellipse, all that is needed is five numbers; the major and minor axes lengths `$ (a,b)
 * $`, the position `$ (x_0,y_0) $`, and the orientation `$ \\theta $`. This is because the basis set
 * includes lines, quadratics, parabolic and hyperbolic functions as well as elliptical functions as
 * possible fits. If the fit is found to be a parabolic or hyperbolic function then the standard
 * [fitEllipse] method is used. The AMS method restricts the fit to parabolic, hyperbolic and
 * elliptical curves by imposing the condition that `$ A^T ( D_x^T D_x + D_y^T D_y) A = 1 $` where the
 * matrices `$ Dx $` and `$ Dy $` are the partial derivatives of the design matrix `$ D $` with respect
 * to x and y. The matrices are formed row by row applying the following to each of the points in the
 * set: `\\begin{align*} D(i,:)&=\\left\\{x_i^2, x_i y_i, y_i^2, x_i, y_i, 1\\right\\} &
 * D_x(i,:)&=\\left\\{2 x_i,y_i,0,1,0,0\\right\\} & D_y(i,:)&=\\left\\{0,x_i,2 y_i,0,1,0\\right\\}
 * \\end{align*}` The AMS method minimizes the cost function `\\begin{equation*} \\epsilon ^2=\\frac{
 * A^T D^T D A }{ A^T (D_x^T D_x + D_y^T D_y) A^T } \\end{equation*}`
 *
 * The minimum cost is found by solving the generalized eigenvalue problem.
 *
 * `\\begin{equation*} D^T D A = \\lambda \\left( D_x^T D_x + D_y^T D_y\\right) A \\end{equation*}`
 *
 * @param points Input 2D point set, stored in std::vector<> or Mat
 */
export declare function fitEllipseAMS(points: InputArray): RotatedRect;

/**
 * The function calculates the ellipse that fits a set of 2D points. It returns the rotated rectangle
 * in which the ellipse is inscribed. The Direct least square (Direct) method by Fitzgibbon1999 is
 * used.
 *
 * For an ellipse, this basis set is `$ \\chi= \\left(x^2, x y, y^2, x, y, 1\\right) $`, which is a set
 * of six free coefficients `$
 * A^T=\\left\\{A_{\\text{xx}},A_{\\text{xy}},A_{\\text{yy}},A_x,A_y,A_0\\right\\} $`. However, to
 * specify an ellipse, all that is needed is five numbers; the major and minor axes lengths `$ (a,b)
 * $`, the position `$ (x_0,y_0) $`, and the orientation `$ \\theta $`. This is because the basis set
 * includes lines, quadratics, parabolic and hyperbolic functions as well as elliptical functions as
 * possible fits. The Direct method confines the fit to ellipses by ensuring that `$ 4 A_{xx} A_{yy}-
 * A_{xy}^2 > 0 $`. The condition imposed is that `$ 4 A_{xx} A_{yy}- A_{xy}^2=1 $` which satisfies the
 * inequality and as the coefficients can be arbitrarily scaled is not overly restrictive.
 *
 * `\\begin{equation*} \\epsilon ^2= A^T D^T D A \\quad \\text{with} \\quad A^T C A =1 \\quad
 * \\text{and} \\quad C=\\left(\\begin{matrix} 0 & 0 & 2 & 0 & 0 & 0 \\\\ 0 & -1 & 0 & 0 & 0 & 0 \\\\ 2
 * & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 0
 * \\end{matrix} \\right) \\end{equation*}`
 *
 * The minimum cost is found by solving the generalized eigenvalue problem.
 *
 * `\\begin{equation*} D^T D A = \\lambda \\left( C\\right) A \\end{equation*}`
 *
 * The system produces only one positive eigenvalue `$ \\lambda$` which is chosen as the solution with
 * its eigenvector `$\\mathbf{u}$`. These are used to find the coefficients
 *
 * `\\begin{equation*} A = \\sqrt{\\frac{1}{\\mathbf{u}^T C \\mathbf{u}}} \\mathbf{u} \\end{equation*}`
 * The scaling factor guarantees that `$A^T C A =1$`.
 *
 * @param points Input 2D point set, stored in std::vector<> or Mat
 */
export declare function fitEllipseDirect(points: InputArray): RotatedRect;

/**
 * The function fitLine fits a line to a 2D or 3D point set by minimizing `$\\sum_i \\rho(r_i)$` where
 * `$r_i$` is a distance between the `$i^{th}$` point, the line and `$\\rho(r)$` is a distance
 * function, one of the following:
 *
 * DIST_L2 `\\[\\rho (r) = r^2/2 \\quad \\text{(the simplest and the fastest least-squares method)}\\]`
 * DIST_L1 `\\[\\rho (r) = r\\]`
 * DIST_L12 `\\[\\rho (r) = 2 \\cdot ( \\sqrt{1 + \\frac{r^2}{2}} - 1)\\]`
 * DIST_FAIR `\\[\\rho \\left (r \\right ) = C^2 \\cdot \\left ( \\frac{r}{C} - \\log{\\left(1 +
 * \\frac{r}{C}\\right)} \\right ) \\quad \\text{where} \\quad C=1.3998\\]`
 * DIST_WELSCH `\\[\\rho \\left (r \\right ) = \\frac{C^2}{2} \\cdot \\left ( 1 -
 * \\exp{\\left(-\\left(\\frac{r}{C}\\right)^2\\right)} \\right ) \\quad \\text{where} \\quad
 * C=2.9846\\]`
 * DIST_HUBER `\\[\\rho (r) = \\fork{r^2/2}{if \\(r < C\\)}{C \\cdot (r-C/2)}{otherwise} \\quad
 * \\text{where} \\quad C=1.345\\]`
 *
 * The algorithm is based on the M-estimator (  ) technique that iteratively fits the line using the
 * weighted least-squares algorithm. After each iteration the weights `$w_i$` are adjusted to be
 * inversely proportional to `$\\rho(r_i)$` .
 *
 * @param points Input vector of 2D or 3D points, stored in std::vector<> or Mat.
 *
 * @param line Output line parameters. In case of 2D fitting, it should be a vector of 4 elements (like
 * Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and (x0, y0)
 * is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like Vec6f) -
 * (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line and (x0,
 * y0, z0) is a point on the line.
 *
 * @param distType Distance used by the M-estimator, see DistanceTypes
 *
 * @param param Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value is
 * chosen.
 *
 * @param reps Sufficient accuracy for the radius (distance between the coordinate origin and the
 * line).
 *
 * @param aeps Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.
 */
export declare function fitLine(
  points: InputArray,
  line: OutputArray,
  distType: int,
  param: double,
  reps: double,
  aeps: double,
): void;

/**
 * The function calculates seven Hu invariants (introduced in Hu62; see also ) defined as:
 *
 * `\\[\\begin{array}{l} hu[0]= \\eta _{20}+ \\eta _{02} \\\\ hu[1]=( \\eta _{20}- \\eta _{02})^{2}+4
 * \\eta _{11}^{2} \\\\ hu[2]=( \\eta _{30}-3 \\eta _{12})^{2}+ (3 \\eta _{21}- \\eta _{03})^{2} \\\\
 * hu[3]=( \\eta _{30}+ \\eta _{12})^{2}+ ( \\eta _{21}+ \\eta _{03})^{2} \\\\ hu[4]=( \\eta _{30}-3
 * \\eta _{12})( \\eta _{30}+ \\eta _{12})[( \\eta _{30}+ \\eta _{12})^{2}-3( \\eta _{21}+ \\eta
 * _{03})^{2}]+(3 \\eta _{21}- \\eta _{03})( \\eta _{21}+ \\eta _{03})[3( \\eta _{30}+ \\eta
 * _{12})^{2}-( \\eta _{21}+ \\eta _{03})^{2}] \\\\ hu[5]=( \\eta _{20}- \\eta _{02})[( \\eta _{30}+
 * \\eta _{12})^{2}- ( \\eta _{21}+ \\eta _{03})^{2}]+4 \\eta _{11}( \\eta _{30}+ \\eta _{12})( \\eta
 * _{21}+ \\eta _{03}) \\\\ hu[6]=(3 \\eta _{21}- \\eta _{03})( \\eta _{21}+ \\eta _{03})[3( \\eta
 * _{30}+ \\eta _{12})^{2}-( \\eta _{21}+ \\eta _{03})^{2}]-( \\eta _{30}-3 \\eta _{12})( \\eta _{21}+
 * \\eta _{03})[3( \\eta _{30}+ \\eta _{12})^{2}-( \\eta _{21}+ \\eta _{03})^{2}] \\\\ \\end{array}\\]`
 *
 * where `$\\eta_{ji}$` stands for `$\\texttt{Moments::nu}_{ji}$` .
 *
 * These values are proved to be invariants to the image scale, rotation, and reflection except the
 * seventh one, whose sign is changed by reflection. This invariance is proved with the assumption of
 * infinite image resolution. In case of raster images, the computed Hu invariants for the original and
 * transformed images are a bit different.
 *
 * [matchShapes]
 *
 * @param moments Input moments computed with moments .
 *
 * @param hu Output Hu invariants.
 */
export declare function HuMoments(moments: any, hu: double): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function HuMoments(m: any, hu: OutputArray): void;

export declare function intersectConvexConvex(
  _p1: InputArray,
  _p2: InputArray,
  _p12: OutputArray,
  handleNested?: bool,
): float;

/**
 * The function tests whether the input contour is convex or not. The contour must be simple, that is,
 * without self-intersections. Otherwise, the function output is undefined.
 *
 * @param contour Input vector of 2D points, stored in std::vector<> or Mat
 */
export declare function isContourConvex(contour: InputArray): bool;

/**
 * The function compares two shapes. All three implemented methods use the Hu invariants (see
 * [HuMoments])
 *
 * @param contour1 First contour or grayscale image.
 *
 * @param contour2 Second contour or grayscale image.
 *
 * @param method Comparison method, see ShapeMatchModes
 *
 * @param parameter Method-specific parameter (not supported now).
 */
export declare function matchShapes(
  contour1: InputArray,
  contour2: InputArray,
  method: int,
  parameter: double,
): double;

/**
 * The function calculates and returns the minimum-area bounding rectangle (possibly rotated) for a
 * specified point set. Developer should keep in mind that the returned [RotatedRect] can contain
 * negative indices when data is close to the containing [Mat] element boundary.
 *
 * @param points Input vector of 2D points, stored in std::vector<> or Mat
 */
export declare function minAreaRect(points: InputArray): RotatedRect;

/**
 * The function finds the minimal enclosing circle of a 2D point set using an iterative algorithm.
 *
 * @param points Input vector of 2D points, stored in std::vector<> or Mat
 */
export declare function minEnclosingCircle(points: InputArray): Circle;

/**
 * The function finds a triangle of minimum area enclosing the given set of 2D points and returns its
 * area. The output for a given 2D point set is shown in the image below. 2D points are depicted in
 * red* and the enclosing triangle in *yellow*.
 *
 *  The implementation of the algorithm is based on O'Rourke's ORourke86 and Klee and Laskowski's
 * KleeLaskowski85 papers. O'Rourke provides a `$\\theta(n)$` algorithm for finding the minimal
 * enclosing triangle of a 2D convex polygon with n vertices. Since the [minEnclosingTriangle] function
 * takes a 2D point set as input an additional preprocessing step of computing the convex hull of the
 * 2D point set is required. The complexity of the [convexHull] function is `$O(n log(n))$` which is
 * higher than `$\\theta(n)$`. Thus the overall complexity of the function is `$O(n log(n))$`.
 *
 * @param points Input vector of 2D points with depth CV_32S or CV_32F, stored in std::vector<> or Mat
 *
 * @param triangle Output vector of three 2D points defining the vertices of the triangle. The depth of
 * the OutputArray must be CV_32F.
 */
export declare function minEnclosingTriangle(
  points: InputArray,
  triangle: OutputArray,
): double;

/**
 * The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. The
 * results are returned in the structure [cv::Moments].
 *
 * moments.
 *
 * Only applicable to contour moments calculations from Python bindings: Note that the numpy type for
 * the input array should be either np.int32 or np.float32.
 *
 * [contourArea], [arcLength]
 *
 * @param array Raster image (single-channel, 8-bit or floating-point 2D array) or an array ( $1 \times
 * N$ or $N \times 1$ ) of 2D points (Point or Point2f ).
 *
 * @param binaryImage If it is true, all non-zero image pixels are treated as 1's. The parameter is
 * used for images only.
 */
export declare function moments(array: InputArray, binaryImage?: bool): Moments;

/**
 * The function determines whether the point is inside a contour, outside, or lies on an edge (or
 * coincides with a vertex). It returns positive (inside), negative (outside), or zero (on an edge)
 * value, correspondingly. When measureDist=false , the return value is +1, -1, and 0, respectively.
 * Otherwise, the return value is a signed distance between the point and the nearest contour edge.
 *
 * See below a sample output of the function where each image pixel is tested against the contour:
 *
 * @param contour Input contour.
 *
 * @param pt Point tested against the contour.
 *
 * @param measureDist If true, the function estimates the signed distance from the point to the nearest
 * contour edge. Otherwise, the function only checks if the point is inside a contour or not.
 */
export declare function pointPolygonTest(
  contour: InputArray,
  pt: Point2f,
  measureDist: bool,
): double;

/**
 * If there is then the vertices of the intersecting region are returned as well.
 *
 * Below are some examples of intersection configurations. The hatched pattern indicates the
 * intersecting region and the red vertices are returned by the function.
 *
 * One of [RectanglesIntersectTypes]
 *
 * @param rect1 First rectangle
 *
 * @param rect2 Second rectangle
 *
 * @param intersectingRegion The output array of the vertices of the intersecting region. It returns at
 * most 8 vertices. Stored as std::vector<cv::Point2f> or cv::Mat as Mx1 of type CV_32FC2.
 */
export declare function rotatedRectangleIntersection(
  rect1: any,
  rect2: any,
  intersectingRegion: OutputArray,
): int;

export declare const CCL_WU: ConnectedComponentsAlgorithmsTypes; // initializer: = 0

export declare const CCL_DEFAULT: ConnectedComponentsAlgorithmsTypes; // initializer: = -1

export declare const CCL_GRANA: ConnectedComponentsAlgorithmsTypes; // initializer: = 1

/**
 * The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal
 * direction.
 *
 */
export declare const CC_STAT_LEFT: ConnectedComponentsTypes; // initializer: = 0

/**
 * The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical
 * direction.
 *
 */
export declare const CC_STAT_TOP: ConnectedComponentsTypes; // initializer: = 1

export declare const CC_STAT_WIDTH: ConnectedComponentsTypes; // initializer: = 2

export declare const CC_STAT_HEIGHT: ConnectedComponentsTypes; // initializer: = 3

export declare const CC_STAT_AREA: ConnectedComponentsTypes; // initializer: = 4

export declare const CC_STAT_MAX: ConnectedComponentsTypes; // initializer: = 5

/**
 * stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of
 * the contour will be either horizontal, vertical or diagonal neighbors, that is,
 * max(abs(x1-x2),abs(y2-y1))==1.
 *
 */
export declare const CHAIN_APPROX_NONE: ContourApproximationModes; // initializer: = 1

/**
 * compresses horizontal, vertical, and diagonal segments and leaves only their end points. For
 * example, an up-right rectangular contour is encoded with 4 points.
 *
 */
export declare const CHAIN_APPROX_SIMPLE: ContourApproximationModes; // initializer: = 2

/**
 * applies one of the flavors of the Teh-Chin chain approximation algorithm TehChin89
 *
 */
export declare const CHAIN_APPROX_TC89_L1: ContourApproximationModes; // initializer: = 3

/**
 * applies one of the flavors of the Teh-Chin chain approximation algorithm TehChin89
 *
 */
export declare const CHAIN_APPROX_TC89_KCOS: ContourApproximationModes; // initializer: = 4

export declare const INTERSECT_NONE: RectanglesIntersectTypes; // initializer: = 0

export declare const INTERSECT_PARTIAL: RectanglesIntersectTypes; // initializer: = 1

export declare const INTERSECT_FULL: RectanglesIntersectTypes; // initializer: = 2

/**
 * retrieves only the extreme outer contours. It sets `hierarchy[i][2]=hierarchy[i][3]=-1` for all the
 * contours.
 *
 */
export declare const RETR_EXTERNAL: RetrievalModes; // initializer: = 0

/**
 * retrieves all of the contours without establishing any hierarchical relationships.
 *
 */
export declare const RETR_LIST: RetrievalModes; // initializer: = 1

/**
 * retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there
 * are external boundaries of the components. At the second level, there are boundaries of the holes.
 * If there is another contour inside a hole of a connected component, it is still put at the top
 * level.
 *
 */
export declare const RETR_CCOMP: RetrievalModes; // initializer: = 2

/**
 * retrieves all of the contours and reconstructs a full hierarchy of nested contours.
 *
 */
export declare const RETR_TREE: RetrievalModes; // initializer: = 3

export declare const RETR_FLOODFILL: RetrievalModes; // initializer: = 4

export declare const CONTOURS_MATCH_I1: ShapeMatchModes; // initializer: =1

export declare const CONTOURS_MATCH_I2: ShapeMatchModes; // initializer: =2

export declare const CONTOURS_MATCH_I3: ShapeMatchModes; // initializer: =3

export type ConnectedComponentsAlgorithmsTypes = any;

export type ConnectedComponentsTypes = any;

export type ContourApproximationModes = any;

export type RectanglesIntersectTypes = any;

export type RetrievalModes = any;

export type ShapeMatchModes = any;
