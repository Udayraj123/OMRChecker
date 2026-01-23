import type {
  bool,
  double,
  InputArray,
  InputArrayOfArrays,
  InputOutputArray,
  int,
  Point,
  Point2d,
  Rect,
  Scalar,
  Size,
  Size2d,
  Size2l,
} from "./_types";
/*
 * # Drawing Functions
 * Drawing functions work with matrices/images of arbitrary depth. The boundaries of the shapes can be rendered with antialiasing (implemented only for 8-bit images for now). All the functions include the parameter color that uses an RGB value (that may be constructed with the Scalar constructor ) for color images and brightness for grayscale images. For color images, the channel ordering is normally *Blue, Green, Red*. This is what imshow, imread, and imwrite expect. So, if you form a color using the Scalar constructor, it should look like:
 *
 * `\[\texttt{Scalar} (blue \_ component, green \_ component, red \_ component[, alpha \_ component])\]`
 *
 * If you are using your own image rendering and I/O functions, you can use any channel ordering. The drawing functions process each channel independently and do not depend on the channel order or even on the used color space. The whole image can be converted from BGR to RGB or to a different color space using cvtColor .
 *
 * If a drawn figure is partially or completely outside the image, the drawing functions clip it. Also, many drawing functions can handle pixel coordinates specified with sub-pixel accuracy. This means that the coordinates can be passed as fixed-point numbers encoded as integers. The number of fractional bits is specified by the shift parameter and the real point coordinates are calculated as `$\texttt{Point}(x,y)\rightarrow\texttt{Point2f}(x*2^{-shift},y*2^{-shift})$` . This feature is especially effective when rendering antialiased shapes.
 *
 *
 *
 * The functions do not support alpha-transparency when the target image is 4-channel. In this case, the color[3] is simply copied to the repainted pixels. Thus, if you want to paint semi-transparent shapes, you can paint them in a separate buffer and then blend it with the main image.
 */
/**
 * The function [cv::arrowedLine] draws an arrow between pt1 and pt2 points in the image. See also
 * [line].
 *
 * @param img Image.
 *
 * @param pt1 The point the arrow starts from.
 *
 * @param pt2 The point the arrow points to.
 *
 * @param color Line color.
 *
 * @param thickness Line thickness.
 *
 * @param line_type Type of the line. See LineTypes
 *
 * @param shift Number of fractional bits in the point coordinates.
 *
 * @param tipLength The length of the arrow tip in relation to the arrow length
 */
export declare function arrowedLine(
  img: InputOutputArray,
  pt1: Point,
  pt2: Point,
  color: any,
  thickness?: int,
  line_type?: int,
  shift?: int,
  tipLength?: double,
): void;

/**
 * The function [cv::circle] draws a simple or filled circle with a given center and radius.
 *
 * @param img Image where the circle is drawn.
 *
 * @param center Center of the circle.
 *
 * @param radius Radius of the circle.
 *
 * @param color Circle color.
 *
 * @param thickness Thickness of the circle outline, if positive. Negative values, like FILLED, mean
 * that a filled circle is to be drawn.
 *
 * @param lineType Type of the circle boundary. See LineTypes
 *
 * @param shift Number of fractional bits in the coordinates of the center and in the radius value.
 */
export declare function circle(
  img: InputOutputArray,
  center: Point,
  radius: int,
  color: any,
  thickness?: int,
  lineType?: int,
  shift?: int,
): void;

/**
 * The function [cv::clipLine] calculates a part of the line segment that is entirely within the
 * specified rectangle. it returns false if the line segment is completely outside the rectangle.
 * Otherwise, it returns true .
 *
 * @param imgSize Image size. The image rectangle is Rect(0, 0, imgSize.width, imgSize.height) .
 *
 * @param pt1 First line point.
 *
 * @param pt2 Second line point.
 */
export declare function clipLine(imgSize: Size, pt1: any, pt2: any): bool;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param imgSize Image size. The image rectangle is Rect(0, 0, imgSize.width, imgSize.height) .
 *
 * @param pt1 First line point.
 *
 * @param pt2 Second line point.
 */
export declare function clipLine(imgSize: Size2l, pt1: any, pt2: any): bool;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param imgRect Image rectangle.
 *
 * @param pt1 First line point.
 *
 * @param pt2 Second line point.
 */
export declare function clipLine(imgRect: Rect, pt1: any, pt2: any): bool;

/**
 * The function draws contour outlines in the image if `$\\texttt{thickness} \\ge 0$` or fills the area
 * bounded by the contours if `$\\texttt{thickness}<0$` . The example below shows how to retrieve
 * connected components from the binary image and label them: :
 *
 * ```cpp
 * #include "opencv2/imgproc.hpp"
 * #include "opencv2/highgui.hpp"
 *
 * using namespace cv;
 * using namespace std;
 *
 * int main( int argc, char** argv )
 * {
 *     Mat src;
 *     // the first command-line parameter must be a filename of the binary
 *     // (black-n-white) image
 *     if( argc != 2 || !(src=imread(argv[1], 0)).data)
 *         return -1;
 *
 *     Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
 *
 *     src = src > 1;
 *     namedWindow( "Source", 1 );
 *     imshow( "Source", src );
 *
 *     vector<vector<Point> > contours;
 *     vector<Vec4i> hierarchy;
 *
 *     findContours( src, contours, hierarchy,
 *         RETR_CCOMP, CHAIN_APPROX_SIMPLE );
 *
 *     // iterate through all the top-level contours,
 *     // draw each connected component with its own random color
 *     int idx = 0;
 *     for( ; idx >= 0; idx = hierarchy[idx][0] )
 *     {
 *         Scalar color( rand()&255, rand()&255, rand()&255 );
 *         drawContours( dst, contours, idx, color, FILLED, 8, hierarchy );
 *     }
 *
 *     namedWindow( "Components", 1 );
 *     imshow( "Components", dst );
 *     waitKey(0);
 * }
 * ```
 *
 * When thickness=[FILLED], the function is designed to handle connected components with holes
 * correctly even when no hierarchy date is provided. This is done by analyzing all the outlines
 * together using even-odd rule. This may give incorrect results if you have a joint collection of
 * separately retrieved contours. In order to solve this problem, you need to call [drawContours]
 * separately for each sub-group of contours, or iterate over the collection using contourIdx
 * parameter.
 *
 * @param image Destination image.
 *
 * @param contours All the input contours. Each contour is stored as a point vector.
 *
 * @param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are
 * drawn.
 *
 * @param color Color of the contours.
 *
 * @param thickness Thickness of lines the contours are drawn with. If it is negative (for example,
 * thickness=FILLED ), the contour interiors are drawn.
 *
 * @param lineType Line connectivity. See LineTypes
 *
 * @param hierarchy Optional information about hierarchy. It is only needed if you want to draw only
 * some of the contours (see maxLevel ).
 *
 * @param maxLevel Maximal level for drawn contours. If it is 0, only the specified contour is drawn.
 * If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
 * draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
 * parameter is only taken into account when there is hierarchy available.
 *
 * @param offset Optional contour shift parameter. Shift all the drawn contours by the specified
 * $\texttt{offset}=(dx,dy)$ .
 */
export declare function drawContours(
  image: InputOutputArray,
  contours: InputArrayOfArrays,
  contourIdx: int,
  color: any,
  thickness?: int,
  lineType?: int,
  hierarchy?: InputArray,
  maxLevel?: int,
  offset?: Point,
): void;

/**
 * The function [cv::drawMarker] draws a marker on a given position in the image. For the moment
 * several marker types are supported, see [MarkerTypes] for more information.
 *
 * @param img Image.
 *
 * @param position The point where the crosshair is positioned.
 *
 * @param color Line color.
 *
 * @param markerType The specific type of marker you want to use, see MarkerTypes
 *
 * @param markerSize The length of the marker axis [default = 20 pixels]
 *
 * @param thickness Line thickness.
 *
 * @param line_type Type of the line, See LineTypes
 */
export declare function drawMarker(
  img: InputOutputArray,
  position: Point,
  color: any,
  markerType?: int,
  markerSize?: int,
  thickness?: int,
  line_type?: int,
): void;

/**
 * The function [cv::ellipse] with more parameters draws an ellipse outline, a filled ellipse, an
 * elliptic arc, or a filled ellipse sector. The drawing code uses general parametric form. A
 * piecewise-linear curve is used to approximate the elliptic arc boundary. If you need more control of
 * the ellipse rendering, you can retrieve the curve using [ellipse2Poly] and then render it with
 * [polylines] or fill it with [fillPoly]. If you use the first variant of the function and want to
 * draw the whole ellipse, not an arc, pass `startAngle=0` and `endAngle=360`. If `startAngle` is
 * greater than `endAngle`, they are swapped. The figure below explains the meaning of the parameters
 * to draw the blue arc.
 *
 * @param img Image.
 *
 * @param center Center of the ellipse.
 *
 * @param axes Half of the size of the ellipse main axes.
 *
 * @param angle Ellipse rotation angle in degrees.
 *
 * @param startAngle Starting angle of the elliptic arc in degrees.
 *
 * @param endAngle Ending angle of the elliptic arc in degrees.
 *
 * @param color Ellipse color.
 *
 * @param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a
 * filled ellipse sector is to be drawn.
 *
 * @param lineType Type of the ellipse boundary. See LineTypes
 *
 * @param shift Number of fractional bits in the coordinates of the center and values of axes.
 */
export declare function ellipse(
  img: InputOutputArray,
  center: Point,
  axes: Size,
  angle: double,
  startAngle: double,
  endAngle: double,
  color: any,
  thickness?: int,
  lineType?: int,
  shift?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param img Image.
 *
 * @param box Alternative ellipse representation via RotatedRect. This means that the function draws an
 * ellipse inscribed in the rotated rectangle.
 *
 * @param color Ellipse color.
 *
 * @param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a
 * filled ellipse sector is to be drawn.
 *
 * @param lineType Type of the ellipse boundary. See LineTypes
 */
export declare function ellipse(
  img: InputOutputArray,
  box: any,
  color: any,
  thickness?: int,
  lineType?: int,
): void;

/**
 * The function ellipse2Poly computes the vertices of a polyline that approximates the specified
 * elliptic arc. It is used by [ellipse]. If `arcStart` is greater than `arcEnd`, they are swapped.
 *
 * @param center Center of the arc.
 *
 * @param axes Half of the size of the ellipse main axes. See ellipse for details.
 *
 * @param angle Rotation angle of the ellipse in degrees. See ellipse for details.
 *
 * @param arcStart Starting angle of the elliptic arc in degrees.
 *
 * @param arcEnd Ending angle of the elliptic arc in degrees.
 *
 * @param delta Angle between the subsequent polyline vertices. It defines the approximation accuracy.
 *
 * @param pts Output vector of polyline vertices.
 */
export declare function ellipse2Poly(
  center: Point,
  axes: Size,
  angle: int,
  arcStart: int,
  arcEnd: int,
  delta: int,
  pts: any,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param center Center of the arc.
 *
 * @param axes Half of the size of the ellipse main axes. See ellipse for details.
 *
 * @param angle Rotation angle of the ellipse in degrees. See ellipse for details.
 *
 * @param arcStart Starting angle of the elliptic arc in degrees.
 *
 * @param arcEnd Ending angle of the elliptic arc in degrees.
 *
 * @param delta Angle between the subsequent polyline vertices. It defines the approximation accuracy.
 *
 * @param pts Output vector of polyline vertices.
 */
export declare function ellipse2Poly(
  center: Point2d,
  axes: Size2d,
  angle: int,
  arcStart: int,
  arcEnd: int,
  delta: int,
  pts: any,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function fillConvexPoly(
  img: InputOutputArray,
  pts: any,
  npts: int,
  color: any,
  lineType?: int,
  shift?: int,
): void;

/**
 * The function [cv::fillConvexPoly] draws a filled convex polygon. This function is much faster than
 * the function [fillPoly] . It can fill not only convex polygons but any monotonic polygon without
 * self-intersections, that is, a polygon whose contour intersects every horizontal line (scan line)
 * twice at the most (though, its top-most and/or the bottom edge could be horizontal).
 *
 * @param img Image.
 *
 * @param points Polygon vertices.
 *
 * @param color Polygon color.
 *
 * @param lineType Type of the polygon boundaries. See LineTypes
 *
 * @param shift Number of fractional bits in the vertex coordinates.
 */
export declare function fillConvexPoly(
  img: InputOutputArray,
  points: InputArray,
  color: any,
  lineType?: int,
  shift?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function fillPoly(
  img: InputOutputArray,
  pts: any,
  npts: any,
  ncontours: int,
  color: any,
  lineType?: int,
  shift?: int,
  offset?: Point,
): void;

/**
 * The function [cv::fillPoly] fills an area bounded by several polygonal contours. The function can
 * fill complex areas, for example, areas with holes, contours with self-intersections (some of their
 * parts), and so forth.
 *
 * @param img Image.
 *
 * @param pts Array of polygons where each polygon is represented as an array of points.
 *
 * @param color Polygon color.
 *
 * @param lineType Type of the polygon boundaries. See LineTypes
 *
 * @param shift Number of fractional bits in the vertex coordinates.
 *
 * @param offset Optional offset of all points of the contours.
 */
export declare function fillPoly(
  img: InputOutputArray,
  pts: InputArrayOfArrays,
  color: any,
  lineType?: int,
  shift?: int,
  offset?: Point,
): void;

/**
 * The fontSize to use for [cv::putText]
 *
 * [cv::putText]
 *
 * @param fontFace Font to use, see cv::HersheyFonts.
 *
 * @param pixelHeight Pixel height to compute the fontScale for
 *
 * @param thickness Thickness of lines used to render the text.See putText for details.
 */
export declare function getFontScaleFromHeight(
  fontFace: any,
  pixelHeight: any,
  thickness?: any,
): double;

/**
 * The function [cv::getTextSize] calculates and returns the size of a box that contains the specified
 * text. That is, the following code renders some text, the tight box surrounding it, and the baseline:
 * :
 *
 * ```cpp
 * String text = "Funny text inside the box";
 * int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
 * double fontScale = 2;
 * int thickness = 3;
 *
 * Mat img(600, 800, CV_8UC3, Scalar::all(0));
 *
 * int baseline=0;
 * Size textSize = getTextSize(text, fontFace,
 *                             fontScale, thickness, &baseline);
 * baseline += thickness;
 *
 * // center the text
 * Point textOrg((img.cols - textSize.width)/2,
 *               (img.rows + textSize.height)/2);
 *
 * // draw the box
 * rectangle(img, textOrg + Point(0, baseline),
 *           textOrg + Point(textSize.width, -textSize.height),
 *           Scalar(0,0,255));
 * // ... and the baseline first
 * line(img, textOrg + Point(0, thickness),
 *      textOrg + Point(textSize.width, thickness),
 *      Scalar(0, 0, 255));
 *
 * // then put the text itself
 * putText(img, text, textOrg, fontFace, fontScale,
 *         Scalar::all(255), thickness, 8);
 * ```
 *
 * The size of a box that contains the specified text.
 *
 * [putText]
 *
 * @param text Input text string.
 *
 * @param fontFace Font to use, see HersheyFonts.
 *
 * @param fontScale Font scale factor that is multiplied by the font-specific base size.
 *
 * @param thickness Thickness of lines used to render the text. See putText for details.
 *
 * @param baseLine y-coordinate of the baseline relative to the bottom-most text point.
 */
export declare function getTextSize(
  text: any,
  fontFace: int,
  fontScale: double,
  thickness: int,
  baseLine: any,
): Size;

/**
 * The function line draws the line segment between pt1 and pt2 points in the image. The line is
 * clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected
 * or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased
 * lines are drawn using Gaussian filtering.
 *
 * @param img Image.
 *
 * @param pt1 First point of the line segment.
 *
 * @param pt2 Second point of the line segment.
 *
 * @param color Line color.
 *
 * @param thickness Line thickness.
 *
 * @param lineType Type of the line. See LineTypes.
 *
 * @param shift Number of fractional bits in the point coordinates.
 */
export declare function line(
  img: InputOutputArray,
  pt1: Point,
  pt2: Point,
  color: any,
  thickness?: int,
  lineType?: int,
  shift?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function polylines(
  img: InputOutputArray,
  pts: any,
  npts: any,
  ncontours: int,
  isClosed: bool,
  color: any,
  thickness?: int,
  lineType?: int,
  shift?: int,
): void;

/**
 * The function [cv::polylines] draws one or more polygonal curves.
 *
 * @param img Image.
 *
 * @param pts Array of polygonal curves.
 *
 * @param isClosed Flag indicating whether the drawn polylines are closed or not. If they are closed,
 * the function draws a line from the last vertex of each curve to its first vertex.
 *
 * @param color Polyline color.
 *
 * @param thickness Thickness of the polyline edges.
 *
 * @param lineType Type of the line segments. See LineTypes
 *
 * @param shift Number of fractional bits in the vertex coordinates.
 */
export declare function polylines(
  img: InputOutputArray,
  pts: InputArrayOfArrays,
  isClosed: bool,
  color: any,
  thickness?: int,
  lineType?: int,
  shift?: int,
): void;

/**
 * The function [cv::putText] renders the specified text string in the image. Symbols that cannot be
 * rendered using the specified font are replaced by question marks. See [getTextSize] for a text
 * rendering code example.
 *
 * @param img Image.
 *
 * @param text Text string to be drawn.
 *
 * @param org Bottom-left corner of the text string in the image.
 *
 * @param fontFace Font type, see HersheyFonts.
 *
 * @param fontScale Font scale factor that is multiplied by the font-specific base size.
 *
 * @param color Text color.
 *
 * @param thickness Thickness of the lines used to draw a text.
 *
 * @param lineType Line type. See LineTypes
 *
 * @param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise, it
 * is at the top-left corner.
 */
export declare function putText(
  img: InputOutputArray,
  text: any,
  org: Point,
  fontFace: int,
  fontScale: double,
  color: Scalar,
  thickness?: int,
  lineType?: int,
  bottomLeftOrigin?: bool,
): void;

/**
 * The function [cv::rectangle] draws a rectangle outline or a filled rectangle whose two opposite
 * corners are pt1 and pt2.
 *
 * @param img Image.
 *
 * @param pt1 Vertex of the rectangle.
 *
 * @param pt2 Vertex of the rectangle opposite to pt1 .
 *
 * @param color Rectangle color or brightness (grayscale image).
 *
 * @param thickness Thickness of lines that make up the rectangle. Negative values, like FILLED, mean
 * that the function has to draw a filled rectangle.
 *
 * @param lineType Type of the line. See LineTypes
 *
 * @param shift Number of fractional bits in the point coordinates.
 */
export declare function rectangle(
  img: InputOutputArray,
  pt1: Point,
  pt2: Point,
  color: any,
  thickness?: int,
  lineType?: int,
  shift?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * use `rec` parameter as alternative specification of the drawn rectangle: `r.tl() and
 * r.br()-Point(1,1)` are opposite corners
 */
export declare function rectangle(
  img: InputOutputArray,
  rec: Rect,
  color: any,
  thickness?: int,
  lineType?: int,
  shift?: int,
): void;

export declare const FONT_HERSHEY_SIMPLEX: HersheyFonts; // initializer: = 0

export declare const FONT_HERSHEY_PLAIN: HersheyFonts; // initializer: = 1

export declare const FONT_HERSHEY_DUPLEX: HersheyFonts; // initializer: = 2

export declare const FONT_HERSHEY_COMPLEX: HersheyFonts; // initializer: = 3

export declare const FONT_HERSHEY_TRIPLEX: HersheyFonts; // initializer: = 4

export declare const FONT_HERSHEY_COMPLEX_SMALL: HersheyFonts; // initializer: = 5

export declare const FONT_HERSHEY_SCRIPT_SIMPLEX: HersheyFonts; // initializer: = 6

export declare const FONT_HERSHEY_SCRIPT_COMPLEX: HersheyFonts; // initializer: = 7

export declare const FONT_ITALIC: HersheyFonts; // initializer: = 16

export declare const FILLED: LineTypes; // initializer: = -1

export declare const LINE_4: LineTypes; // initializer: = 4

export declare const LINE_8: LineTypes; // initializer: = 8

export declare const LINE_AA: LineTypes; // initializer: = 16

export declare const MARKER_CROSS: MarkerTypes; // initializer: = 0

export declare const MARKER_TILTED_CROSS: MarkerTypes; // initializer: = 1

export declare const MARKER_STAR: MarkerTypes; // initializer: = 2

export declare const MARKER_DIAMOND: MarkerTypes; // initializer: = 3

export declare const MARKER_SQUARE: MarkerTypes; // initializer: = 4

export declare const MARKER_TRIANGLE_UP: MarkerTypes; // initializer: = 5

export declare const MARKER_TRIANGLE_DOWN: MarkerTypes; // initializer: = 6

/**
 * Only a subset of Hershey fonts  are supported
 *
 */
export type HersheyFonts = any;

/**
 * Only a subset of Hershey fonts  are supported
 *
 */
export type LineTypes = any;

/**
 * Only a subset of Hershey fonts  are supported
 *
 */
export type MarkerTypes = any;
