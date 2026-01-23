import type {
  double,
  InputArray,
  InputOutputArray,
  int,
  OutputArray,
  Point,
  Rect,
  Scalar,
} from "./_types";
/*
 * # Miscellaneous Image Transformations
 *
 */
/**
 * The function transforms a grayscale image to a binary image according to the formulae:
 *
 * **THRESH_BINARY** `\\[dst(x,y) = \\fork{\\texttt{maxValue}}{if \\(src(x,y) >
 * T(x,y)\\)}{0}{otherwise}\\]`
 * **THRESH_BINARY_INV** `\\[dst(x,y) = \\fork{0}{if \\(src(x,y) >
 * T(x,y)\\)}{\\texttt{maxValue}}{otherwise}\\]` where `$T(x,y)$` is a threshold calculated
 * individually for each pixel (see adaptiveMethod parameter).
 *
 * The function can process the image in-place.
 *
 * [threshold], [blur], [GaussianBlur]
 *
 * @param src Source 8-bit single-channel image.
 *
 * @param dst Destination image of the same size and the same type as src.
 *
 * @param maxValue Non-zero value assigned to the pixels for which the condition is satisfied
 *
 * @param adaptiveMethod Adaptive thresholding algorithm to use, see AdaptiveThresholdTypes. The
 * BORDER_REPLICATE | BORDER_ISOLATED is used to process boundaries.
 *
 * @param thresholdType Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV, see
 * ThresholdTypes.
 *
 * @param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the
 * pixel: 3, 5, 7, and so on.
 *
 * @param C Constant subtracted from the mean or weighted mean (see the details below). Normally, it is
 * positive but may be zero or negative as well.
 */
export declare function adaptiveThreshold(
  src: InputArray,
  dst: OutputArray,
  maxValue: double,
  adaptiveMethod: int,
  thresholdType: int,
  blockSize: int,
  C: double,
): void;

/**
 * Performs linear blending of two images: `\\[ \\texttt{dst}(i,j) =
 * \\texttt{weights1}(i,j)*\\texttt{src1}(i,j) + \\texttt{weights2}(i,j)*\\texttt{src2}(i,j) \\]`
 *
 * @param src1 It has a type of CV_8UC(n) or CV_32FC(n), where n is a positive integer.
 *
 * @param src2 It has the same type and size as src1.
 *
 * @param weights1 It has a type of CV_32FC1 and the same size with src1.
 *
 * @param weights2 It has a type of CV_32FC1 and the same size with src1.
 *
 * @param dst It is created if it does not have the same size and type with src1.
 */
export declare function blendLinear(
  src1: InputArray,
  src2: InputArray,
  weights1: InputArray,
  weights2: InputArray,
  dst: OutputArray,
): void;

/**
 * The function [cv::distanceTransform] calculates the approximate or precise distance from every
 * binary image pixel to the nearest zero pixel. For zero image pixels, the distance will obviously be
 * zero.
 *
 * When maskSize == [DIST_MASK_PRECISE] and distanceType == [DIST_L2] , the function runs the algorithm
 * described in Felzenszwalb04 . This algorithm is parallelized with the TBB library.
 *
 * In other cases, the algorithm Borgefors86 is used. This means that for a pixel the function finds
 * the shortest path to the nearest zero pixel consisting of basic shifts: horizontal, vertical,
 * diagonal, or knight's move (the latest is available for a `$5\\times 5$` mask). The overall distance
 * is calculated as a sum of these basic distances. Since the distance function should be symmetric,
 * all of the horizontal and vertical shifts must have the same cost (denoted as a ), all the diagonal
 * shifts must have the same cost (denoted as `b`), and all knight's moves must have the same cost
 * (denoted as `c`). For the [DIST_C] and [DIST_L1] types, the distance is calculated precisely,
 * whereas for [DIST_L2] (Euclidean distance) the distance can be calculated only with a relative error
 * (a `$5\\times 5$` mask gives more accurate results). For `a`,`b`, and `c`, OpenCV uses the values
 * suggested in the original paper:
 *
 * DIST_L1: `a = 1, b = 2`
 * DIST_L2:
 *
 * `3 x 3`: `a=0.955, b=1.3693`
 * `5 x 5`: `a=1, b=1.4, c=2.1969`
 *
 * DIST_C: `a = 1, b = 1`
 *
 * Typically, for a fast, coarse distance estimation [DIST_L2], a `$3\\times 3$` mask is used. For a
 * more accurate distance estimation [DIST_L2], a `$5\\times 5$` mask or the precise algorithm is used.
 * Note that both the precise and the approximate algorithms are linear on the number of pixels.
 *
 * This variant of the function does not only compute the minimum distance for each pixel `$(x, y)$`
 * but also identifies the nearest connected component consisting of zero pixels
 * (labelType==[DIST_LABEL_CCOMP]) or the nearest zero pixel (labelType==[DIST_LABEL_PIXEL]). Index of
 * the component/pixel is stored in `labels(x, y)`. When labelType==[DIST_LABEL_CCOMP], the function
 * automatically finds connected components of zero pixels in the input image and marks them with
 * distinct labels. When labelType==[DIST_LABEL_CCOMP], the function scans through the input image and
 * marks all the zero pixels with distinct labels.
 *
 * In this mode, the complexity is still linear. That is, the function provides a very fast way to
 * compute the Voronoi diagram for a binary image. Currently, the second variant can use only the
 * approximate distance transform algorithm, i.e. maskSize=[DIST_MASK_PRECISE] is not supported yet.
 *
 * @param src 8-bit, single-channel (binary) source image.
 *
 * @param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
 * single-channel image of the same size as src.
 *
 * @param labels Output 2D array of labels (the discrete Voronoi diagram). It has the type CV_32SC1 and
 * the same size as src.
 *
 * @param distanceType Type of distance, see DistanceTypes
 *
 * @param maskSize Size of the distance transform mask, see DistanceTransformMasks. DIST_MASK_PRECISE
 * is not supported by this variant. In case of the DIST_L1 or DIST_C distance type, the parameter is
 * forced to 3 because a $3\times 3$ mask gives the same result as $5\times 5$ or any larger aperture.
 *
 * @param labelType Type of the label array to build, see DistanceTransformLabelTypes.
 */
export declare function distanceTransform(
  src: InputArray,
  dst: OutputArray,
  labels: OutputArray,
  distanceType: int,
  maskSize: int,
  labelType?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * @param src 8-bit, single-channel (binary) source image.
 *
 * @param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
 * single-channel image of the same size as src .
 *
 * @param distanceType Type of distance, see DistanceTypes
 *
 * @param maskSize Size of the distance transform mask, see DistanceTransformMasks. In case of the
 * DIST_L1 or DIST_C distance type, the parameter is forced to 3 because a $3\times 3$ mask gives the
 * same result as $5\times 5$ or any larger aperture.
 *
 * @param dstType Type of output image. It can be CV_8U or CV_32F. Type CV_8U can be used only for the
 * first variant of the function and distanceType == DIST_L1.
 */
export declare function distanceTransform(
  src: InputArray,
  dst: OutputArray,
  distanceType: int,
  maskSize: int,
  dstType?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 *
 * variant without `mask` parameter
 */
export declare function floodFill(
  image: InputOutputArray,
  seedPoint: Point,
  newVal: Scalar,
  rect?: any,
  loDiff?: Scalar,
  upDiff?: Scalar,
  flags?: int,
): int;

/**
 * The function [cv::floodFill] fills a connected component starting from the seed point with the
 * specified color. The connectivity is determined by the color/brightness closeness of the neighbor
 * pixels. The pixel at `$(x,y)$` is considered to belong to the repainted domain if:
 *
 * in case of a grayscale image and floating range `\\[\\texttt{src} (x',y')- \\texttt{loDiff} \\leq
 * \\texttt{src} (x,y) \\leq \\texttt{src} (x',y')+ \\texttt{upDiff}\\]`
 * in case of a grayscale image and fixed range `\\[\\texttt{src} ( \\texttt{seedPoint} .x,
 * \\texttt{seedPoint} .y)- \\texttt{loDiff} \\leq \\texttt{src} (x,y) \\leq \\texttt{src} (
 * \\texttt{seedPoint} .x, \\texttt{seedPoint} .y)+ \\texttt{upDiff}\\]`
 * in case of a color image and floating range `\\[\\texttt{src} (x',y')_r- \\texttt{loDiff} _r \\leq
 * \\texttt{src} (x,y)_r \\leq \\texttt{src} (x',y')_r+ \\texttt{upDiff} _r,\\]` `\\[\\texttt{src}
 * (x',y')_g- \\texttt{loDiff} _g \\leq \\texttt{src} (x,y)_g \\leq \\texttt{src} (x',y')_g+
 * \\texttt{upDiff} _g\\]` and `\\[\\texttt{src} (x',y')_b- \\texttt{loDiff} _b \\leq \\texttt{src}
 * (x,y)_b \\leq \\texttt{src} (x',y')_b+ \\texttt{upDiff} _b\\]`
 * in case of a color image and fixed range `\\[\\texttt{src} ( \\texttt{seedPoint} .x,
 * \\texttt{seedPoint} .y)_r- \\texttt{loDiff} _r \\leq \\texttt{src} (x,y)_r \\leq \\texttt{src} (
 * \\texttt{seedPoint} .x, \\texttt{seedPoint} .y)_r+ \\texttt{upDiff} _r,\\]` `\\[\\texttt{src} (
 * \\texttt{seedPoint} .x, \\texttt{seedPoint} .y)_g- \\texttt{loDiff} _g \\leq \\texttt{src} (x,y)_g
 * \\leq \\texttt{src} ( \\texttt{seedPoint} .x, \\texttt{seedPoint} .y)_g+ \\texttt{upDiff} _g\\]` and
 * `\\[\\texttt{src} ( \\texttt{seedPoint} .x, \\texttt{seedPoint} .y)_b- \\texttt{loDiff} _b \\leq
 * \\texttt{src} (x,y)_b \\leq \\texttt{src} ( \\texttt{seedPoint} .x, \\texttt{seedPoint} .y)_b+
 * \\texttt{upDiff} _b\\]`
 *
 * where `$src(x',y')$` is the value of one of pixel neighbors that is already known to belong to the
 * component. That is, to be added to the connected component, a color/brightness of the pixel should
 * be close enough to:
 *
 * Color/brightness of one of its neighbors that already belong to the connected component in case of a
 * floating range.
 * Color/brightness of the seed point in case of a fixed range.
 *
 * Use these functions to either mark a connected component with the specified color in-place, or build
 * a mask and then extract the contour, or copy the region to another image, and so on.
 *
 * Since the mask is larger than the filled image, a pixel `$(x, y)$` in image corresponds to the pixel
 * `$(x+1, y+1)$` in the mask .
 *
 * [findContours]
 *
 * @param image Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by the
 * function unless the FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See the
 * details below.
 *
 * @param mask Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixels
 * taller than image. Since this is both an input and output parameter, you must take responsibility of
 * initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example, an
 * edge detector output can be used as a mask to stop filling at edges. On output, pixels in the mask
 * corresponding to filled pixels in the image are set to 1 or to the a value specified in flags as
 * described below. Additionally, the function fills the border of the mask with ones to simplify
 * internal processing. It is therefore possible to use the same mask in multiple calls to the function
 * to make sure the filled areas do not overlap.
 *
 * @param seedPoint Starting point.
 *
 * @param newVal New value of the repainted domain pixels.
 *
 * @param rect Optional output parameter set by the function to the minimum bounding rectangle of the
 * repainted domain.
 *
 * @param loDiff Maximal lower brightness/color difference between the currently observed pixel and one
 * of its neighbors belonging to the component, or a seed pixel being added to the component.
 *
 * @param upDiff Maximal upper brightness/color difference between the currently observed pixel and one
 * of its neighbors belonging to the component, or a seed pixel being added to the component.
 *
 * @param flags Operation flags. The first 8 bits contain a connectivity value. The default value of 4
 * means that only the four nearest neighbor pixels (those that share an edge) are considered. A
 * connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner)
 * will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill the
 * mask (the default value is 1). For example, 4 | ( 255 << 8 ) will consider 4 nearest neighbours and
 * fill the mask with a value of 255. The following additional options occupy higher bits and therefore
 * may be further combined with the connectivity and mask fill values using bit-wise or (|), see
 * FloodFillFlags.
 */
export declare function floodFill(
  image: InputOutputArray,
  mask: InputOutputArray,
  seedPoint: Point,
  newVal: Scalar,
  rect?: any,
  loDiff?: Scalar,
  upDiff?: Scalar,
  flags?: int,
): int;

/**
 * The function implements the .
 *
 * @param img Input 8-bit 3-channel image.
 *
 * @param mask Input/output 8-bit single-channel mask. The mask is initialized by the function when
 * mode is set to GC_INIT_WITH_RECT. Its elements may have one of the GrabCutClasses.
 *
 * @param rect ROI containing a segmented object. The pixels outside of the ROI are marked as "obvious
 * background". The parameter is only used when mode==GC_INIT_WITH_RECT .
 *
 * @param bgdModel Temporary array for the background model. Do not modify it while you are processing
 * the same image.
 *
 * @param fgdModel Temporary arrays for the foreground model. Do not modify it while you are processing
 * the same image.
 *
 * @param iterCount Number of iterations the algorithm should make before returning the result. Note
 * that the result can be refined with further calls with mode==GC_INIT_WITH_MASK or mode==GC_EVAL .
 *
 * @param mode Operation mode that could be one of the GrabCutModes
 */
export declare function grabCut(
  img: InputArray,
  mask: InputOutputArray,
  rect: Rect,
  bgdModel: InputOutputArray,
  fgdModel: InputOutputArray,
  iterCount: int,
  mode?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function integral(
  src: InputArray,
  sum: OutputArray,
  sdepth?: int,
): void;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function integral(
  src: InputArray,
  sum: OutputArray,
  sqsum: OutputArray,
  sdepth?: int,
  sqdepth?: int,
): void;

/**
 * The function calculates one or more integral images for the source image as follows:
 *
 * `\\[\\texttt{sum} (X,Y) = \\sum _{x<X,y<Y} \\texttt{image} (x,y)\\]`
 *
 * `\\[\\texttt{sqsum} (X,Y) = \\sum _{x<X,y<Y} \\texttt{image} (x,y)^2\\]`
 *
 * `\\[\\texttt{tilted} (X,Y) = \\sum _{y<Y,abs(x-X+1) \\leq Y-y-1} \\texttt{image} (x,y)\\]`
 *
 * Using these integral images, you can calculate sum, mean, and standard deviation over a specific
 * up-right or rotated rectangular region of the image in a constant time, for example:
 *
 * `\\[\\sum _{x_1 \\leq x < x_2, \\, y_1 \\leq y < y_2} \\texttt{image} (x,y) = \\texttt{sum}
 * (x_2,y_2)- \\texttt{sum} (x_1,y_2)- \\texttt{sum} (x_2,y_1)+ \\texttt{sum} (x_1,y_1)\\]`
 *
 * It makes possible to do a fast blurring or fast block correlation with a variable window size, for
 * example. In case of multi-channel images, sums for each channel are accumulated independently.
 *
 * As a practical example, the next figure shows the calculation of the integral of a straight
 * rectangle Rect(3,3,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the
 * original image are shown, as well as the relative pixels in the integral images sum and tilted .
 *
 * @param src input image as $W \times H$, 8-bit or floating-point (32f or 64f).
 *
 * @param sum integral image as $(W+1)\times (H+1)$ , 32-bit integer or floating-point (32f or 64f).
 *
 * @param sqsum integral image for squared pixel values; it is $(W+1)\times (H+1)$, double-precision
 * floating-point (64f) array.
 *
 * @param tilted integral for the image rotated by 45 degrees; it is $(W+1)\times (H+1)$ array with the
 * same data type as sum.
 *
 * @param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or
 * CV_64F.
 *
 * @param sqdepth desired depth of the integral image of squared pixel values, CV_32F or CV_64F.
 */
export declare function integral(
  src: InputArray,
  sum: OutputArray,
  sqsum: OutputArray,
  tilted: OutputArray,
  sdepth?: int,
  sqdepth?: int,
): void;

/**
 * The function applies fixed-level thresholding to a multiple-channel array. The function is typically
 * used to get a bi-level (binary) image out of a grayscale image ( [compare] could be also used for
 * this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
 * values. There are several types of thresholding supported by the function. They are determined by
 * type parameter.
 *
 * Also, the special values [THRESH_OTSU] or [THRESH_TRIANGLE] may be combined with one of the above
 * values. In these cases, the function determines the optimal threshold value using the Otsu's or
 * Triangle algorithm and uses it instead of the specified thresh.
 *
 * Currently, the Otsu's and Triangle methods are implemented only for 8-bit single-channel images.
 *
 * the computed threshold value if Otsu's or Triangle methods used.
 *
 * [adaptiveThreshold], [findContours], [compare], [min], [max]
 *
 * @param src input array (multiple-channel, 8-bit or 32-bit floating point).
 *
 * @param dst output array of the same size and type and the same number of channels as src.
 *
 * @param thresh threshold value.
 *
 * @param maxval maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
 *
 * @param type thresholding type (see ThresholdTypes).
 */
export declare function threshold(
  src: InputArray,
  dst: OutputArray,
  thresh: double,
  maxval: double,
  type: int,
): double;

/**
 * The function implements one of the variants of watershed, non-parametric marker-based segmentation
 * algorithm, described in Meyer92 .
 *
 * Before passing the image to the function, you have to roughly outline the desired regions in the
 * image markers with positive (>0) indices. So, every region is represented as one or more connected
 * components with the pixel values 1, 2, 3, and so on. Such markers can be retrieved from a binary
 * mask using [findContours] and [drawContours] (see the watershed.cpp demo). The markers are "seeds"
 * of the future image regions. All the other pixels in markers , whose relation to the outlined
 * regions is not known and should be defined by the algorithm, should be set to 0's. In the function
 * output, each pixel in markers is set to a value of the "seed" components or to -1 at boundaries
 * between the regions.
 *
 * Any two neighbor connected components are not necessarily separated by a watershed boundary (-1's
 * pixels); for example, they can touch each other in the initial marker image passed to the function.
 *
 * [findContours]
 *
 * @param image Input 8-bit 3-channel image.
 *
 * @param markers Input/output 32-bit single-channel image (map) of markers. It should have the same
 * size as image .
 */
export declare function watershed(
  image: InputArray,
  markers: InputOutputArray,
): void;

/**
 * the threshold value `$T(x,y)$` is a mean of the `$\\texttt{blockSize} \\times \\texttt{blockSize}$`
 * neighborhood of `$(x, y)$` minus C
 *
 */
export declare const ADAPTIVE_THRESH_MEAN_C: AdaptiveThresholdTypes; // initializer: = 0

/**
 * the threshold value `$T(x, y)$` is a weighted sum (cross-correlation with a Gaussian window) of the
 * `$\\texttt{blockSize} \\times \\texttt{blockSize}$` neighborhood of `$(x, y)$` minus C . The default
 * sigma (standard deviation) is used for the specified blockSize . See [getGaussianKernel]
 *
 */
export declare const ADAPTIVE_THRESH_GAUSSIAN_C: AdaptiveThresholdTypes; // initializer: = 1

/**
 * each connected component of zeros in src (as well as all the non-zero pixels closest to the
 * connected component) will be assigned the same label
 *
 */
export declare const DIST_LABEL_CCOMP: DistanceTransformLabelTypes; // initializer: = 0

/**
 * each zero pixel (and all the non-zero pixels closest to it) gets its own label.
 *
 */
export declare const DIST_LABEL_PIXEL: DistanceTransformLabelTypes; // initializer: = 1

export declare const DIST_MASK_3: DistanceTransformMasks; // initializer: = 3

export declare const DIST_MASK_5: DistanceTransformMasks; // initializer: = 5

export declare const DIST_MASK_PRECISE: DistanceTransformMasks; // initializer: = 0

export declare const DIST_USER: DistanceTypes; // initializer: = -1

export declare const DIST_L1: DistanceTypes; // initializer: = 1

export declare const DIST_L2: DistanceTypes; // initializer: = 2

export declare const DIST_C: DistanceTypes; // initializer: = 3

export declare const DIST_L12: DistanceTypes; // initializer: = 4

export declare const DIST_FAIR: DistanceTypes; // initializer: = 5

export declare const DIST_WELSCH: DistanceTypes; // initializer: = 6

export declare const DIST_HUBER: DistanceTypes; // initializer: = 7

/**
 * If set, the difference between the current pixel and seed pixel is considered. Otherwise, the
 * difference between neighbor pixels is considered (that is, the range is floating).
 *
 */
export declare const FLOODFILL_FIXED_RANGE: FloodFillFlags; // initializer: = 1 << 16

/**
 * If set, the function does not change the image ( newVal is ignored), and only fills the mask with
 * the value specified in bits 8-16 of flags as described above. This option only make sense in
 * function variants that have the mask parameter.
 *
 */
export declare const FLOODFILL_MASK_ONLY: FloodFillFlags; // initializer: = 1 << 17

export declare const GC_BGD: GrabCutClasses; // initializer: = 0

export declare const GC_FGD: GrabCutClasses; // initializer: = 1

export declare const GC_PR_BGD: GrabCutClasses; // initializer: = 2

export declare const GC_PR_FGD: GrabCutClasses; // initializer: = 3

/**
 * The function initializes the state and the mask using the provided rectangle. After that it runs
 * iterCount iterations of the algorithm.
 *
 */
export declare const GC_INIT_WITH_RECT: GrabCutModes; // initializer: = 0

/**
 * The function initializes the state using the provided mask. Note that GC_INIT_WITH_RECT and
 * GC_INIT_WITH_MASK can be combined. Then, all the pixels outside of the ROI are automatically
 * initialized with GC_BGD .
 *
 */
export declare const GC_INIT_WITH_MASK: GrabCutModes; // initializer: = 1

/**
 * The value means that the algorithm should just resume.
 *
 */
export declare const GC_EVAL: GrabCutModes; // initializer: = 2

/**
 * The value means that the algorithm should just run the grabCut algorithm (a single iteration) with
 * the fixed model
 *
 */
export declare const GC_EVAL_FREEZE_MODEL: GrabCutModes; // initializer: = 3

export declare const THRESH_BINARY: ThresholdTypes; // initializer: = 0

export declare const THRESH_BINARY_INV: ThresholdTypes; // initializer: = 1

export declare const THRESH_TRUNC: ThresholdTypes; // initializer: = 2

export declare const THRESH_TOZERO: ThresholdTypes; // initializer: = 3

export declare const THRESH_TOZERO_INV: ThresholdTypes; // initializer: = 4

export declare const THRESH_MASK: ThresholdTypes; // initializer: = 7

export declare const THRESH_OTSU: ThresholdTypes; // initializer: = 8

export declare const THRESH_TRIANGLE: ThresholdTypes; // initializer: = 16

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type AdaptiveThresholdTypes = any;

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type DistanceTransformLabelTypes = any;

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type DistanceTransformMasks = any;

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type DistanceTypes = any;

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type FloodFillFlags = any;

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type GrabCutClasses = any;

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type GrabCutModes = any;

/**
 * adaptive threshold algorithm
 *
 * [adaptiveThreshold]
 *
 */
export type ThresholdTypes = any;
