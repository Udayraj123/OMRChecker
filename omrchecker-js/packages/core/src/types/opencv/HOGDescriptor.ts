import type {
  bool,
  double,
  FileNode,
  FileStorage,
  float,
  InputArray,
  InputOutputArray,
  int,
  Point,
  Size,
  size_t,
  UMat,
} from "./_types";

/**
 * the HOG descriptor algorithm introduced by Navneet Dalal and Bill Triggs Dalal2005 .
 *
 * useful links:
 *
 * Source:
 * [opencv2/objdetect.hpp](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/objdetect.hpp#L377).
 *
 */
export declare class HOGDescriptor {
  public blockSize: Size;

  public blockStride: Size;

  public cellSize: Size;

  public derivAperture: int;

  public free_coef: float;

  public gammaCorrection: bool;

  public histogramNormType: any;

  public L2HysThreshold: double;

  public nbins: int;

  public nlevels: int;

  public oclSvmDetector: UMat;

  public signedGradient: bool;

  public svmDetector: any;

  public winSigma: double;

  public winSize: Size;

  /**
   *   aqual to [HOGDescriptor](Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9 )
   */
  public constructor();

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param _winSize sets winSize with given value.
   *
   * @param _blockSize sets blockSize with given value.
   *
   * @param _blockStride sets blockStride with given value.
   *
   * @param _cellSize sets cellSize with given value.
   *
   * @param _nbins sets nbins with given value.
   *
   * @param _derivAperture sets derivAperture with given value.
   *
   * @param _winSigma sets winSigma with given value.
   *
   * @param _histogramNormType sets histogramNormType with given value.
   *
   * @param _L2HysThreshold sets L2HysThreshold with given value.
   *
   * @param _gammaCorrection sets gammaCorrection with given value.
   *
   * @param _nlevels sets nlevels with given value.
   *
   * @param _signedGradient sets signedGradient with given value.
   */
  public constructor(
    _winSize: Size,
    _blockSize: Size,
    _blockStride: Size,
    _cellSize: Size,
    _nbins: int,
    _derivAperture?: int,
    _winSigma?: double,
    _histogramNormType?: any,
    _L2HysThreshold?: double,
    _gammaCorrection?: bool,
    _nlevels?: int,
    _signedGradient?: bool,
  );

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param filename The file name containing HOGDescriptor properties and coefficients for the linear
   * SVM classifier.
   */
  public constructor(filename: String);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param d the HOGDescriptor which cloned to create a new one.
   */
  public constructor(d: HOGDescriptor);

  public checkDetectorSize(): bool;

  /**
   * @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
   *
   * @param descriptors Matrix of the type CV_32F
   *
   * @param winStride Window stride. It must be a multiple of block stride.
   *
   * @param padding Padding
   *
   * @param locations Vector of Point
   */
  public compute(
    img: InputArray,
    descriptors: any,
    winStride?: Size,
    padding?: Size,
    locations?: Point,
  ): InputArray;

  /**
   * @param img Matrix contains the image to be computed
   *
   * @param grad Matrix of type CV_32FC2 contains computed gradients
   *
   * @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
   *
   * @param paddingTL Padding from top-left
   *
   * @param paddingBR Padding from bottom-right
   */
  public computeGradient(
    img: InputArray,
    grad: InputOutputArray,
    angleOfs: InputOutputArray,
    paddingTL?: Size,
    paddingBR?: Size,
  ): InputArray;

  /**
   * @param c cloned HOGDescriptor
   */
  public copyTo(c: HOGDescriptor): HOGDescriptor;

  /**
   * @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
   *
   * @param foundLocations Vector of point where each point contains left-top corner point of detected
   * object boundaries.
   *
   * @param weights Vector that will contain confidence values for each detected object.
   *
   * @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
   * it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
   * the free coefficient is omitted (which is allowed), you can specify it manually here.
   *
   * @param winStride Window stride. It must be a multiple of block stride.
   *
   * @param padding Padding
   *
   * @param searchLocations Vector of Point includes set of requested locations to be evaluated.
   */
  public detect(
    img: InputArray,
    foundLocations: any,
    weights: any,
    hitThreshold?: double,
    winStride?: Size,
    padding?: Size,
    searchLocations?: Point,
  ): InputArray;

  /**
   * @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
   *
   * @param foundLocations Vector of point where each point contains left-top corner point of detected
   * object boundaries.
   *
   * @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
   * it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
   * the free coefficient is omitted (which is allowed), you can specify it manually here.
   *
   * @param winStride Window stride. It must be a multiple of block stride.
   *
   * @param padding Padding
   *
   * @param searchLocations Vector of Point includes locations to search.
   */
  public detect(
    img: InputArray,
    foundLocations: any,
    hitThreshold?: double,
    winStride?: Size,
    padding?: Size,
    searchLocations?: Point,
  ): InputArray;

  /**
   * @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
   *
   * @param foundLocations Vector of rectangles where each rectangle contains the detected object.
   *
   * @param foundWeights Vector that will contain confidence values for each detected object.
   *
   * @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
   * it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
   * the free coefficient is omitted (which is allowed), you can specify it manually here.
   *
   * @param winStride Window stride. It must be a multiple of block stride.
   *
   * @param padding Padding
   *
   * @param scale Coefficient of the detection window increase.
   *
   * @param finalThreshold Final threshold
   *
   * @param useMeanshiftGrouping indicates grouping algorithm
   */
  public detectMultiScale(
    img: InputArray,
    foundLocations: any,
    foundWeights: any,
    hitThreshold?: double,
    winStride?: Size,
    padding?: Size,
    scale?: double,
    finalThreshold?: double,
    useMeanshiftGrouping?: bool,
  ): InputArray;

  /**
   * @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
   *
   * @param foundLocations Vector of rectangles where each rectangle contains the detected object.
   *
   * @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
   * it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
   * the free coefficient is omitted (which is allowed), you can specify it manually here.
   *
   * @param winStride Window stride. It must be a multiple of block stride.
   *
   * @param padding Padding
   *
   * @param scale Coefficient of the detection window increase.
   *
   * @param finalThreshold Final threshold
   *
   * @param useMeanshiftGrouping indicates grouping algorithm
   */
  public detectMultiScale(
    img: InputArray,
    foundLocations: any,
    hitThreshold?: double,
    winStride?: Size,
    padding?: Size,
    scale?: double,
    finalThreshold?: double,
    useMeanshiftGrouping?: bool,
  ): InputArray;

  /**
   * @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
   *
   * @param foundLocations Vector of rectangles where each rectangle contains the detected object.
   *
   * @param locations Vector of DetectionROI
   *
   * @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
   * it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
   * the free coefficient is omitted (which is allowed), you can specify it manually here.
   *
   * @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a
   * group of rectangles to retain it.
   */
  public detectMultiScaleROI(
    img: InputArray,
    foundLocations: any,
    locations: any,
    hitThreshold?: double,
    groupThreshold?: int,
  ): InputArray;

  /**
   * @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
   *
   * @param locations Vector of Point
   *
   * @param foundLocations Vector of Point where each Point is detected object's top-left point.
   *
   * @param confidences confidences
   *
   * @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
   * it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
   * the free coefficient is omitted (which is allowed), you can specify it manually here
   *
   * @param winStride winStride
   *
   * @param padding padding
   */
  public detectROI(
    img: InputArray,
    locations: any,
    foundLocations: any,
    confidences: any,
    hitThreshold?: double,
    winStride?: any,
    padding?: any,
  ): InputArray;

  public getDescriptorSize(): size_t;

  public getWinSigma(): double;

  /**
   * @param rectList Input/output vector of rectangles. Output vector includes retained and grouped
   * rectangles. (The Python list is not modified in place.)
   *
   * @param weights Input/output vector of weights of rectangles. Output vector includes weights of
   * retained and grouped rectangles. (The Python list is not modified in place.)
   *
   * @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a
   * group of rectangles to retain it.
   *
   * @param eps Relative difference between sides of the rectangles to merge them into a group.
   */
  public groupRectangles(
    rectList: any,
    weights: any,
    groupThreshold: int,
    eps: double,
  ): any;

  /**
   * @param filename Path of the file to read.
   *
   * @param objname The optional name of the node to read (if empty, the first top-level node will be
   * used).
   */
  public load(filename: String, objname?: String): String;

  /**
   * @param fn File node
   */
  public read(fn: FileNode): FileNode;

  /**
   * @param filename File name
   *
   * @param objname Object name
   */
  public save(filename: String, objname?: String): String;

  /**
   * @param svmdetector coefficients for the linear SVM classifier.
   */
  public setSVMDetector(svmdetector: InputArray): InputArray;

  /**
   * @param fs File storage
   *
   * @param objname Object name
   */
  public write(fs: FileStorage, objname: String): FileStorage;

  public static getDaimlerPeopleDetector(): any;

  public static getDefaultPeopleDetector(): any;
}

export declare const DEFAULT_NLEVELS: any; // initializer: = 64

export declare const DESCR_FORMAT_COL_BY_COL: DescriptorStorageFormat; // initializer:

export declare const DESCR_FORMAT_ROW_BY_ROW: DescriptorStorageFormat; // initializer:

export declare const L2Hys: HistogramNormType; // initializer: = 0

export type DescriptorStorageFormat = any;

export type HistogramNormType = any;
