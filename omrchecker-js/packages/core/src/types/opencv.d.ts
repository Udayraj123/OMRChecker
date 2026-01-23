/**
 * Minimal TypeScript declarations for OpenCV.js
 *
 * Contains only types actually used in the codebase plus commonly used types for future development.
 */

declare namespace cv {
  // ==================== Core Classes ====================

  interface EmscriptenEmbindInstance {
    delete(): void;
    isDeleted(): boolean;
  }

  class Mat implements EmscriptenEmbindInstance {
    constructor();
    constructor(rows: number, cols: number, type: number);
    constructor(rows: number, cols: number, type: number, scalar: Scalar);
    constructor(size: Size, type: number);
    constructor(size: Size, type: number, scalar: Scalar);
    constructor(mat: Mat, roi: Rect);

    delete(): void;
    isDeleted(): boolean;
    clone(): Mat;
    copyTo(dst: Mat, mask?: Mat): void;
    convertTo(dst: Mat, rtype: number, alpha?: number, beta?: number): void;
    setTo(value: Scalar, mask?: Mat): Mat;

    roi(rect: Rect): Mat;

    rows: number;
    cols: number;
    data: Uint8Array;
    data8S: Int8Array;
    data16U: Uint16Array;
    data16S: Int16Array;
    data32S: Int32Array;
    data32F: Float32Array;
    data64F: Float64Array;

    channels(): number;
    type(): number;
    depth(): number;
    empty(): boolean;
    size(): Size;
    total(): number;
    elemSize(): number;
    elemSize1(): number;
    step: number[];

    ptr(row: number, col?: number): number;
    ucharPtr(row: number, col?: number): Uint8Array;
    shortPtr(row: number, col?: number): Int16Array;
    intPtr(row: number, col?: number): Int32Array;
    floatPtr(row: number, col?: number): Float32Array;
    doublePtr(row: number, col?: number): Float64Array;
    intAt(row: number, col: number): number;
    intAt(index: number[]): number;

    static zeros(rows: number, cols: number, type: number): Mat;
    static ones(rows: number, cols: number, type: number): Mat;
    static eye(rows: number, cols: number, type: number): Mat;
  }

  class Scalar {
    constructor(v0?: number, v1?: number, v2?: number, v3?: number);
    static all(v: number): Scalar;
    0: number;
    1: number;
    2: number;
    3?: number;
  }

  class Point {
    constructor(x?: number, y?: number);
    x: number;
    y: number;
  }

  class Point2f {
    constructor(x?: number, y?: number);
    x: number;
    y: number;
  }

  class Vec3 {
    constructor(x?: number, y?: number, z?: number);
    0: number;
    1: number;
    2: number;
  }

  class Size {
    constructor(width?: number, height?: number);
    width: number;
    height: number;
  }

  class Size2f {
    constructor(width?: number, height?: number);
    width: number;
    height: number;
  }

  class Rect {
    constructor(x?: number, y?: number, width?: number, height?: number);
    x: number;
    y: number;
    width: number;
    height: number;
  }

  class Rect2f {
    constructor(x?: number, y?: number, width?: number, height?: number);
    x: number;
    y: number;
    width: number;
    height: number;
  }

  class RotatedRect {
    constructor();
    constructor(center: Point2f, size: Size2f, angle: number);
    center: Point2f;
    size: Size2f;
    angle: number;
    points(): Point2f[];
    boundingRect(): Rect;
    boundingRect2f(): Rect2f;
    static points(rect: RotatedRect): Point2f[];
  }

  class Range {
    constructor(start?: number, end?: number);
    start: number;
    end: number;
    static all(): Range;
  }

  class TermCriteria {
    constructor(type?: number, maxCount?: number, epsilon?: number);
    type: number;
    maxCount: number;
    epsilon: number;
  }

  // ==================== Vector Types ====================

  class MatVector implements EmscriptenEmbindInstance {
    constructor();
    delete(): void;
    isDeleted(): boolean;
    push_back(mat: Mat): void;
    get(index: number): Mat;
    set(index: number, mat: Mat): void;
    size(): number;
  }

  class RectVector implements EmscriptenEmbindInstance {
    constructor();
    delete(): void;
    isDeleted(): boolean;
    push_back(rect: Rect): void;
    get(index: number): Rect;
    set(index: number, rect: Rect): void;
    size(): number;
  }

  class PointVector implements EmscriptenEmbindInstance {
    constructor();
    delete(): void;
    isDeleted(): boolean;
    push_back(point: Point): void;
    get(index: number): Point;
    set(index: number, point: Point): void;
    size(): number;
  }

  class KeyPoint {
    constructor();
    constructor(x: number, y: number, size: number, angle?: number, response?: number, octave?: number, class_id?: number);
    pt: Point2f;
    size: number;
    angle: number;
    response: number;
    octave: number;
    class_id: number;
  }

  class KeyPointVector implements EmscriptenEmbindInstance {
    constructor();
    delete(): void;
    isDeleted(): boolean;
    push_back(keypoint: KeyPoint): void;
    get(index: number): KeyPoint;
    set(index: number, keypoint: KeyPoint): void;
    size(): number;
  }

  class DMatch {
    constructor();
    constructor(queryIdx: number, trainIdx: number, distance: number);
    constructor(queryIdx: number, trainIdx: number, imgIdx: number, distance: number);
    queryIdx: number;
    trainIdx: number;
    imgIdx: number;
    distance: number;
  }

  class DMatchVector implements EmscriptenEmbindInstance {
    constructor();
    delete(): void;
    isDeleted(): boolean;
    push_back(match: DMatch): void;
    get(index: number): DMatch;
    set(index: number, match: DMatch): void;
    size(): number;
  }

  class DMatchVectorVector implements EmscriptenEmbindInstance {
    constructor();
    delete(): void;
    isDeleted(): boolean;
    push_back(matches: DMatchVector): void;
    get(index: number): DMatchVector;
    set(index: number, matches: DMatchVector): void;
    size(): number;
  }

  // ==================== Feature Detection ====================

  class Algorithm implements EmscriptenEmbindInstance {
    delete(): void;
    isDeleted(): boolean;
    clear(): void;
    empty(): boolean;
  }

  class Feature2D extends Algorithm {
    detect(image: Mat, keypoints: KeyPointVector, mask?: Mat): void;
    compute(image: Mat, keypoints: KeyPointVector, descriptors: Mat): void;
    detectAndCompute(image: Mat, mask: Mat, keypoints: KeyPointVector, descriptors: Mat, useProvidedKeypoints?: boolean): void;
  }

  class ORB extends Feature2D {
    constructor();
    static create(
      nfeatures?: number,
      scaleFactor?: number,
      nlevels?: number,
      edgeThreshold?: number,
      firstLevel?: number,
      WTA_K?: number,
      scoreType?: number,
      patchSize?: number,
      fastThreshold?: number
    ): ORB;
  }

  class DescriptorMatcher extends Algorithm {
    match(queryDescriptors: Mat, trainDescriptors: Mat, matches: DMatchVector, mask?: Mat): void;
    knnMatch(queryDescriptors: Mat, trainDescriptors: Mat, matches: DMatchVectorVector, k: number, mask?: Mat, compactResult?: boolean): void;
  }

  class BFMatcher extends DescriptorMatcher {
    constructor(normType?: number, crossCheck?: boolean);
  }

  // ==================== Mat Type Constants ====================

  const CV_8U: number;
  const CV_8S: number;
  const CV_16U: number;
  const CV_16S: number;
  const CV_32S: number;
  const CV_32F: number;
  const CV_64F: number;

  const CV_8UC1: number;
  const CV_8UC2: number;
  const CV_8UC3: number;
  const CV_8UC4: number;
  const CV_32SC2: number;
  const CV_32FC2: number;

  // ==================== Interpolation Flags ====================

  const INTER_NEAREST: number;
  const INTER_LINEAR: number;
  const INTER_AREA: number;
  const INTER_CUBIC: number;
  const INTER_LANCZOS4: number;

  // ==================== Border Types ====================

  const BORDER_CONSTANT: number;
  const BORDER_REPLICATE: number;
  const BORDER_REFLECT: number;
  const BORDER_DEFAULT: number;

  // ==================== Color Conversion Codes ====================

  const COLOR_BGR2GRAY: number;
  const COLOR_GRAY2BGR: number;
  const COLOR_BGR2RGB: number;
  const COLOR_RGB2BGR: number;
  const COLOR_BGRA2GRAY: number;
  const COLOR_RGBA2GRAY: number;
  const COLOR_BGR2HSV: number;
  const COLOR_HSV2BGR: number;
  const COLOR_RGBA2BGR: number;
  const COLOR_GRAY2RGBA: number;
  const COLOR_RGB2RGBA: number;

  // ==================== Threshold Types ====================

  const THRESH_BINARY: number;
  const THRESH_BINARY_INV: number;
  const THRESH_TRUNC: number;
  const THRESH_TOZERO: number;
  const THRESH_OTSU: number;

  // ==================== Morphology Types ====================

  const MORPH_RECT: number;
  const MORPH_CROSS: number;
  const MORPH_ELLIPSE: number;
  const MORPH_OPEN: number;
  const MORPH_CLOSE: number;
  const MORPH_GRADIENT: number;
  const MORPH_ERODE: number;
  const MORPH_DILATE: number;

  // ==================== Norm Types ====================

  const NORM_INF: number;
  const NORM_L1: number;
  const NORM_L2: number;
  const NORM_MINMAX: number;
  const NORM_HAMMING: number;

  // ==================== Template Match Modes ====================

  const TM_SQDIFF: number;
  const TM_SQDIFF_NORMED: number;
  const TM_CCORR: number;
  const TM_CCORR_NORMED: number;
  const TM_CCOEFF: number;
  const TM_CCOEFF_NORMED: number;

  // ==================== Rotation Flags ====================

  type RotateFlags = number;
  const ROTATE_90_CLOCKWISE: number;
  const ROTATE_180: number;
  const ROTATE_90_COUNTERCLOCKWISE: number;

  // ==================== Line Types ====================

  const FILLED: number;
  const LINE_4: number;
  const LINE_8: number;
  const LINE_AA: number;

  // ==================== Font Types ====================

  const FONT_HERSHEY_SIMPLEX: number;
  const FONT_HERSHEY_PLAIN: number;
  const FONT_HERSHEY_DUPLEX: number;
  const FONT_HERSHEY_COMPLEX: number;

  // ==================== Contour Retrieval Modes ====================

  const RETR_EXTERNAL: number;
  const RETR_LIST: number;
  const RETR_TREE: number;

  // ==================== Contour Approximation Modes ====================

  const CHAIN_APPROX_NONE: number;
  const CHAIN_APPROX_SIMPLE: number;

  // ==================== Homography Methods ====================

  const RANSAC: number;
  const LMEDS: number;
  const RHO: number;

  // ==================== Functions ====================

  // I/O
  function imread(canvas: HTMLCanvasElement | HTMLImageElement | string): Mat;
  function imshow(canvas: HTMLCanvasElement | string, mat: Mat): void;
  function matFromArray(rows: number, cols: number, type: number, array: number[] | ArrayLike<number>): Mat;
  function matFromImageData(imageData: ImageData): Mat;

  // Color Conversion
  function cvtColor(src: Mat, dst: Mat, code: number, dstCn?: number): void;

  // Geometric Transformations
  function resize(src: Mat, dst: Mat, dsize: Size, fx?: number, fy?: number, interpolation?: number): void;
  function rotate(src: Mat, dst: Mat, rotateCode: number): void;
  function flip(src: Mat, dst: Mat, flipCode: number): void;
  function warpAffine(src: Mat, dst: Mat, M: Mat, dsize: Size, flags?: number, borderMode?: number, borderValue?: Scalar): void;
  function warpPerspective(src: Mat, dst: Mat, M: Mat, dsize: Size, flags?: number, borderMode?: number, borderValue?: Scalar): void;
  function getRotationMatrix2D(center: Point2f, angle: number, scale: number): Mat;
  function getPerspectiveTransform(src: Mat, dst: Mat, solveMethod?: number): Mat;
  function getAffineTransform(src: Mat, dst: Mat): Mat;
  function perspectiveTransform(src: Mat, dst: Mat, m: Mat): void;

  // Thresholding
  function threshold(src: Mat, dst: Mat, thresh: number, maxval: number, type: number): number;
  function adaptiveThreshold(src: Mat, dst: Mat, maxValue: number, adaptiveMethod: number, thresholdType: number, blockSize: number, C: number): void;

  // Filtering
  function GaussianBlur(src: Mat, dst: Mat, ksize: Size, sigmaX: number, sigmaY?: number, borderType?: number): void;
  function medianBlur(src: Mat, dst: Mat, ksize: number): void;
  function blur(src: Mat, dst: Mat, ksize: Size, anchor?: Point, borderType?: number): void;

  // Morphology
  function morphologyEx(src: Mat, dst: Mat, op: number, kernel: Mat, anchor?: Point, iterations?: number, borderType?: number, borderValue?: Scalar): void;
  function erode(src: Mat, dst: Mat, kernel: Mat, anchor?: Point, iterations?: number, borderType?: number, borderValue?: Scalar): void;
  function dilate(src: Mat, dst: Mat, kernel: Mat, anchor?: Point, iterations?: number, borderType?: number, borderValue?: Scalar): void;
  function getStructuringElement(shape: number, ksize: Size, anchor?: Point): Mat;

  // Edge Detection
  function Canny(image: Mat, edges: Mat, threshold1: number, threshold2: number, apertureSize?: number, L2gradient?: boolean): void;

  // Contours
  function findContours(image: Mat, contours: MatVector, hierarchy: Mat, mode: number, method: number, offset?: Point): void;
  function drawContours(image: Mat, contours: MatVector, contourIdx: number, color: Scalar, thickness?: number, lineType?: number, hierarchy?: Mat, maxLevel?: number, offset?: Point): void;
  function contourArea(contour: Mat, oriented?: boolean): number;
  function arcLength(curve: Mat, closed: boolean): number;
  function approxPolyDP(curve: Mat, approxCurve: Mat, epsilon: number, closed: boolean): void;
  function convexHull(points: Mat, hull: Mat, clockwise?: boolean, returnPoints?: boolean): void;
  function boundingRect(array: Mat): Rect;
  function minAreaRect(points: Mat): RotatedRect;

  // Core Operations
  function normalize(src: Mat, dst: Mat, alpha?: number, beta?: number, norm_type?: number, dtype?: number, mask?: Mat): void;
  function minMaxLoc(src: Mat, mask?: Mat): { minVal: number; maxVal: number; minLoc: Point; maxLoc: Point };
  function mean(src: Mat, mask?: Mat): Scalar;
  function meanStdDev(src: Mat, mean: Mat, stddev: Mat, mask?: Mat): void;
  function countNonZero(src: Mat): number;

  // Histogram
  function calcHist(images: MatVector, channels: number[], mask: Mat, hist: Mat, histSize: number[], ranges: number[], accumulate?: boolean): void;

  // Template Matching
  function matchTemplate(image: Mat, templ: Mat, result: Mat, method: number, mask?: Mat): void;

  // Arithmetic Operations
  function add(src1: Mat, src2: Mat | Scalar, dst: Mat, mask?: Mat, dtype?: number): void;
  function subtract(src1: Mat, src2: Mat | Scalar, dst: Mat, mask?: Mat, dtype?: number): void;
  function multiply(src1: Mat, src2: Mat | Scalar, dst: Mat, scale?: number, dtype?: number): void;
  function divide(src1: Mat | number, src2: Mat | Scalar, dst: Mat, scale?: number, dtype?: number): void;
  function addWeighted(src1: Mat, alpha: number, src2: Mat, beta: number, gamma: number, dst: Mat, dtype?: number): void;
  function absdiff(src1: Mat, src2: Mat | Scalar, dst: Mat): void;

  // Bitwise Operations
  function bitwise_and(src1: Mat, src2: Mat, dst: Mat, mask?: Mat): void;
  function bitwise_or(src1: Mat, src2: Mat, dst: Mat, mask?: Mat): void;
  function bitwise_not(src: Mat, dst: Mat, mask?: Mat): void;

  // Comparison Operations
  function inRange(src: Mat, lowerb: Mat | Scalar, upperb: Mat | Scalar, dst: Mat): void;
  function compare(src1: Mat, src2: Mat | number, dst: Mat, cmpop: number): void;

  // Lookup Table
  function LUT(src: Mat, lut: Mat, dst: Mat): void;

  // Channel Operations
  function split(m: Mat, mv: MatVector): void;
  function merge(mv: MatVector, dst: Mat): void;

  // Border Operations
  function copyMakeBorder(src: Mat, dst: Mat, top: number, bottom: number, left: number, right: number, borderType: number, value?: Scalar): void;

  // Concatenation
  function hconcat(src: MatVector, dst: Mat): void;
  function vconcat(src: MatVector, dst: Mat): void;

  // Homography

  // Drawing
  function circle(img: Mat, center: Point, radius: number, color: Scalar, thickness?: number, lineType?: number, shift?: number): void;
  function rectangle(img: Mat, pt1: Point | Rect, pt2: Point | Scalar, color?: Scalar, thickness?: number, lineType?: number, shift?: number): void;
  function line(img: Mat, pt1: Point, pt2: Point, color: Scalar, thickness?: number, lineType?: number, shift?: number): void;
  function polylines(img: Mat, pts: MatVector, isClosed: boolean, color: Scalar, thickness?: number, lineType?: number, shift?: number): void;
  function fillPoly(img: Mat, pts: MatVector, color: Scalar, lineType?: number, shift?: number, offset?: Point): void;
  function arrowedLine(img: Mat, pt1: Point, pt2: Point, color: Scalar, thickness?: number, lineType?: number, shift?: number, tipLength?: number): void;
  function putText(img: Mat, text: string, org: Point, fontFace: number, fontScale: number, color: Scalar, thickness?: number, lineType?: number, bottomLeftOrigin?: boolean): void;

  function getTextSize(text: string, fontFace: number, fontScale: number, thickness: number, baseLine?: number): { width: number; height: number; baseLine: number };
  function findHomography(srcPoints: Mat, dstPoints: Mat, method?: number, ransacReprojThreshold?: number, mask?: Mat, maxIters?: number, confidence?: number): Mat;

  // Utility functions
  function getBuildInformation(): string;
  function setNumThreads(nthreads: number): void;

  // FS for Emscripten file system
  const FS: any;

  // Runtime initialization callback
  let onRuntimeInitialized: (() => void) | undefined;
}

// Declare global cv variable
declare var cv: typeof cv;
