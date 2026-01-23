// Scalar, Point, Rect, etc are defined by opencv.js (helpers.js) and we need to declare them manually:

export declare class Range {
  public start: number;
  public end: number;
  public constructor(start: number, end: number);
}

export declare class Scalar extends Array<number> {
  public static all(...v: number[]): Scalar;
}
// Hack: expose Mat super classes like Mat_, InputArray, Vector, OutputArray we make them alias of Mat to simplify and make it work
export { Mat as InputArray };
export { Mat as InputOutputArray };
export { Mat as OutputArray };
export { MatVector as InputArrayOfArrays };
export { MatVector as InputOutputArrayOfArrays };
export { MatVector as OutputArrayOfArrays };
export { Scalar as GScalar };
export { Point as Point2f };
export { Point as KeyPoint };
export { Point as Point2l };
export { Size as Point2d };
export { Size as Size2d };
export { Size as Size2f };
export { Size as Size2l };
export { Rect as Rect_ };

export declare class Point {
  public constructor(x: number, y: number);
  public x: number;
  public y: number;
}

export declare class Circle {
  public constructor(center: Point, radius: number);
  public center: Point;
  public radius: number;
}

export declare class Size {
  public constructor(width: number, height: number);
  public width: number;
  public height: number;
}

export declare class Rect {
  public constructor();
  public constructor(point: Point, size: Size);
  public constructor(x: number, y: number, width: number, height: number);
  public x: number;
  public y: number;
  public width: number;
  public height: number;
}

export declare class TermCriteria {
  public type: number;
  public maxCount: number;
  public epsilon: number;
  public constructor();
  public constructor(type: number, maxCount: number, epsilon: number);
}
export declare const TermCriteria_EPS: any;
export declare const TermCriteria_COUNT: any;
export declare const TermCriteria_MAX_ITER: any;

export declare class MinMaxLoc {
  public minVal: number;
  public maxVal: number;
  public minLoc: Point;
  public maxLoc: Point;
  public constructor();
  public constructor(
    minVal: number,
    maxVal: number,
    minLoc: Point,
    maxLoc: Point,
  );
}

// expose emscripten / opencv.js specifics

export declare function exceptionFromPtr(err: number): any;
export declare function onRuntimeInitialized(): any;
export declare function FS_createDataFile(
  parent: string,
  name: string,
  data: Uint8Array,
  canRead: boolean,
  canWrite: boolean,
  canOwn: boolean,
): any;

import { Algorithm, type LineTypes, Mat, type NormTypes, RotatedRect } from ".";
import "../_cv";

/**
 * Base class for Contrast Limited Adaptive Histogram Equalization.
 */
export declare class CLAHE extends Algorithm {
  /**
   * @param clipLimit Threshold for contrast limiting. Default.  40.0,
   * @param totalGridSize Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column. Default: Size(8, 8)
   */
  constructor(clipLimit?: double, totalGridSize?: Size);
  /**
   * Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.
   * @param src Source image of type CV_8UC1 or CV_16UC1.
   * @param dst Destination image.
   */
  apply(src: Mat, dst: Mat): void;
  collectGarbage(): void;
  /**
   * Returns threshold value for contrast limiting.
   */
  getClipLimit(): double;
  /**
   * Returns Size defines the number of tiles in row and column.
   */
  getTilesGridSize(): Size;
  /**
   * Sets threshold for contrast limiting.
   */
  setClipLimit(clipLimit: double): void;
  /**
   * Sets size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles.
   * @param tileGridSize defines the number of tiles in row and column.
   */
  setTilesGridSize(tileGridSize: Size): void;
}

// emscripten embind internals
export declare function getInheritedInstanceCount(...a: any[]): any;
export declare function getLiveInheritedInstances(...a: any[]): any;
export declare function flushPendingDeletes(...a: any[]): any;
export declare function setDelayFunction(...a: any[]): any;

export declare class EmscriptenEmbindInstance {
  isAliasOf(other: any): bool;
  clone(): any;
  delete(): any;
  isDeleted(): boolean;
  deleteLater(): any;
}

export declare class InternalError extends Error {}
export declare class BindingError extends Error {}
export declare class UnboundTypeError extends Error {}
export declare class PureVirtualError extends Error {}

export declare class Vector<T> extends EmscriptenEmbindInstance {
  get(i: number): T;
  get(i: number, j: number, data: any): T;
  set(i: number, t: T): void;
  put(i: number, j: number, data: any): any;
  size(): number;
  push_back(n: T): any;
  resize(count: number, value?: T): void;
}

export declare class Vec3d extends Vector<any> {}
export declare class IntVector extends Vector<number> {}
export declare class FloatVector extends Vector<number> {}
export declare class DoubleVector extends Vector<number> {}
export declare class PointVector extends Vector<Point> {}
export declare class KeyPointVector extends Vector<any> {}
export declare class DMatchVector extends Vector<any> {}
export declare class DMatchVectorVector extends Vector<Vector<any>> {}
export declare class MatVector extends Vector<Mat> {}

export declare class RectVector extends Rect implements Vector<Rect> {
  get(i: number): Rect;
  isAliasOf(...a: any[]): any;
  clone(...a: any[]): any;
  delete(...a: any[]): any;
  isDeleted(...a: any[]): any;
  deleteLater(...a: any[]): any;
  set(i: number, t: Rect): void;
  put(i: number, j: number, data: any): any;
  size(): number;
  push_back(n: Rect): void;
  resize(count: number, value?: Rect | undefined): void;
  delete(): void;
}

export declare class VideoCapture {
  public constructor(videoSource: HTMLVideoElement | string);
  public read(m: Mat): any;
  public video: HTMLVideoElement;
}

export type MatSize = () => Size;

export declare function matFromImageData(imageData: ImageData): Mat;
export declare function matFromArray(
  rows: number,
  cols: number,
  type: any,
  array: number[] | ArrayBufferView,
): Mat;

export declare class ImageData {
  data: ArrayBufferView;
  width: number;
  height: number;
}

// TODO this types should be exposed by the tool - want to make it work:
export declare const CV_8U: CVDataType;
export declare const CV_8UC1: CVDataType;
export declare const CV_8UC2: CVDataType;
export declare const CV_8UC3: CVDataType;
export declare const CV_8UC4: CVDataType;
export declare const CV_8S: CVDataType;
export declare const CV_8SC1: CVDataType;
export declare const CV_8SC2: CVDataType;
export declare const CV_8SC3: CVDataType;
export declare const CV_8SC4: CVDataType;
export declare const CV_16U: CVDataType;
export declare const CV_16UC1: CVDataType;
export declare const CV_16UC2: CVDataType;
export declare const CV_16UC3: CVDataType;
export declare const CV_16UC4: CVDataType;
export declare const CV_16S: CVDataType;
export declare const CV_16SC1: CVDataType;
export declare const CV_16SC2: CVDataType;
export declare const CV_16SC3: CVDataType;
export declare const CV_16SC4: CVDataType;
export declare const CV_32S: CVDataType;
export declare const CV_32SC1: CVDataType;
export declare const CV_32SC2: CVDataType;
export declare const CV_32SC3: CVDataType;
export declare const CV_32SC4: CVDataType;
export declare const CV_32F: CVDataType;
export declare const CV_32FC1: CVDataType;
export declare const CV_32FC2: CVDataType;
export declare const CV_32FC3: CVDataType;
export declare const CV_32FC4: CVDataType;
export declare const CV_64F: CVDataType;
export declare const CV_64FC1: CVDataType;
export declare const CV_64FC2: CVDataType;
export declare const CV_64FC3: CVDataType;
export declare const CV_64FC4: CVDataType;

export type CVDataType = any;

export declare function ellipse1(
  dst: Mat,
  rotatedRect: RotatedRect,
  ellipseColor: Scalar,
  arg0: number,
  line: LineTypes,
): void;
export declare function imread(
  canvasOrImageHtmlElement: HTMLElement | string,
): Mat;
export declare function norm1(a: Mat, b: Mat, type: NormTypes): number;
export declare function imshow(
  canvasSource: HTMLElement | string,
  mat: Mat,
): void;
export declare function matFromArray(
  rows: number,
  cols: number,
  type: any,
  array: any,
): Mat;

// Missing imports:
export type Mat4 = any;
export type Mat3 = any;
export type Vec3 = any;
export type float_type = any;
export type int = number;
export type bool = boolean;
export type FileNode = any;
export type FileStorage = any;
export type Ptr = any;
export type size_t = any;
export type double = number;
export type float = number;
export type UMat = any;
export type Matrix = any;
export type BucketKey = any;
export type Bucket = any;
export type LshStats = any;
export type MatAllocator = any;
export type uchar = any;
export type MatStep = any;
export type UMatData = any;
export type typename = any;
export type Vec = any;
export type Point_ = any;
export type Point3_ = any;
export type MatCommaInitializer_ = any;
export type MatIterator_ = any;
export type MatConstIterator_ = any;
export type AccessFlag = any;
export type UMatUsageFlags = any;
export type _Tp = any;
export type Matx_AddOp = any;
export type Matx_SubOp = any;
export type _T2 = any;
export type Matx_ScaleOp = any;
export type Matx_MulOp = any;
export type Matx_DivOp = any;
export type Matx_MatMulOp = any;
export type Matx_TOp = any;
export type diag_type = any;
export type _EqPredicate = any;
export type cvhalDFT = any;
export type schar = any;
export type ushort = any;
export type short = any;
export type int64 = any;
export type ErrorCallback = any;
export type unsigned = any;
export type uint64 = any;
export type float16_t = any;
export type AsyncArray = any;
export type Net = any;
export type Moments = any;
export type uint64_t = any;
export type uint32_t = any;
export type int32_t = any;
export type int64_t = any;
