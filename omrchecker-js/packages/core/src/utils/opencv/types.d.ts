/**
 * TypeScript declarations for OpenCV.js namespace
 *
 * Provides type definitions for cv namespace types used throughout the codebase.
 * The actual implementations come from @techstark/opencv-js at runtime.
 */

declare namespace cv {
  /**
   * OpenCV Mat class for image data
   */
  class Mat {
    constructor(rows: number, cols: number, type?: number, scalar?: Scalar);
    clone(): Mat;
    isDeleted(): boolean;
    delete(): void;
    empty(): boolean;
    cols: number;
    rows: number;
    type(): number;
    setTo(scalar: Scalar): void;
    static ones(rows: number, cols: number, type: number): Mat;
  }

  /**
   * OpenCV Scalar class for color values
   */
  class Scalar {
    constructor(...values: number[]);
  }

  /**
   * OpenCV Point class for 2D coordinates
   */
  class Point {
    constructor(x: number, y: number);
    x: number;
    y: number;
  }

  /**
   * OpenCV Size class for dimensions
   */
  class Size {
    constructor(width: number, height: number);
    width: number;
    height: number;
  }

  /**
   * OpenCV Rect class for rectangles
   */
  class Rect {
    constructor(x: number, y: number, width: number, height: number);
    x: number;
    y: number;
    width: number;
    height: number;
  }

  /**
   * OpenCV MatVector for collections of Mat objects
   */
  class MatVector {
    constructor();
    push_back(mat: Mat): void;
  }

  /**
   * Create Mat from array data
   */
  function matFromArray(
    rows: number,
    cols: number,
    type: number,
    array: number[]
  ): Mat;

  /**
   * Normalize image values
   */
  function normalize(
    src: Mat,
    dst: Mat,
    alpha: number,
    beta: number,
    normType: number,
    dtype?: number
  ): void;

  /**
   * Erode image
   */
  function erode(
    src: Mat,
    dst: Mat,
    kernel: Mat,
    anchor?: Point,
    iterations?: number
  ): void;

  // Common OpenCV constants
  const CV_8UC1: number;
  const CV_8UC3: number;
  const CV_8U: number;
  const CV_32SC2: number;
  const CV_32FC2: number;
  const NORM_MINMAX: number;
  const NORM_HAMMING: number;
}
