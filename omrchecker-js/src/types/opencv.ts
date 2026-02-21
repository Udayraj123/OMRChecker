/**
 * OpenCV.js Type Definitions
 * Minimal type definitions for OpenCV.js types used in OMRChecker
 */

/**
 * Represents an OpenCV Mat object (matrix/image)
 * This is a placeholder until proper OpenCV.js types are loaded
 */
export interface MatLike {
  rows: number;
  cols: number;
  data: Uint8Array | Uint8ClampedArray;
  delete(): void;
  clone(): MatLike;
  roi(rect: { x: number; y: number; width: number; height: number }): MatLike;
}

/**
 * Point in 2D space
 */
export interface Point {
  x: number;
  y: number;
}

/**
 * Rectangle definition
 */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Size definition
 */
export interface Size {
  width: number;
  height: number;
}
