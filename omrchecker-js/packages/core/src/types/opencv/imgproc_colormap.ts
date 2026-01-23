import type { InputArray, int, OutputArray } from "./_types";

/*
 * # Colormap Transformations
 *
 */

/**
 * Applies a colormap on a given image.
 *
 * @param src The source image, which should be grayscale. Should be 8-bit, 16-bit, or floating-point.
 * @param dst The result is the colored image.
 * @param colormap The colormap to apply.
 */
export declare function applyColorMap(
  src: InputArray,
  dst: OutputArray,
  colormap: int,
): void;

/**
 * Applies a user colormap on a given image.
 *
 * @param src The source image, which should be grayscale. Should be 8-bit, 16-bit, or floating-point.
 * @param dst The result is the colored image.
 * @param userColor The colormap to apply of type CV_8UC1 or CV_8UC3 and size 256.
 */
export declare function applyColorMap(
  src: InputArray,
  dst: OutputArray,
  userColor: InputArray,
): void;

/**
 * Colormap types used by the applyColorMap function.
 */
export type ColormapTypes = any;

export declare const COLORMAP_AUTUMN: ColormapTypes; // initializer: = 0
export declare const COLORMAP_BONE: ColormapTypes; // initializer: = 1
export declare const COLORMAP_JET: ColormapTypes; // initializer: = 2
export declare const COLORMAP_WINTER: ColormapTypes; // initializer: = 3
export declare const COLORMAP_RAINBOW: ColormapTypes; // initializer: = 4
export declare const COLORMAP_OCEAN: ColormapTypes; // initializer: = 5
export declare const COLORMAP_SUMMER: ColormapTypes; // initializer: = 6
export declare const COLORMAP_SPRING: ColormapTypes; // initializer: = 7
export declare const COLORMAP_COOL: ColormapTypes; // initializer: = 8
export declare const COLORMAP_HSV: ColormapTypes; // initializer: = 9
export declare const COLORMAP_PINK: ColormapTypes; // initializer: = 10
export declare const COLORMAP_HOT: ColormapTypes; // initializer: = 11
export declare const COLORMAP_PARULA: ColormapTypes; // initializer: = 12
export declare const COLORMAP_MAGMA: ColormapTypes; // initializer: = 13
export declare const COLORMAP_INFERNO: ColormapTypes; // initializer: = 14
export declare const COLORMAP_PLASMA: ColormapTypes; // initializer: = 15
export declare const COLORMAP_VIRIDIS: ColormapTypes; // initializer: = 16
export declare const COLORMAP_CIVIDIS: ColormapTypes; // initializer: = 17
export declare const COLORMAP_TWILIGHT: ColormapTypes; // initializer: = 18
export declare const COLORMAP_TWILIGHT_SHIFTED: ColormapTypes; // initializer: = 19
export declare const COLORMAP_TURBO: ColormapTypes; // initializer: = 20
export declare const COLORMAP_DEEPGREEN: ColormapTypes; // initializer: = 21