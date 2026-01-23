import type { double, InputArray, int, OutputArray } from "./_types";
/*
 * # Inpainting
 * the inpainting algorithm
 */
/**
 * The function reconstructs the selected image area from the pixel near the area boundary. The
 * function may be used to remove dust and scratches from a scanned photo, or to remove undesirable
 * objects from still images or video. See  for more details.
 *
 * An example using the inpainting technique can be found at opencv_source_code/samples/cpp/inpaint.cpp
 * (Python) An example using the inpainting technique can be found at
 * opencv_source_code/samples/python/inpaint.py
 *
 * @param src Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.
 *
 * @param inpaintMask Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that
 * needs to be inpainted.
 *
 * @param dst Output image with the same size and type as src .
 *
 * @param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered by
 * the algorithm.
 *
 * @param flags Inpainting method that could be cv::INPAINT_NS or cv::INPAINT_TELEA
 */
export declare function inpaint(
  src: InputArray,
  inpaintMask: InputArray,
  dst: OutputArray,
  inpaintRadius: double,
  flags: int,
): void;

export declare const INPAINT_NS: any; // initializer: = 0

export declare const INPAINT_TELEA: any; // initializer: = 1
