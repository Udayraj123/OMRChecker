import type {
  bool,
  InputArray,
  OutputArray,
  OutputArrayOfArrays,
} from "./_types";

/**
 * QR Code detection and decoding class.
 *
 * This class implements QR code detection and decoding functionality.
 * It can detect QR codes in an image and decode their content.
 *
 * Source:
 * [opencv2/objdetect.hpp](https://github.com/opencv/opencv/tree/master/modules/objdetect/include/opencv2/objdetect.hpp).
 */
export declare class QRCodeDetector {
  /**
   * QRCodeDetector constructor
   */
  public constructor();

  /**
   * Detects QR code in image and returns the quadrangle containing the code.
   *
   * @param img grayscale or color (BGR) image containing (or not) QR code.
   * @param points Output vector of vertices of the minimum-area quadrangle containing the code.
   */
  public detect(img: InputArray, points: OutputArray): bool;

  /**
   * Decodes QR code in image once it's found by the detect() method.
   *
   * @param img grayscale or color (BGR) image containing QR code.
   * @param points Quadrangle vertices found by detect() method (or some other algorithm).
   * @param straight_qrcode The optional output image containing rectified and binarized QR code
   */
  public decode(
    img: InputArray,
    points: InputArray,
    straight_qrcode?: OutputArray,
  ): string;

  /**
   * Both detects and decodes QR code
   *
   * @param img grayscale or color (BGR) image containing QR code.
   * @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
   * @param straight_qrcode The optional output image containing rectified and binarized QR code
   */
  public detectAndDecode(
    img: InputArray,
    points?: OutputArray,
    straight_qrcode?: OutputArray,
  ): string;

  /**
   * Detects QR codes in image and returns the vector of the quadrangles containing the codes.
   *
   * @param img grayscale or color (BGR) image containing (or not) QR codes.
   * @param points Output vector of vector of vertices of the minimum-area quadrangle containing the codes.
   */
  public detectMulti(img: InputArray, points: OutputArrayOfArrays): bool;

  /**
   * Decodes QR codes in image once it's found by the detectMulti() method.
   *
   * @param img grayscale or color (BGR) image containing QR codes.
   * @param points vector of Quadrangle vertices found by detectMulti() method (or some other algorithm).
   * @param decoded_info UTF8-encoded output vector of String or empty vector of String if the codes cannot be decoded.
   * @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
   */
  public decodeMulti(
    img: InputArray,
    points: InputArray,
    decoded_info: any,
    straight_qrcode?: OutputArrayOfArrays,
  ): bool;

  /**
   * Both detects and decodes QR codes
   *
   * @param img grayscale or color (BGR) image containing QR codes.
   * @param decoded_info UTF8-encoded output vector of String or empty vector of String if the codes cannot be decoded.
   * @param points optional output vector of vertices of the found QR code quadrangles. Will be empty if not found.
   * @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
   */
  public detectAndDecodeMulti(
    img: InputArray,
    decoded_info: any,
    points?: OutputArrayOfArrays,
    straight_qrcode?: OutputArrayOfArrays,
  ): bool;

  /**
   * Aruco-based QR code detector
   */
  public setUseAruco(use_aruco: bool): void;

  /**
   * Get if Aruco-based QR code detector is used
   */
  public getUseAruco(): bool;

  /**
   * Releases the object
   */
  public delete(): void;
}
