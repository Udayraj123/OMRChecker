import type { float, Point2f, Rect, Size2f } from "./_types";

/**
 * Each rectangle is specified by the center point (mass center), length of each side (represented by
 * [Size2f](#dc/d84/group__core__basic_1gab34496d2466b5f69930ab74c70f117d4}) structure) and the
 * rotation angle in degrees.
 *
 * The sample below demonstrates how to use [RotatedRect](#db/dd6/classcv_1_1RotatedRect}):
 *
 * ```cpp
 *     Mat test_image(200, 200, CV_8UC3, Scalar(0));
 *     RotatedRect rRect = RotatedRect(Point2f(100,100), Size2f(100,50), 30);
 *
 *     Point2f vertices[4];
 *     rRect.points(vertices);
 *     for (int i = 0; i < 4; i++)
 *         line(test_image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 2);
 *
 *     Rect brect = rRect.boundingRect();
 *     rectangle(test_image, brect, Scalar(255,0,0), 2);
 *
 *     imshow("rectangles", test_image);
 *     waitKey(0);
 * ```
 *
 * [CamShift](#dc/d6b/group__video__track_1gaef2bd39c8356f423124f1fe7c44d54a1}),
 * [fitEllipse](#d3/dc0/group__imgproc__shape_1gaf259efaad93098103d6c27b9e4900ffa}),
 * [minAreaRect](#d3/dc0/group__imgproc__shape_1ga3d476a3417130ae5154aea421ca7ead9}), CvBox2D
 *
 * Source:
 * [opencv2/core/types.hpp](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/core/types.hpp#L534).
 *
 */
export declare class RotatedRect {
  public angle: float;

  public center: Point2f;

  public size: Size2f;

  public constructor();

  /**
   *   full constructor
   *
   * @param center The rectangle mass center.
   *
   * @param size Width and height of the rectangle.
   *
   * @param angle The rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc.,
   * the rectangle becomes an up-right rectangle.
   */
  public constructor(center: Point2f, size: Size2f, angle: float);

  /**
   *   Any 3 end points of the [RotatedRect]. They must be given in order (either clockwise or
   * anticlockwise).
   */
  public constructor(point1: Point2f, point2: Point2f, point3: Point2f);

  public static boundingRect(rect: RotatedRect): Rect;

  public static boundingRect2f(rect: RotatedRect): Rect;

  /**
    returns 4 vertices of the rectangle
   * @param rect The rotated rectangle
   * @returns Array of 4 points in order: bottomLeft, topLeft, topRight, bottomRight
   */
  public static points(rect: RotatedRect): Point2f[];
}
