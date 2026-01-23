import type {
  bool,
  double,
  FileNode,
  InputArray,
  int,
  Mat,
  Ptr,
  Size,
} from "./_types";

export declare class CascadeClassifier extends Mat {
  public cc: Ptr;

  public constructor();

  /**
   * @param filename Name of the file from which the classifier is loaded.
   */
  public constructor(filename: String);

  /**
   *   The function is parallelized with the TBB library.
   *
   * (Python) A face detection example using cascade classifiers can be found at
   * opencv_source_code/samples/python/facedetect.py
   *
   * @param image Matrix of the type CV_8U containing an image where objects are detected.
   *
   * @param objects Vector of rectangles where each rectangle contains the detected object, the
   * rectangles may be partially outside the original image.
   *
   * @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
   *
   * @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
   * to retain it.
   *
   * @param flags Parameter with the same meaning for an old cascade as in the function
   * cvHaarDetectObjects. It is not used for a new cascade.
   *
   * @param minSize Minimum possible object size. Objects smaller than that are ignored.
   *
   * @param maxSize Maximum possible object size. Objects larger than that are ignored. If maxSize ==
   * minSize model is evaluated on single scale.
   */
  public detectMultiScale(
    image: InputArray,
    objects: any,
    scaleFactor?: double,
    minNeighbors?: int,
    flags?: int,
    minSize?: Size,
    maxSize?: Size,
  ): InputArray;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param image Matrix of the type CV_8U containing an image where objects are detected.
   *
   * @param objects Vector of rectangles where each rectangle contains the detected object, the
   * rectangles may be partially outside the original image.
   *
   * @param numDetections Vector of detection numbers for the corresponding objects. An object's number
   * of detections is the number of neighboring positively classified rectangles that were joined
   * together to form the object.
   *
   * @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
   *
   * @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
   * to retain it.
   *
   * @param flags Parameter with the same meaning for an old cascade as in the function
   * cvHaarDetectObjects. It is not used for a new cascade.
   *
   * @param minSize Minimum possible object size. Objects smaller than that are ignored.
   *
   * @param maxSize Maximum possible object size. Objects larger than that are ignored. If maxSize ==
   * minSize model is evaluated on single scale.
   */
  public detectMultiScale(
    image: InputArray,
    objects: any,
    numDetections: any,
    scaleFactor?: double,
    minNeighbors?: int,
    flags?: int,
    minSize?: Size,
    maxSize?: Size,
  ): InputArray;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts. This function allows you to retrieve the final stage
   * decision certainty of classification. For this, one needs to set `outputRejectLevels` on true and
   * provide the `rejectLevels` and `levelWeights` parameter. For each resulting detection,
   * `levelWeights` will then contain the certainty of classification at the final stage. This value can
   * then be used to separate strong from weaker classifications.
   *
   *   A code sample on how to use it efficiently can be found below:
   *
   *   ```cpp
   *   Mat img;
   *   vector<double> weights;
   *   vector<int> levels;
   *   vector<Rect> detections;
   *   CascadeClassifier model("/path/to/your/model.xml");
   *   model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
   *   cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
   *   ```
   */
  public detectMultiScale(
    image: InputArray,
    objects: any,
    rejectLevels: any,
    levelWeights: any,
    scaleFactor?: double,
    minNeighbors?: int,
    flags?: int,
    minSize?: Size,
    maxSize?: Size,
    outputRejectLevels?: bool,
  ): InputArray;

  public empty(): bool;

  public getFeatureType(): int;

  public getMaskGenerator(): Ptr;

  public getOldCascade(): any;

  public getOriginalWindowSize(): Size;

  public isOldFormatCascade(): bool;

  /**
   * @param filename Name of the file from which the classifier is loaded. The file may contain an old
   * HAAR classifier trained by the haartraining application or a new cascade classifier trained by the
   * traincascade application.
   */
  public load(filename: String): String;

  /**
   *   The file may contain a new cascade classifier (trained traincascade application) only.
   */
  public read(node: FileNode): FileNode;

  public setMaskGenerator(maskGenerator: Ptr): Ptr;

  public static convert(oldcascade: String, newcascade: String): String;
}
