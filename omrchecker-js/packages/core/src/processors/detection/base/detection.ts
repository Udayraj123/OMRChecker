/**
 * Base classes for field detection
 *
 * Port of Python's FieldDetection and TextDetection base classes.
 */

/**
 * Base class for text detection results
 * Stores detected text, bounding box, rotated rectangle and confidence score
 */
export class TextDetection {
  constructor(
    public detectedText: string | null,
    public boundingBox: [number, number, number, number],
    public rotatedRectangle: any,
    public confidentScore: number
  ) {}

  /**
   * Check if detection result is null
   */
  isNull(): boolean {
    return this.detectedText === null;
  }
}

/**
 * Abstract base class for field detection
 * All detection processors should extend this class
 */
export abstract class FieldDetection {
  protected field: any;
  protected grayImage: any;
  protected coloredImage: any;

  constructor(field: any, grayImage: any, coloredImage: any) {
    this.field = field;
    this.grayImage = grayImage;
    this.coloredImage = coloredImage;
    // Note: field object can have the corresponding runtime config for the detection
    this.runDetection(field, grayImage, coloredImage);
  }

  /**
   * Abstract method to be implemented by subclasses
   * Performs the actual detection logic
   */
  protected abstract runDetection(
    field: any,
    grayImage: any,
    coloredImage: any
  ): void;
}

