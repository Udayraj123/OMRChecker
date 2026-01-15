/**
 * Base class for image preprocessing.
 *
 * TypeScript port of src/processors/image/base.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * All image preprocessors inherit from this class and implement the
 * unified Processor interface with process(context) method.
 */

import type * as cv from '@techstark/opencv-js';
import { Processor, ProcessingContext } from '../base';
import { ImageUtils } from '../../utils/ImageUtils';
import { Logger } from '../../utils/logger';
import type { TuningConfig } from '../../template/types';

const logger = new Logger('ImageTemplatePreprocessor');

/**
 * Options for image preprocessors
 */
export interface ImagePreprocessorOptions {
  /**
   * Specific options for this preprocessor
   */
  [key: string]: any;

  /**
   * Tuning options for the preprocessor
   */
  tuningOptions?: Record<string, any>;

  /**
   * Processing image shape [width, height]
   */
  processingImageShape?: [number, number];

  /**
   * Output configuration
   */
  output?: any;
}

/**
 * Configuration for saving images during processing
 */
export interface SaveImageOps {
  /**
   * Function to append an image to be saved
   */
  appendSaveImage: (name: string, image: cv.Mat) => void;

  /**
   * Tuning configuration
   */
  tuningConfig: TuningConfig;
}

/**
 * Base class for image preprocessing.
 *
 * All image preprocessors inherit from this class and implement the
 * unified Processor interface with process(context) method.
 */
export abstract class ImageTemplatePreprocessor extends Processor {
  protected options: ImagePreprocessorOptions;
  protected tuningOptions: Record<string, any>;
  protected relativeDir: string;
  protected description: string = 'UNKNOWN';
  protected appendSaveImage: (name: string, image: cv.Mat) => void;
  protected tuningConfig: any;
  protected processingImageShape: [number, number] | null;
  protected output: any;

  /**
   * Initialize the image preprocessor.
   *
   * @param options - Preprocessor-specific options
   * @param relativeDir - Relative directory for file operations
   * @param saveImageOps - Operations for saving images
   * @param defaultProcessingImageShape - Default shape for processing images
   */
  constructor(
    options: ImagePreprocessorOptions,
    relativeDir: string,
    saveImageOps: SaveImageOps,
    defaultProcessingImageShape: [number, number]
  ) {
    super();

    // Initialize processor-specific attributes
    this.options = options;
    this.tuningOptions = options.tuningOptions || {};
    this.relativeDir = relativeDir;

    // Image preprocessing specific attributes
    this.appendSaveImage = saveImageOps.appendSaveImage;
    this.tuningConfig = saveImageOps.tuningConfig;

    // Note: we're taking this at preProcessor level because it represents
    // the need of a preProcessor's coordinate system (e.g. zone selectors)
    this.processingImageShape = options.processingImageShape || defaultProcessingImageShape;
    this.output = options.output;
  }

  /**
   * Get relative path joined with the given path.
   *
   * @param path - Path to join with relative directory
   * @returns Joined path
   */
  protected getRelativePath(path: string): string {
    return `${this.relativeDir}/${path}`;
  }

  /**
   * Apply filter to the image and return modified images.
   *
   * @param image - Grayscale image to process
   * @param coloredImage - Colored image to process
   * @param template - Template configuration
   * @param filePath - Path to the file being processed
   * @returns Tuple of [processed grayscale image, processed colored image, updated template]
   */
  protected abstract applyFilter(
    image: cv.Mat,
    coloredImage: cv.Mat,
    template: any,
    filePath: string
  ): [cv.Mat, cv.Mat, any];

  /**
   * Get the class name for this preprocessor.
   *
   * @returns Class name
   */
  protected abstract getClassName(): string;

  /**
   * Get the name of this processor (required by unified Processor interface).
   *
   * @returns Processor name
   */
  getName(): string {
    return this.getClassName();
  }

  /**
   * Process images using the unified processor interface.
   *
   * This is the interface that all processors must implement.
   *
   * @param context - Processing context with images and state
   * @returns Updated context with processed images
   */
  process(context: ProcessingContext): ProcessingContext {
    logger.debug(`Starting ${this.getName()} processor`);

    let grayImage = context.grayImage;
    let coloredImage = context.coloredImage;
    const template = context.template;
    const filePath = context.filePath;

    // Resize images to preprocessor's processing shape
    if (this.processingImageShape) {
      grayImage = ImageUtils.resizeToShape(this.processingImageShape, grayImage) as cv.Mat;

      if (this.tuningConfig.outputs.colored_outputs_enabled) {
        coloredImage = ImageUtils.resizeToShape(
          this.processingImageShape,
          coloredImage
        ) as cv.Mat;
      }
    }

    // Apply the specific filter
    const [processedGray, processedColored, updatedTemplate] = this.applyFilter(
      grayImage,
      coloredImage,
      template,
      filePath
    );

    // Update context
    context.grayImage = processedGray;
    context.coloredImage = processedColored;
    context.template = updatedTemplate;

    logger.debug(`Completed ${this.getName()} processor`);

    return context;
  }

  /**
   * Return a list of file paths that should be excluded from processing.
   *
   * @returns Array of file paths to exclude
   */
  excludeFiles(): string[] {
    return [];
  }
}

