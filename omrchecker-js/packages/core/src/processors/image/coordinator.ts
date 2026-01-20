/**
 * Image processing coordinator for the unified processor architecture.
 *
 * TypeScript port of src/processors/image/coordinator.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * This processor runs all image preprocessing steps in sequence.
 */

import type * as cv from '@techstark/opencv-js';
import { Processor, ProcessingContext } from '../base';
import { ImageUtils } from '../../utils/ImageUtils';
import { Logger } from '../../utils/logger';

const logger = new Logger('PreprocessingProcessor');

/**
 * Processor that runs all image preprocessing steps.
 *
 * This processor:
 * 1. Creates a copy of the template layout
 * 2. Resizes images to processing dimensions
 * 3. Runs all preprocessors in sequence (they now implement Processor interface directly)
 * 4. Optionally resizes to output dimensions
 */
export class PreprocessingProcessor extends Processor {
  private tuningConfig: any;

  /**
   * Initialize the preprocessing processor.
   *
   * @param template - The template containing preprocessors and configuration
   */
  constructor(template: any) {
    super();
    // Note: Python stores self.template = template but doesn't use it
    // We skip storing it to avoid unused variable warnings
    this.tuningConfig = template.tuningConfig || template.tuning_config;
  }

  /**
   * Get the name of this processor.
   *
   * @returns Processor name
   */
  getName(): string {
    return 'Preprocessing';
  }

  /**
   * Execute all preprocessing steps.
   *
   * @param context - Processing context with input images
   * @returns Updated context with preprocessed images and updated template
   */
  process(context: ProcessingContext): ProcessingContext {
    logger.debug(`Starting ${this.getName()} processor`);

    // Get a copy of the template layout for mutation
    // Note: In browser context, template structure might be different
    // This is a simplified version that works with the current template structure
    const templateLayout = context.template.templateLayout || context.template.template_layout;

    if (!templateLayout) {
      logger.warn('No template layout found, skipping preprocessing');
      return context;
    }

    // Get processing image shape
    const processingImageShape = templateLayout.processingImageShape ||
                                 templateLayout.processing_image_shape;

    let grayImage = context.grayImage;
    let coloredImage = context.coloredImage;

    // Resize to conform to common preprocessor input requirements
    if (processingImageShape) {
      grayImage = ImageUtils.resizeToShape(processingImageShape, grayImage) as cv.Mat;

      if (this.tuningConfig.outputs?.colored_outputs_enabled) {
        coloredImage = ImageUtils.resizeToShape(processingImageShape, coloredImage) as cv.Mat;
      }
    }

    const showPreprocessorsDiff = this.tuningConfig.outputs?.show_preprocessors_diff || {};

    // Update context for preprocessors
    context.grayImage = grayImage;
    context.coloredImage = coloredImage;

    // Get preprocessors list
    const preProcessors = templateLayout.preProcessors ||
                         templateLayout.pre_processors ||
                         [];

    // Run preprocessors in sequence using their process() method
    for (const preProcessor of preProcessors) {
      const preProcessorName = preProcessor.getName ? preProcessor.getName() :
                               preProcessor.get_name ? preProcessor.get_name() :
                               'Unknown';

      // Show Before Preview (browser version - could emit event or callback)
      if (showPreprocessorsDiff[preProcessorName]) {
        logger.debug(`Before ${preProcessorName}: ${context.filePath}`);
        // Note: In browser, InteractionUtils.show would be replaced with
        // canvas rendering or event emission. This is a placeholder.
        // Users can implement their own visualization by listening to events.
      }

      // Process using unified interface - preprocessors now implement process(context)
      context = preProcessor.process(context);

      // Show After Preview
      if (showPreprocessorsDiff[preProcessorName]) {
        logger.debug(`After ${preProcessorName}: ${context.filePath}`);
      }
    }

    // Resize to output requirements if specified
    const outputImageShape = templateLayout.outputImageShape ||
                            templateLayout.output_image_shape;

    if (outputImageShape) {
      context.grayImage = ImageUtils.resizeToShape(outputImageShape, context.grayImage) as cv.Mat;

      if (this.tuningConfig.outputs?.colored_outputs_enabled) {
        context.coloredImage = ImageUtils.resizeToShape(
          outputImageShape,
          context.coloredImage
        ) as cv.Mat;
      }
    }

    logger.debug(`Completed ${this.getName()} processor`);

    return context;
  }
}

