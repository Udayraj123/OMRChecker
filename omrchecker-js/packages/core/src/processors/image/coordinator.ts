/**
 * Image processing coordinator for the unified processor architecture.
 *
 * TypeScript port of src/processors/image/coordinator.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * This processor runs all image preprocessing steps in sequence.
 */

import { Processor, ProcessingContext } from '../base';
import { ImageUtils } from '../../utils/ImageUtils';
import { Logger } from '../../utils/logger';
import { InteractionUtils } from '../../utils/InteractionUtils';

const logger = new Logger('PreprocessingCoordinator');

/**
 * Coordinates all image preprocessing steps in sequence.
 *
 * This is NOT an individual preprocessor. It orchestrates all preprocessors
 * defined in template.templateLayout.pre_processors.
 *
 * Responsibilities:
 * 1. Creates a copy of the template layout (in future)
 * 2. Resizes images to processing dimensions
 * 3. Runs all preprocessors in sequence (they implement Processor interface)
 * 4. Optionally resizes to output dimensions
 * 5. Shows before/after diffs when configured
 */
export class PreprocessingCoordinator extends Processor {
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
    this.tuningConfig = template.tuning_config || {};
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
    const processingImageShape = templateLayout.processing_image_shape;

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
    const preProcessors = templateLayout.pre_processors || [];

    // Run preprocessors in sequence using their process() method
    for (const preProcessor of preProcessors) {
      const preProcessorName = preProcessor.getName();

      // Show Before Preview - display actual images in debug container
      if (showPreprocessorsDiff[preProcessorName]) {
        logger.debug(`Before ${preProcessorName}: ${context.filePath}`);
        
        InteractionUtils.show(
          `Before ${preProcessorName}`,
          context.grayImage,
          { title: `${context.filePath} - Before ${preProcessorName}` }
        );
        
        // Also show colored if enabled
        if (this.tuningConfig.outputs?.colored_outputs_enabled && context.coloredImage) {
          InteractionUtils.show(
            `Before ${preProcessorName} (Color)`,
            context.coloredImage,
            { title: `${context.filePath} - Before ${preProcessorName} (Color)` }
          );
        }
      }

      // Process using unified interface - preprocessors now implement process(context)
      context = preProcessor.process(context);

      // Show After Preview - display actual images in debug container
      if (showPreprocessorsDiff[preProcessorName]) {
        logger.debug(`After ${preProcessorName}: ${context.filePath}`);
        
        InteractionUtils.show(
          `After ${preProcessorName}`,
          context.grayImage,
          { title: `${context.filePath} - After ${preProcessorName}` }
        );
        
        // Also show colored if enabled
        if (this.tuningConfig.outputs?.colored_outputs_enabled && context.coloredImage) {
          InteractionUtils.show(
            `After ${preProcessorName} (Color)`,
            context.coloredImage,
            { title: `${context.filePath} - After ${preProcessorName} (Color)` }
          );
        }
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

