/**
 * Alignment Processor for template alignment.
 *
 * TypeScript port of src/processors/alignment/processor.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * This processor performs feature-based alignment if a reference image
 * is provided in the template configuration.
 */

import { Processor, ProcessingContext } from '../base';
import { Logger } from '../../utils/logger';
import { applyTemplateAlignment } from './templateAlignment';

const logger = new Logger('AlignmentProcessor');

/**
 * Processor that applies template alignment to images.
 *
 * This processor performs feature-based alignment if a reference image
 * is provided in the template configuration.
 */
export class AlignmentProcessor extends Processor {
  // @ts-ignore
  private _template: any; // Template type (avoiding circular dependencies)
  private tuningConfig: any;

  /**
   * Initialize the alignment processor.
   *
   * @param template - The template containing alignment configuration
   */
  constructor(template: any) {
    super();
    this._template = template;
    this.tuningConfig = template.tuningConfig || template.tuning_config;
  }

  /**
   * Get the name of this processor.
   *
   * @returns Processor name
   */
  getName(): string {
    return 'Alignment';
  }

  /**
   * Execute alignment on the images.
   *
   * @param context - Processing context with preprocessed images
   * @returns Updated context with aligned images and template
   */
  process(context: ProcessingContext): ProcessingContext {
    logger.debug(`Starting ${this.getName()} processor`);

    const grayImage = context.grayImage;
    const coloredImage = context.coloredImage;
    const template = context.template;

    // Only apply alignment if images are valid and alignment is configured
    const alignment = template.alignment;
    const hasAlignmentImage = alignment?.grayAlignmentImage ||
                              alignment?.gray_alignment_image;

    if (grayImage && hasAlignmentImage) {
      const result = applyTemplateAlignment(
        grayImage,
        coloredImage,
        template,
        this.tuningConfig
      );

      // Update context with aligned images
      context.grayImage = result.grayImage;
      context.coloredImage = result.coloredImage;
      context.template = result.template;
    } else {
      logger.debug('Skipping alignment - no reference image configured');
    }

    logger.debug(`Completed ${this.getName()} processor`);

    return context;
  }
}

