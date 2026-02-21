import { Template } from './core/template';
import { Config } from './core/config';
import { processImage } from './processors/pipeline';
import type { ProcessingResult } from './types';

/**
 * Main OMRChecker class - entry point for browser-based OMR processing
 */
export class OMRChecker {
  /**
   * Load a template from JSON
   */
  static async loadTemplate(templatePath: string, config?: Config): Promise<Template> {
    const response = await fetch(templatePath);
    const templateJson = await response.json();
    return new Template(templateJson, config);
  }

  /**
   * Process an image with a template
   */
  static async processImage(
    image: HTMLImageElement | ImageData,
    template: Template,
  ): Promise<ProcessingResult> {
    return processImage(image, template);
  }

  /**
   * Process multiple images in batch
   */
  static async processBatch(
    images: (HTMLImageElement | ImageData)[],
    template: Template,
    options?: { parallel?: boolean; maxWorkers?: number },
  ): Promise<ProcessingResult[]> {
    // TODO: Implement batch processing with Web Workers
    const results: ProcessingResult[] = [];
    for (const image of images) {
      results.push(await this.processImage(image, template));
    }
    return results;
  }
}
