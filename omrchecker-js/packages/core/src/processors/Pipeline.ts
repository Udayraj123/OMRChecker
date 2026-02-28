/**
 * Migrated from Python: src/processors/pipeline.py
 * Agent: Processor-Delta
 * Phase: 1
 *
 * Simplified processing pipeline using unified Processor interface.
 */

import cv from '@techstark/opencv-js';
import { ProcessingContext, Processor, createProcessingContext } from './base';

/**
 * Simplified pipeline that orchestrates processors.
 * 
 * This pipeline provides a clean, testable interface for processing
 * OMR images through multiple processors with a unified interface:
 * 1. Preprocessing (rotation, cropping, filtering)
 * 2. Alignment (feature-based alignment)
 * 3. ReadOMR (detection & interpretation)
 * 
 * Benefits:
 * - All processors use the same interface
 * - Easy to test each processor independently
 * - Simple to extend with new processors
 * - Type-safe ProcessingContext
 * - Consistent error handling
 */
export class ProcessingPipeline {
  private template: any;
  private tuningConfig: any;
  private processors: Processor[];

  /**
   * Initialize the pipeline with a template
   * 
   * @param template - The template containing all configuration and runners
   */
  constructor(template: any) {
    this.template = template;
    this.tuningConfig = template.tuningConfig || template.tuning_config;
    this.processors = [];
  }

  /**
   * Process a single OMR file through all processors
   * 
   * @param filePath - Path to the file being processed
   * @param grayImage - Grayscale input image
   * @param coloredImage - Colored input image
   * @returns ProcessingContext containing all results (omrResponse, metrics, etc.)
   */
  async processFile(
    filePath: string,
    grayImage: cv.Mat,
    coloredImage: cv.Mat
  ): Promise<ProcessingContext> {
    console.log(`Starting pipeline for file: ${filePath}`);

    // Create initial context
    let context = createProcessingContext(
      filePath,
      grayImage,
      coloredImage,
      this.template
    );

    // Execute each processor in sequence
    for (const processor of this.processors) {
      const processorName = processor.getName();
      console.debug(`Executing processor: ${processorName}`);
      
      // Support both sync and async processors
      const result = processor.process(context);
      context = result instanceof Promise ? await result : result;
    }

    console.log(`Completed pipeline for file: ${filePath}`);

    return context;
  }

  /**
   * Add a custom processor to the pipeline
   * 
   * This allows for extensibility - users can add their own processors
   * for custom processing requirements.
   * 
   * @param processor - The processor to add to the pipeline
   */
  addProcessor(processor: Processor): void {
    this.processors.push(processor);
  }

  /**
   * Remove a processor from the pipeline by name
   * 
   * @param processorName - Name of the processor to remove
   */
  removeProcessor(processorName: string): void {
    this.processors = this.processors.filter(
      p => p.getName() !== processorName
    );
  }

  /**
   * Get the names of all processors in the pipeline
   * 
   * @returns List of processor names
   */
  getProcessorNames(): string[] {
    return this.processors.map(processor => processor.getName());
  }

  /**
   * Get the current template
   */
  getTemplate(): any {
    return this.template;
  }

  /**
   * Get the tuning configuration
   */
  getTuningConfig(): any {
    return this.tuningConfig;
  }
}
