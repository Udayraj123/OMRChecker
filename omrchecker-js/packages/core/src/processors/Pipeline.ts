/**
 * Simplified processing pipeline using unified Processor interface.
 *
 * TypeScript port of src/processors/pipeline.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * This pipeline provides a clean, testable interface for processing
 * OMR images through multiple processors with a unified interface:
 * 1. Preprocessing (rotation, cropping, filtering)
 * 2. Alignment (feature-based alignment)
 * 3. ReadOMR (detection & interpretation)
 */

import type * as cv from '@techstark/opencv-js';
import { Processor, ProcessingContext, createProcessingContext } from './base';
import { Logger } from '../utils/logger';

const logger = new Logger('ProcessingPipeline');

/**
 * Configuration options for the pipeline.
 */
export interface PipelineConfig {
  /**
   * Path to ML model for bubble detection (optional)
   */
  mlModelPath?: string;

  /**
   * Path to ML model for field block detection (optional)
   */
  fieldBlockModelPath?: string;

  /**
   * Enable ML-based field block detection
   */
  useFieldBlockDetection?: boolean;

  /**
   * Enable ML-based shift detection
   */
  enableShiftDetection?: boolean;

  /**
   * Enable training data collection
   */
  collectTrainingData?: boolean;

  /**
   * Confidence threshold for training data collection
   */
  confidenceThreshold?: number;

  /**
   * Directory for training data export
   */
  trainingDataDir?: string;
}

/**
 * Simplified pipeline that orchestrates processors.
 *
 * Benefits:
 * - All processors use the same interface
 * - Easy to test each processor independently
 * - Simple to extend with new processors
 * - Type-safe ProcessingContext
 * - Consistent error handling
 */
export class ProcessingPipeline {
  private template: any; // Template type (avoiding circular dependencies)
  private config: PipelineConfig;
  private processors: Processor[] = [];

  /**
   * Initialize the pipeline with a template.
   *
   * @param template - The template containing all configuration and runners
   * @param config - Optional pipeline configuration
   */
  constructor(template: any, config: PipelineConfig = {}) {
    this.template = template;
    this.config = config;

    // Initialize default processors
    this.initializeProcessors();
  }

  /**
   * Initialize all processors in the correct order.
   */
  private initializeProcessors(): void {
    // Note: Actual processor implementations will be added as they are ported
    // For now, this is a placeholder structure that matches the Python implementation

    logger.info('Initializing processing pipeline');

    // 1. Preprocessing (image filters, rotation, cropping)
    // this.processors.push(new PreprocessingProcessor(this.template));

    // 2. Alignment (feature-based alignment to template)
    // this.processors.push(new AlignmentProcessor(this.template));

    // 3. Optional: ML Field Block Detection (Stage 1)
    if (this.config.useFieldBlockDetection && this.config.fieldBlockModelPath) {
      logger.info('ML Field Block Detection (Stage 1) would be enabled');
      // TODO: Add ML field block detector when ported
      // this.processors.push(new MLFieldBlockDetector(...));

      // Optional: Shift Detection
      if (this.config.enableShiftDetection) {
        logger.info('ML-based shift detection would be enabled');
        // TODO: Add shift detection processor when ported
        // this.processors.push(new ShiftDetectionProcessor(...));
      }
    }

    // 4. Traditional + ML Bubble Detection (Stage 2)
    // this.processors.push(new ReadOMRProcessor(this.template, this.config.mlModelPath));

    // 5. Optional: Training Data Collector
    if (this.config.collectTrainingData) {
      this.addTrainingDataCollector();
    }

    logger.debug(`Initialized ${this.processors.length} processors`);
  }

  /**
   * Add training data collector processor.
   */
  private addTrainingDataCollector(): void {
    try {
      const confidenceThreshold = this.config.confidenceThreshold ?? 0.85;

      logger.info(
        `Training data collection enabled (confidence threshold: ${confidenceThreshold})`
      );

      // TODO: Add training data collector when ported
      // Will use this.config.trainingDataDir when implemented
      // this.processors.push(new TrainingDataCollector(this.config.trainingDataDir ?? 'outputs/training_data'));
    } catch (error) {
      logger.warn(`Failed to add training data collector: ${error}`);
    }
  }

  /**
   * Process a single OMR file through all processors.
   *
   * @param filePath - Path to the file being processed
   * @param grayImage - Grayscale input image
   * @param coloredImage - Colored input image
   * @returns ProcessingContext containing all results
   */
  async processFile(
    filePath: string,
    grayImage: cv.Mat,
    coloredImage: cv.Mat
  ): Promise<ProcessingContext> {
    logger.info(`Starting pipeline for file: ${filePath}`);

    // Create initial context
    let context = createProcessingContext(filePath, grayImage, coloredImage, this.template);

    // Execute each processor in sequence
    for (const processor of this.processors) {
      const processorName = processor.getName();
      logger.debug(`Executing processor: ${processorName}`);

      try {
        const result = processor.process(context);
        // Handle both sync and async processors
        context = result instanceof Promise ? await result : result;
      } catch (error) {
        logger.error(`Error in processor ${processorName}: ${error}`);
        throw error;
      }
    }

    logger.info(`Completed pipeline for file: ${filePath}`);

    return context;
  }

  /**
   * Add a custom processor to the pipeline.
   *
   * This allows for extensibility - users can add their own processors
   * for custom processing requirements.
   *
   * @param processor - The processor to add to the pipeline
   */
  addProcessor(processor: Processor): void {
    this.processors.push(processor);
    logger.debug(`Added processor: ${processor.getName()}`);
  }

  /**
   * Remove a processor from the pipeline by name.
   *
   * @param processorName - Name of the processor to remove
   */
  removeProcessor(processorName: string): void {
    const initialLength = this.processors.length;
    this.processors = this.processors.filter((p) => p.getName() !== processorName);

    if (this.processors.length < initialLength) {
      logger.debug(`Removed processor: ${processorName}`);
    }
  }

  /**
   * Get the names of all processors in the pipeline.
   *
   * @returns List of processor names
   */
  getProcessorNames(): string[] {
    return this.processors.map((processor) => processor.getName());
  }

  /**
   * Get all processors in the pipeline.
   *
   * @returns Array of processors
   */
  getProcessors(): Processor[] {
    return [...this.processors];
  }

  /**
   * Clear all processors from the pipeline.
   */
  clearProcessors(): void {
    this.processors = [];
    logger.debug('Cleared all processors');
  }

  /**
   * Get processor by name.
   *
   * @param name - Name of the processor to retrieve
   * @returns Processor instance or undefined
   */
  getProcessorByName(name: string): Processor | undefined {
    return this.processors.find((p) => p.getName() === name);
  }
}

