/**
 * OMRProcessor - Main orchestrator for OMR processing pipeline.
 *
 * This class coordinates the complete OMR workflow:
 * 1. Template loading and validation
 * 2. Image preprocessing
 * 3. Bubble detection
 * 4. Answer evaluation
 * 5. Result generation
 *
 * Port of Python's entry point logic and template processing.
 */

import * as cv from '@techstark/opencv-js';
import { ProcessingPipeline } from '../processors/Pipeline';
import { PreprocessingProcessor } from '../processors/image/coordinator';
import { AlignmentProcessor } from '../processors/alignment/AlignmentProcessor';
import {
  BubblesFieldDetection,
  BubbleFieldDetectionResult,
} from '../processors/detection';
import { GlobalThreshold, type ThresholdConfig } from '../processors/threshold/GlobalThreshold';
import { EvaluationProcessor } from '../processors/evaluation/EvaluationProcessor';
import { TemplateLoader, type ParsedTemplate } from '../template/TemplateLoader';
import { type TemplateConfig } from '../template/types';
import { Logger } from '../utils/logger';

const logger = new Logger('OMRProcessor');

/**
 * Configuration options for OMR processing
 */
export interface OMRProcessorConfig {
  /** Enable debug mode with additional logging */
  debug?: boolean;
  /** Save intermediate images for debugging */
  saveIntermediateImages?: boolean;
  /** Output directory for results */
  outputDirectory?: string;
  /** Custom threshold configuration */
  thresholdConfig?: {
    defaultThreshold?: number;
    minJump?: number;
  };
}

/**
 * Result of processing a single OMR sheet
 */
export interface OMRSheetResult {
  /** File path of the processed image */
  filePath: string;
  /** Detected answers by field ID */
  responses: Record<string, string | null>;
  /** Multi-marked fields */
  multiMarkedFields: string[];
  /** Empty fields (no answer detected) */
  emptyFields: string[];
  /** Evaluation score (if answer key provided) */
  score?: number;
  /** Maximum possible score */
  maxScore?: number;
  /** Detailed field results using proper Python architecture */
  fieldResults: Record<string, BubbleFieldDetectionResult>;
  /** Processing time in milliseconds */
  processingTimeMs: number;
  /** Any warnings or errors during processing */
  warnings: string[];
  /** Detection statistics */
  statistics: {
    totalFields: number;
    answeredFields: number;
    unansweredFields: number;
    multiMarkedFields: number;
    avgConfidence: number;
  };
}

/**
 * Main OMR processing class.
 *
 * Example usage:
 * ```typescript
 * const processor = new OMRProcessor(templateConfig);
 * const result = await processor.processImage(image, 'sheet001.jpg');
 * console.log('Detected answers:', result.responses);
 * ```
 */
export class OMRProcessor {
  private template: ParsedTemplate;
  private pipeline: ProcessingPipeline;
  private thresholdStrategy: GlobalThreshold;
  private evaluationProcessor: EvaluationProcessor | null = null;
  private config: OMRProcessorConfig;

  /**
   * Create a new OMR processor.
   *
   * @param templateConfig - Template configuration
   * @param config - Processing configuration
   * @param answerKey - Optional answer key for evaluation
   */
  constructor(
    templateConfig: TemplateConfig,
    processorConfig: OMRProcessorConfig = {},
    _answerKey?: Record<string, string>
  ) {
    logger.info('Initializing OMR Processor');

    this.config = {
      debug: false,
      saveIntermediateImages: false,
      ...processorConfig,
    };

    // Load and parse template
    this.template = TemplateLoader.loadFromJSON(templateConfig);
    logger.info(`Template loaded: ${this.template.fields.size} fields`);

    // Initialize threshold strategy
    this.thresholdStrategy = new GlobalThreshold();

    // Initialize processing pipeline
    this.pipeline = new ProcessingPipeline(this.template);

    // Add preprocessing processor
    this.pipeline.addProcessor(new PreprocessingProcessor(this.template));
    logger.debug('Added preprocessing processor');

    // Add alignment processor if configured
    if (templateConfig.alignment) {
      this.pipeline.addProcessor(new AlignmentProcessor(this.template));
      logger.debug('Added alignment processor');
    }

    // Initialize evaluation processor if answer key provided
    if (_answerKey) {
      // Note: EvaluationProcessor needs evaluation config, not just template
      // For now, we'll skip automatic initialization
      // Evaluation can be done separately using the detected responses
      logger.debug('Answer key provided - evaluation support pending');
    }

    logger.info('OMR Processor initialized successfully');
  }

  /**
   * Process a single OMR sheet image.
   *
   * @param image - Grayscale input image
   * @param filePath - File path for logging/debugging
   * @param coloredImage - Optional colored version for visualization
   * @returns Processing result with detected answers
   */
  async processImage(
    image: cv.Mat,
    filePath: string,
    coloredImage?: cv.Mat
  ): Promise<OMRSheetResult> {
    const startTime = Date.now();
    logger.info(`Processing image: ${filePath}`);

    const warnings: string[] = [];
    const responses: Record<string, string | null> = {};
    const fieldResults: Record<string, BubbleFieldDetectionResult> = {};
    const multiMarkedFields: string[] = [];
    const emptyFields: string[] = [];

    try {
      // Step 1: Run preprocessing pipeline
      logger.debug('Running preprocessing pipeline');
      const context = await this.pipeline.processFile(
        filePath,
        image,
        coloredImage || image.clone()
      );

      // Step 2: Detect bubbles for each field using proper Python architecture
      if (this.config.debug) {
        logger.debug(`Detecting bubbles for ${this.template.fields.size} fields`);
      }

      for (const [fieldId, field] of this.template.fields) {
        try {
          // Use BubblesFieldDetection following Python architecture
          const detection = new BubblesFieldDetection(
            fieldId,
            fieldId, // fieldLabel
            field.bubbles,
            context.grayImage
          );

          const result = detection.getResult();
          fieldResults[fieldId] = result;

          // Calculate threshold and determine marked bubbles
          const bubbleMeanValues = result.meanValues;
          const defaultThresholdConfig: ThresholdConfig = {
            defaultThreshold: 200,
            minJump: 30,
          };
          const thresholdConfig: ThresholdConfig = {
            ...defaultThresholdConfig,
            ...this.config.thresholdConfig,
          };
          const thresholdResult = this.thresholdStrategy.calculateThreshold(
            bubbleMeanValues,
            thresholdConfig
          );

          // Find marked bubbles (below threshold)
          const markedBubbles = result.bubbleMeans.filter(
            bm => bm.meanValue < thresholdResult.thresholdValue
          );

          // Determine detected answer
          let detectedAnswer: string | null = null;
          let isMultiMarked = false;

          if (markedBubbles.length === 1) {
            detectedAnswer = markedBubbles[0].unitBubble.label;
          } else if (markedBubbles.length > 1) {
            isMultiMarked = true;
            // Take first marked bubble but flag as multi-marked
            detectedAnswer = markedBubbles[0].unitBubble.label;
          }

          // Store detected answer
          if (detectedAnswer) {
            responses[fieldId] = detectedAnswer;
          } else {
            responses[fieldId] = field.emptyValue || null;
            emptyFields.push(fieldId);
          }

          // Track multi-marked fields
          if (isMultiMarked) {
            multiMarkedFields.push(fieldId);
            warnings.push(`Field ${fieldId} has multiple bubbles marked`);
          }

          logger.debug(
            `Field ${fieldId}: ${detectedAnswer || 'EMPTY'} (multi: ${isMultiMarked}, ` +
            `quality: ${result.scanQuality})`
          );
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          logger.error(`Error detecting field ${fieldId}: ${errorMessage}`);
          warnings.push(`Field ${fieldId}: ${errorMessage}`);
          responses[fieldId] = null;
          emptyFields.push(fieldId);
        }
      }

      // Step 3: Evaluate if answer key is available
      let score: number | undefined;
      let maxScore: number | undefined;

      if (this.evaluationProcessor) {
        try {
          logger.debug('Evaluating responses');
          // Note: Evaluation processor needs full implementation
          // For now, we'll skip automatic evaluation
          // The evaluation can be done separately using the responses
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          logger.error(`Error evaluating: ${errorMessage}`);
          warnings.push(`Evaluation error: ${errorMessage}`);
        }
      }

      const processingTimeMs = Date.now() - startTime;
      logger.info(
        `Processing complete: ${filePath} (${processingTimeMs}ms) - ` +
        `${Object.keys(responses).length} fields detected`
      );

      // Calculate statistics
      const answeredFields = Object.values(responses).filter((r) => r !== null).length;
      const totalFields = Object.keys(fieldResults).length;
      let totalConfidence = 0;
      let confidenceCount = 0;

      // Calculate average confidence from scan quality
      for (const result of Object.values(fieldResults)) {
        // Use std deviation as a proxy for confidence
        // Higher std deviation = better scan quality = higher confidence
        const confidence = Math.min(result.stdDeviation / 100, 1.0);
        totalConfidence += confidence;
        confidenceCount++;
      }

      const avgConfidence = confidenceCount > 0 ? totalConfidence / confidenceCount : 0;

      return {
        filePath,
        responses,
        multiMarkedFields,
        emptyFields,
        score,
        maxScore,
        fieldResults,
        processingTimeMs,
        warnings,
        statistics: {
          totalFields,
          answeredFields,
          unansweredFields: emptyFields.length,
          multiMarkedFields: multiMarkedFields.length,
          avgConfidence,
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Fatal error processing ${filePath}: ${errorMessage}`);

      return {
        filePath,
        responses: {},
        multiMarkedFields: [],
        emptyFields: [],
        fieldResults: {},
        processingTimeMs: Date.now() - startTime,
        warnings: [`Fatal error: ${errorMessage}`],
        statistics: {
          totalFields: 0,
          answeredFields: 0,
          unansweredFields: 0,
          multiMarkedFields: 0,
          avgConfidence: 0,
        },
      };
    }
  }

  /**
   * Process multiple OMR sheets in batch.
   *
   * @param images - Array of [image, filePath] tuples
   * @returns Array of processing results
   */
  async processBatch(
    images: Array<[cv.Mat, string, cv.Mat?]>
  ): Promise<OMRSheetResult[]> {
    logger.info(`Processing batch of ${images.length} images`);

    const results: OMRSheetResult[] = [];

    for (const [image, filePath, coloredImage] of images) {
      const result = await this.processImage(image, filePath, coloredImage);
      results.push(result);
    }

    logger.info(`Batch processing complete: ${results.length} images processed`);
    return results;
  }

  /**
   * Get template information.
   *
   * @returns Parsed template
   */
  getTemplate(): ParsedTemplate {
    return this.template;
  }

  /**
   * Get list of field IDs in the template.
   *
   * @returns Array of field IDs
   */
  getFieldIds(): string[] {
    return Array.from(this.template.fields.keys());
  }

  /**
   * Export results to CSV format.
   *
   * @param results - Array of sheet results
   * @returns CSV string
   */
  exportToCSV(results: OMRSheetResult[]): string {
    if (results.length === 0) {
      return '';
    }

    // Get sorted field IDs from template
    const fieldIds = TemplateLoader.getSortedFieldIds(this.template);

    // CSV header
    const header = ['File', ...fieldIds, 'Score', 'Warnings'].join(',');
    const rows: string[] = [header];

    // CSV rows
    for (const result of results) {
      const row: string[] = [result.filePath];

      // Add field responses
      for (const fieldId of fieldIds) {
        const response = result.responses[fieldId];
        row.push(response !== null && response !== undefined ? response : '');
      }

      // Add score
      row.push(
        result.score !== undefined && result.maxScore !== undefined
          ? `${result.score}/${result.maxScore}`
          : ''
      );

      // Add warnings count
      row.push(result.warnings.length > 0 ? `${result.warnings.length} warnings` : '');

      rows.push(row.join(','));
    }

    return rows.join('\n');
  }

  /**
   * Get processing statistics.
   *
   * @param results - Array of sheet results
   * @returns Statistics summary
   */
  getStatistics(results: OMRSheetResult[]): {
    totalSheets: number;
    averageProcessingTime: number;
    totalWarnings: number;
    multiMarkedSheets: number;
    emptyFieldsCount: number;
  } {
    return {
      totalSheets: results.length,
      averageProcessingTime:
        results.reduce((sum, r) => sum + r.processingTimeMs, 0) / results.length || 0,
      totalWarnings: results.reduce((sum, r) => sum + r.warnings.length, 0),
      multiMarkedSheets: results.filter((r) => r.multiMarkedFields.length > 0).length,
      emptyFieldsCount: results.reduce((sum, r) => sum + r.emptyFields.length, 0),
    };
  }
}

