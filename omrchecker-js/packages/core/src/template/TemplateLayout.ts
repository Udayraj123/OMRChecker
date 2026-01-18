/**
 * Template layout class with methods for template manipulation.
 *
 * Port of Python TemplateLayout class from src/processors/layout/template_layout.py.
 * Wraps the TemplateLayout interface and adds methods for copying, preprocessing,
 * alignment setup, and other template operations.
 */

import * as cv from '@techstark/opencv-js';
import { OMRCheckerError } from '../core/exceptions';
import { FieldDetectionType } from '../processors/constants';
import { FieldBlock } from '../processors/layout/fieldBlock/base';
import type { Field } from '../processors/layout/field/base';
import { BUILTIN_BUBBLE_FIELD_TYPES } from '../utils/constants';
import { ImageUtils } from '../utils/image';
import { Logger } from '../utils/logger';
import { parseFields, alphanumericalSortKey } from '../utils/parsing';
import type {
  TemplateConfig,
  BubbleFieldType,
  AlignmentConfig,
  OutputColumnsConfig,
} from './types';
import type { TemplateLayoutData } from './TemplateLoader';

const logger = new Logger('TemplateLayout');

/**
 * Template layout class with methods for template manipulation.
 *
 * This class wraps the TemplateLayout interface and provides methods
 * for copying, preprocessing, alignment setup, and validation.
 */
export class TemplateLayout {
  public templateDimensions: [number, number];
  public bubbleDimensions: [number, number];
  public fieldBlocks: FieldBlock[];
  public allFields: Field[];
  public allFieldDetectionTypes: string[];
  public allParsedLabels: Set<string>;
  public fieldBlocksOffset: [number, number];
  public globalEmptyValue: string;
  public bubbleFieldTypesData: Record<string, BubbleFieldType>;
  public outputColumns: string[] = [];
  public nonCustomLabels: Set<string> = new Set();
  public customLabels: Record<string, string[]> = {};
  public preProcessors: any[] = []; // TODO: Type this properly
  public processingImageShape: [number, number];
  public outputImageShape?: [number, number];
  public alignment: {
    margins?: { top?: number; bottom?: number; left?: number; right?: number };
    maxDisplacement?: number;
    referenceImage?: string;
    reference_image_path?: string | null;
    gray_alignment_image?: cv.Mat;
    colored_alignment_image?: cv.Mat;
    [key: string]: any;
  } = {};

  /**
   * Create a new TemplateLayout instance.
   *
   * @param layout - Template layout data from TemplateLoader
   * @param config - Template configuration
   * @param tuningConfig - Tuning configuration
   */
  constructor(
    layout: TemplateLayoutData,
    config: TemplateConfig,
    _tuningConfig?: any
  ) {
    this.templateDimensions = layout.templateDimensions;
    this.bubbleDimensions = layout.bubbleDimensions;
    this.fieldBlocks = layout.fieldBlocks;
    this.allFields = layout.allFields;
    this.allFieldDetectionTypes = layout.allFieldDetectionTypes;
    this.allParsedLabels = new Set(layout.allParsedLabels);
    this.fieldBlocksOffset = layout.fieldBlocksOffset;
    this.globalEmptyValue = layout.globalEmptyValue;
    this.bubbleFieldTypesData = layout.bubbleFieldTypesData;

    // Set processing image shape (defaults to template dimensions)
    const [pageWidth, pageHeight] = this.templateDimensions;
    this.processingImageShape =
      config.processingImageShape || ([pageHeight, pageWidth] as [number, number]);

    // Parse output columns
    if (config.outputColumns) {
      this.parseOutputColumns(config.outputColumns);
    }

    // Setup preprocessors
    if (config.preProcessors) {
      this.setupPreProcessors(config.preProcessors, ''); // TODO: Pass relative_dir
    }

    // Parse custom labels
    if (config.customLabels) {
      this.parseCustomLabels(config.customLabels);
    }

    // Fill output columns if empty
    if (this.outputColumns.length === 0 && config.outputColumns) {
      this.fillOutputColumns(
        Array.from(this.nonCustomLabels),
        Object.keys(this.customLabels),
        config.outputColumns
      );
    }

    // Validate template columns
    this.validateTemplateColumns(
      Array.from(this.nonCustomLabels),
      Object.keys(this.customLabels)
    );

    // Setup alignment
    if (config.alignment) {
      this.setupAlignment(config.alignment, ''); // TODO: Pass relative_dir
    }
  }

  /**
   * Get list of files to exclude from processing.
   *
   * @returns Array of file paths to exclude
   */
  getExcludeFiles(): string[] {
    const excludedFiles: string[] = [];
    if (this.alignment.reference_image_path) {
      excludedFiles.push(this.alignment.reference_image_path);
    }
    return excludedFiles;
  }

  /**
   * Create a shallow copy of the template layout for shifting operations.
   *
   * Deep copies field blocks (which can be mutated by processors).
   *
   * @returns Shallow copy of template layout with deep-copied field blocks
   */
  getCopyForShifting(): TemplateLayout {
    // Create shallow copy
    const templateLayout = Object.assign(
      Object.create(Object.getPrototypeOf(this)),
      this
    );

    // Deep copy field blocks (they can be mutated)
    templateLayout.fieldBlocks = this.fieldBlocks.map((block) =>
      block.getCopyForShifting()
    );

    return templateLayout;
  }

  /**
   * Apply preprocessors to images.
   *
   * Note: This method is maintained for backward compatibility but uses
   * the unified processor interface internally.
   *
   * @param filePath - Path of the file being processed
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image
   * @param tuningConfig - Tuning configuration
   * @returns Processed images and updated template layout
   */
  applyPreprocessors(
    _filePath: string,
    grayImage: cv.Mat,
    coloredImage: cv.Mat,
    tuningConfig?: any
  ): [cv.Mat, cv.Mat, TemplateLayout] {
    const nextTemplateLayout = this.getCopyForShifting();

    // Reset shifts in the copied template layout
    nextTemplateLayout.resetAllShifts();

    // Resize to conform to common preprocessor input requirements
    const [resizedGray] = ImageUtils.resizeToShape(
      nextTemplateLayout.processingImageShape,
      grayImage
    );
    let processedGrayImage = resizedGray;

    let processedColoredImage = coloredImage;
    if (tuningConfig?.outputs?.coloredOutputsEnabled) {
      const [resizedColored] = ImageUtils.resizeToShape(
        nextTemplateLayout.processingImageShape,
        coloredImage
      );
      processedColoredImage = resizedColored;
    }

    // Run preprocessors in sequence
    // TODO: Implement full preprocessor pipeline
    // For now, just return resized images
    logger.debug('Preprocessors would be applied here');

    const templateLayout = nextTemplateLayout;

    if (templateLayout.outputImageShape) {
      // Resize to output requirements
      const [resizedGrayOut] = ImageUtils.resizeToShape(
        templateLayout.outputImageShape,
        processedGrayImage
      );
      processedGrayImage = resizedGrayOut;
      if (tuningConfig?.outputs?.coloredOutputsEnabled) {
        const [resizedColoredOut] = ImageUtils.resizeToShape(
          templateLayout.outputImageShape,
          processedColoredImage
        );
        processedColoredImage = resizedColoredOut;
      }
    }

    return [processedGrayImage, processedColoredImage, templateLayout];
  }

  /**
   * Parse output columns configuration.
   *
   * @param outputColumns - Output columns configuration
   */
  parseOutputColumns(outputColumns: OutputColumnsConfig): void {
    const customOrder = outputColumns.customOrder || [];
    const sortType = outputColumns.sortType;

    // Make sure sort_type is set to CUSTOM if output columns are custom
    if (customOrder.length > 0 && sortType && sortType !== 'CUSTOM') {
      logger.fatal(
        `Custom output columns are passed but sort_type is not CUSTOM: ${sortType}. Please set sortType to CUSTOM in outputColumns.`
      );
      throw new OMRCheckerError(
        `Invalid sort type: ${sortType} for custom columns`,
        { sort_type: sortType }
      );
    }

    this.outputColumns = parseFields('Output Columns', customOrder);
  }

  /**
   * Setup preprocessors from configuration.
   *
   * @param preProcessorsObject - Array of preprocessor configurations
   * @param relativeDir - Relative directory for preprocessor resources
   */
  setupPreProcessors(_preProcessorsObject: any[], _relativeDir: string): void {
    // TODO: Implement preprocessor setup
    // This requires PROCESSOR_MANAGER which may not be ported yet
    this.preProcessors = [];
    logger.debug('Preprocessor setup would be implemented here');
  }

  /**
   * Parse custom bubble field types.
   *
   * @param customBubbleFieldTypes - Custom bubble field types configuration
   */
  parseCustomBubbleFieldTypes(
    customBubbleFieldTypes?: Record<string, BubbleFieldType>
  ): void {
    if (!customBubbleFieldTypes) {
      // Convert readonly arrays to mutable arrays
      this.bubbleFieldTypesData = Object.fromEntries(
        Object.entries(BUILTIN_BUBBLE_FIELD_TYPES).map(([k, v]) => [
          k,
          { bubbleValues: [...v.bubbleValues], direction: v.direction },
        ])
      ) as Record<string, BubbleFieldType>;
    } else {
      this.bubbleFieldTypesData = {
        ...Object.fromEntries(
          Object.entries(BUILTIN_BUBBLE_FIELD_TYPES).map(([k, v]) => [
            k,
            { bubbleValues: [...v.bubbleValues], direction: v.direction },
          ])
        ),
        ...customBubbleFieldTypes,
      } as Record<string, BubbleFieldType>;
    }
  }

  /**
   * Validate field blocks configuration.
   *
   * @param fieldBlocksObject - Field blocks configuration
   */
  validateFieldBlocks(fieldBlocksObject: Record<string, any>): void {
    for (const [blockName, fieldBlockObject] of Object.entries(fieldBlocksObject)) {
      // TODO: Check for validations if any for OCR
      if (
        fieldBlockObject.fieldDetectionType === FieldDetectionType.BUBBLES_THRESHOLD
      ) {
        const bubbleFieldType = fieldBlockObject.bubbleFieldType;
        if (!this.bubbleFieldTypesData[bubbleFieldType]) {
          logger.fatal(
            `Cannot find definition for ${bubbleFieldType} in customBubbleFieldTypes`
          );
          throw new OMRCheckerError(
            `Invalid bubble field type: ${bubbleFieldType} in block ${blockName}. Have you defined customBubbleFieldTypes?`,
            {
              bubble_field_type: bubbleFieldType,
              block_name: blockName,
            }
          );
        }
      }

      const fieldLabels = fieldBlockObject.fieldLabels;
      if (fieldLabels.length > 1 && !fieldBlockObject.labelsGap) {
        logger.fatal(
          `More than one fieldLabels(${fieldLabels}) provided, but labelsGap not present for block ${blockName}`
        );
        throw new OMRCheckerError(
          `More than one fieldLabels provided, but labelsGap not present for block ${blockName}`,
          {
            field_labels: fieldLabels,
            block_name: blockName,
          }
        );
      }
    }
  }

  /**
   * Setup alignment configuration.
   *
   * @param alignmentObject - Alignment configuration
   * @param relativeDir - Relative directory for alignment reference image
   */
  setupAlignment(alignmentObject: AlignmentConfig, _relativeDir: string): void {
    this.alignment = {
      ...alignmentObject,
      margins: alignmentObject.margins,
      reference_image_path: null,
    };

    const relativePath = alignmentObject.referenceImage;

    if (relativePath) {
      // TODO: Resolve path properly
      this.alignment.reference_image_path = relativePath; // Path(relativeDir, relativePath)

      // TODO: Load and preprocess alignment images
      // This requires ImageUtils.read_image_util which may need to be ported
      logger.debug('Alignment image loading would be implemented here');
    }
  }

  /**
   * Parse custom labels configuration.
   *
   * @param customLabelsObject - Custom labels configuration
   */
  parseCustomLabels(customLabelsObject: Record<string, string[]>): void {
    const allParsedCustomLabels = new Set<string>();
    this.customLabels = {};

    for (const [customLabel, labelStrings] of Object.entries(customLabelsObject)) {
      const parsedLabels = parseFields(`Custom Label: ${customLabel}`, labelStrings);
      const parsedLabelsSet = new Set(parsedLabels);
      this.customLabels[customLabel] = parsedLabels;

      const missingCustomLabels = Array.from(parsedLabelsSet).filter(
        (label) => !this.allParsedLabels.has(label)
      );

      if (missingCustomLabels.length > 0) {
        logger.fatal(
          `For '${customLabel}', Missing labels - ${missingCustomLabels.join(', ')}`
        );
        throw new OMRCheckerError(
          `Missing field block label(s) in the given template for ${missingCustomLabels.join(', ')} from '${customLabel}'`,
          {
            custom_label: customLabel,
            missing_labels: missingCustomLabels,
          }
        );
      }

      // Check for overlapping labels
      const overlap = Array.from(parsedLabelsSet).filter((label) =>
        allParsedCustomLabels.has(label)
      );
      if (overlap.length > 0) {
        logger.fatal(
          `field strings overlap for labels: ${labelStrings.join(', ')} and existing custom labels: ${Array.from(allParsedCustomLabels).join(', ')}`
        );
        throw new OMRCheckerError(
          `The field strings for custom label '${customLabel}' overlap with other existing custom labels`,
          {
            custom_label: customLabel,
            label_strings: labelStrings,
          }
        );
      }

      parsedLabels.forEach((label) => allParsedCustomLabels.add(label));
    }

    this.nonCustomLabels = new Set(
      Array.from(this.allParsedLabels).filter(
        (label) => !allParsedCustomLabels.has(label)
      )
    );
  }

  /**
   * Get concatenated OMR response with custom labels.
   *
   * @param rawOmrResponse - Raw OMR response by field label
   * @returns Concatenated OMR response
   */
  getConcatenatedOmrResponse(
    rawOmrResponse: Record<string, string>
  ): Record<string, string> {
    const concatenatedOmrResponse: Record<string, string> = {};

    // Add custom labels (concatenated)
    for (const [fieldLabel, concatenateKeys] of Object.entries(this.customLabels)) {
      const customLabel = concatenateKeys.map((k) => rawOmrResponse[k] || '').join('');
      concatenatedOmrResponse[fieldLabel] = customLabel;
    }

    // Add non-custom labels (as-is)
    for (const fieldLabel of this.nonCustomLabels) {
      concatenatedOmrResponse[fieldLabel] = rawOmrResponse[fieldLabel] || '';
    }

    return concatenatedOmrResponse;
  }

  /**
   * Fill output columns if empty.
   *
   * @param nonCustomColumns - Non-custom column labels
   * @param allCustomColumns - All custom column labels
   * @param outputColumns - Output columns configuration
   */
  fillOutputColumns(
    nonCustomColumns: string[],
    allCustomColumns: string[],
    outputColumns: OutputColumnsConfig
  ): void {
    const allTemplateColumns = [...nonCustomColumns, ...allCustomColumns];
    const sortType = outputColumns.sortType || 'ALPHANUMERIC';
    const sortOrder = outputColumns.sortOrder || 'ASC';
    const reverse = sortOrder === 'DESC';

    if (sortType === 'ALPHANUMERIC') {
      this.outputColumns = [...allTemplateColumns].sort((a, b) => {
        const keyA = alphanumericalSortKey(a);
        const keyB = alphanumericalSortKey(b);
        const comparison = keyA < keyB ? -1 : keyA > keyB ? 1 : 0;
        return reverse ? -comparison : comparison;
      });
    } else {
      this.outputColumns = [...allTemplateColumns].sort((a, b) => {
        const comparison = a.localeCompare(b);
        return reverse ? -comparison : comparison;
      });
    }
  }

  /**
   * Validate template columns configuration.
   *
   * @param nonCustomColumns - Non-custom column labels
   * @param allCustomColumns - All custom column labels
   */
  validateTemplateColumns(
    nonCustomColumns: string[],
    allCustomColumns: string[]
  ): void {
    const outputColumnsSet = new Set(this.outputColumns);
    const allCustomColumnsSet = new Set(allCustomColumns);

    const missingOutputColumns = Array.from(outputColumnsSet).filter(
      (col) => !allCustomColumnsSet.has(col) && !this.allParsedLabels.has(col)
    );

    if (missingOutputColumns.length > 0) {
      logger.fatal(`Missing output columns: ${missingOutputColumns.join(', ')}`);
      throw new OMRCheckerError(
        'Some columns are missing in the field blocks for the given output columns',
        { missing_output_columns: missingOutputColumns }
      );
    }

    const allTemplateColumnsSet = new Set([...nonCustomColumns, ...allCustomColumns]);
    const missingLabelColumns = Array.from(allTemplateColumnsSet).filter(
      (col) => !outputColumnsSet.has(col)
    );

    if (missingLabelColumns.length > 0) {
      logger.warn(
        `Some label columns are not covered in the given output columns: ${missingLabelColumns.join(', ')}`
      );
    }
  }

  /**
   * Reset all shifts for all field blocks and fields.
   */
  resetAllShifts(): void {
    for (const fieldBlock of this.fieldBlocks) {
      fieldBlock.resetAllShifts();
    }
  }

  /**
   * Convert template layout to JSON (for serialization).
   *
   * @returns JSON representation of template layout
   */
  toJSON(): Record<string, any> {
    return {
      template_dimensions: this.templateDimensions,
      field_blocks: this.fieldBlocks.map((block) => block.toJSON()),
    };
  }
}

