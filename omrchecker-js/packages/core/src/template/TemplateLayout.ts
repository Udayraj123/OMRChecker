/**
 * Template layout class with methods for template manipulation.
 *
 * Port of Python TemplateLayout class from src/processors/layout/template_layout.py.
 * Wraps the TemplateLayout interface and adds methods for copying, preprocessing,
 * alignment setup, and other template operations.
 */

import { OMRCheckerError } from '../core/exceptions';
import { FieldDetectionType } from '../processors/constants';
import { FieldBlock } from '../processors/layout/fieldBlock/base';
import type { Field } from '../processors/layout/field/base';
import { createProcessingContext, Processor } from '../processors/base';
import { ImageTemplatePreprocessor } from '../processors/image/base';
import { PROCESSOR_MANAGER } from '../processors/image/processorManager';
import { BUILTIN_BUBBLE_FIELD_TYPES } from '../utils/constants';
import { ImageUtils } from '../utils/ImageUtils';
import { InteractionUtils } from '../utils/InteractionUtils';
import { Logger } from '../utils/logger';
import { parseFields, alphanumericalSortKey, defaultDump } from '../utils/parsing';
import { SaveImageOps } from '../utils/SaveImageOps';
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
  public preProcessors: (ImageTemplatePreprocessor | Processor)[] = [];
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

    // Setup alignment (async - image loading happens in background)
    // Note: In browser environment, alignment image loading is async
    // Alignment images will be available after the promise resolves
    if (config.alignment) {
      this.setupAlignment(config.alignment, '', _tuningConfig).catch((error) => {
        logger.warn(`Failed to setup alignment: ${error}`);
      });
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
  async applyPreprocessors(
    _filePath: string,
    grayImage: cv.Mat,
    coloredImage: cv.Mat,
    tuningConfig?: any
  ): Promise<[cv.Mat, cv.Mat, TemplateLayout]> {
    const nextTemplateLayout = this.getCopyForShifting();

    // Reset shifts in the copied template layout
    nextTemplateLayout.resetAllShifts();

    // Resize to conform to common preprocessor input requirements
    const resizedGrayResult = ImageUtils.resizeToShape(
      nextTemplateLayout.processingImageShape,
      grayImage
    );
    // resizeToShape returns Mat | Mat[] - handle both cases
    let processedGrayImage = Array.isArray(resizedGrayResult)
      ? resizedGrayResult[0]
      : resizedGrayResult;

    let processedColoredImage = coloredImage;
    if (tuningConfig?.outputs?.colored_outputs_enabled) {
      const resizedColoredResult = ImageUtils.resizeToShape(
        nextTemplateLayout.processingImageShape,
        coloredImage
      );
      processedColoredImage = Array.isArray(resizedColoredResult)
        ? resizedColoredResult[0]
        : resizedColoredResult;
    }

    // Run preprocessors in sequence using unified processor interface
    const showPreprocessorsDiff =
      tuningConfig?.outputs?.show_preprocessors_diff || {};

    let currentTemplateLayout = nextTemplateLayout;

    for (const preProcessor of currentTemplateLayout.preProcessors) {
      const preProcessorName = preProcessor.getName();

      // Show Before Preview
      if (showPreprocessorsDiff[preProcessorName]) {
        InteractionUtils.show(
          `Before ${preProcessorName}: ${_filePath}`,
          tuningConfig?.outputs?.colored_outputs_enabled
            ? processedColoredImage
            : processedGrayImage,
          {
            title: `Before ${preProcessorName}`,
            resizeToFit: true,
          }
        );
      }

      // Apply filter using unified processor interface
      const context = createProcessingContext(
        _filePath,
        processedGrayImage,
        processedColoredImage,
        currentTemplateLayout
      );
      const processResult = preProcessor.process(context);

      // Handle both sync and async processors
      // For now, we assume sync processing (async would require making this method async)
      // If a processor returns a Promise, we'll need to await it
      const updatedContext =
        processResult instanceof Promise ? await processResult : processResult;

      // Extract results from context
      processedGrayImage = updatedContext.grayImage;
      processedColoredImage = updatedContext.coloredImage;
      currentTemplateLayout = updatedContext.template as TemplateLayout;

      // Show After Preview
      if (showPreprocessorsDiff[preProcessorName]) {
        InteractionUtils.show(
          `After ${preProcessorName}: ${_filePath}`,
          tuningConfig?.outputs?.colored_outputs_enabled
            ? processedColoredImage
            : processedGrayImage,
          {
            title: `After ${preProcessorName}`,
            resizeToFit: true,
          }
        );
      }
    }

    const templateLayout = currentTemplateLayout;

    let finalGrayImage = processedGrayImage;
    let finalColoredImage = processedColoredImage;

    if (templateLayout.outputImageShape) {
      // Resize to output requirements
      const resizedGrayOutResult = ImageUtils.resizeToShape(
        templateLayout.outputImageShape,
        processedGrayImage
      );
      finalGrayImage = Array.isArray(resizedGrayOutResult)
        ? resizedGrayOutResult[0]
        : resizedGrayOutResult;
      if (tuningConfig?.outputs?.colored_outputs_enabled) {
        const resizedColoredOutResult = ImageUtils.resizeToShape(
          templateLayout.outputImageShape,
          processedColoredImage
        );
        finalColoredImage = Array.isArray(resizedColoredOutResult)
          ? resizedColoredOutResult[0]
          : resizedColoredOutResult;
      }
    }

    return [finalGrayImage, finalColoredImage, templateLayout];
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
   * Port of Python's setup_pre_processors method.
   *
   * @param preProcessorsObject - Array of preprocessor configurations
   * @param relativeDir - Relative directory for preprocessor resources
   * @param saveImageOps - SaveImageOps instance for debug image saving
   */
  setupPreProcessors(
    preProcessorsObject: any[],
    relativeDir: string,
    saveImageOps?: SaveImageOps
  ): void {
    // Load image preprocessors
    this.preProcessors = [];

    for (const preProcessorConfig of preProcessorsObject) {
      const processorName = preProcessorConfig.name;

      if (!processorName) {
        logger.warn('Preprocessor configuration missing name, skipping');
        continue;
      }

      // Get processor factory from manager
      const processorFactory = PROCESSOR_MANAGER[processorName];

      if (!processorFactory) {
        logger.warn(
          `Unknown processor name: ${processorName}. Supported processors: ${Object.keys(PROCESSOR_MANAGER).join(', ')}`
        );
        continue;
      }

      // Create saveImageOps if not provided
      // Use the SaveImageOps class instance or create a new instance
      let imageOps: SaveImageOps;
      if (saveImageOps) {
        imageOps = saveImageOps;
      } else {
        // Create a new SaveImageOps instance with minimal config
        const minimalTuningConfig = { outputs: { save_image_level: 0 } } as any;
        imageOps = new SaveImageOps(minimalTuningConfig);
      }

      // Instantiate processor using factory
      const preProcessorInstance = processorFactory(
        preProcessorConfig.options || {},
        relativeDir,
        imageOps,
        this.processingImageShape
      );

      this.preProcessors.push(preProcessorInstance);
      logger.debug(`Loaded preprocessor: ${processorName}`);
    }

    logger.info(`Setup ${this.preProcessors.length} preprocessors`);
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
          { bubble_values: [...v.bubble_values], direction: v.direction },
        ])
      ) as Record<string, BubbleFieldType>;
    } else {
      this.bubbleFieldTypesData = {
        ...Object.fromEntries(
          Object.entries(BUILTIN_BUBBLE_FIELD_TYPES).map(([k, v]) => [
            k,
            { bubble_values: [...v.bubble_values], direction: v.direction },
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
   * Port of Python's setup_alignment method.
   *
   * @param alignmentObject - Alignment configuration
   * @param relativeDir - Relative directory for alignment reference image
   * @param tuningConfig - Tuning configuration (optional, for image loading)
   */
  async setupAlignment(
    alignmentObject: AlignmentConfig,
    relativeDir: string,
    tuningConfig?: any
  ): Promise<void> {
    this.alignment = {
      ...alignmentObject,
      margins: alignmentObject.margins,
      reference_image_path: null,
    };

    const relativePath = alignmentObject.referenceImage;

    if (relativePath) {
      // Resolve path: combine relativeDir and relativePath
      // In browser environment, paths are typically URLs or data URLs
      // For file paths, we combine them with a separator
      const resolvedPath = relativeDir
        ? `${relativeDir}/${relativePath}`.replace(/\/+/g, '/')
        : relativePath;
      this.alignment.reference_image_path = resolvedPath;

      // Load and preprocess alignment images
      // Note: In browser, this requires the image to be loaded asynchronously
      try {
        // For browser: assume relativePath is a URL or data URL
        // In Node.js: could be a file path
        const imageSource: File | Blob | string = resolvedPath;

        // Load images using ImageUtils.readImageUtil
        // Note: readImageUtil is async in TypeScript (browser environment)
        const coloredOutputsEnabled =
          tuningConfig?.outputs?.colored_outputs_enabled || false;
        const [grayAlignmentImage, coloredAlignmentImage] =
          await ImageUtils.readImageUtil(imageSource, coloredOutputsEnabled);

        // Preprocess alignment images using apply_preprocessors
        // Create a copy of template layout for preprocessing
        const nextTemplateLayout = this.getCopyForShifting();
        nextTemplateLayout.resetAllShifts();

        // Apply preprocessors to alignment images
        const [
          processedGrayAlignmentImage,
          processedColoredAlignmentImage,
          _updatedTemplate,
        ] = await this.applyPreprocessors(
          resolvedPath,
          grayAlignmentImage,
          coloredAlignmentImage || grayAlignmentImage.clone(),
          tuningConfig
        );

        // Store preprocessed alignment images
        this.alignment.gray_alignment_image = processedGrayAlignmentImage;
        this.alignment.colored_alignment_image = processedColoredAlignmentImage;

        logger.debug('Alignment images loaded and preprocessed');
      } catch (error) {
        logger.warn(
          `Failed to load alignment image from ${resolvedPath}: ${error}`
        );
        // Continue without alignment images - alignment will be skipped
        this.alignment.reference_image_path = null;
      }
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
   * Parse and add a field block to the template layout.
   *
   * Port of Python's parse_and_add_field_block method.
   *
   * @param blockName - Name of the field block
   * @param fieldBlockObject - Field block configuration object
   * @returns Created FieldBlock instance
   */
  parseAndAddFieldBlock(
    blockName: string,
    fieldBlockObject: any
  ): FieldBlock {
    const filledFieldBlockObject = this.prefillFieldBlock(fieldBlockObject);
    const blockInstance = new FieldBlock(
      blockName,
      filledFieldBlockObject,
      this.fieldBlocksOffset
    );

    // TODO: support custom field types like Barcode and OCR
    this.fieldBlocks.push(blockInstance);
    this.validateParsedFieldBlock(
      filledFieldBlockObject.fieldLabels,
      blockInstance
    );

    // Update allFields and allFieldDetectionTypes
    this.allFields.push(...blockInstance.fields);
    if (!this.allFieldDetectionTypes.includes(blockInstance.fieldDetectionType)) {
      this.allFieldDetectionTypes.push(blockInstance.fieldDetectionType);
    }

    return blockInstance;
  }

  /**
   * Prefill field block with default values.
   *
   * Port of Python's prefill_field_block method.
   *
   * @param fieldBlockObject - Field block configuration object
   * @returns Filled field block configuration
   */
  prefillFieldBlock(fieldBlockObject: any): any {
    const filledFieldBlockObject = { ...fieldBlockObject };

    if (
      fieldBlockObject.fieldDetectionType ===
      FieldDetectionType.BUBBLES_THRESHOLD
    ) {
      const bubbleFieldType = fieldBlockObject.bubbleFieldType;
      const fieldTypeData = this.bubbleFieldTypesData[bubbleFieldType];
      Object.assign(filledFieldBlockObject, {
        bubbleFieldType: bubbleFieldType,
        emptyValue: this.globalEmptyValue,
        bubbleDimensions: this.bubbleDimensions,
        ...fieldTypeData,
      });
    } else if (
      fieldBlockObject.fieldDetectionType === FieldDetectionType.OCR ||
      fieldBlockObject.fieldDetectionType === FieldDetectionType.BARCODE_QR
    ) {
      Object.assign(filledFieldBlockObject, {
        emptyValue: this.globalEmptyValue,
        labelsGap: 0,
      });
    }

    return filledFieldBlockObject;
  }

  /**
   * Validate parsed field block.
   *
   * Port of Python's validate_parsed_field_block method.
   *
   * @param fieldLabels - Field labels array
   * @param blockInstance - FieldBlock instance to validate
   */
  validateParsedFieldBlock(
    fieldLabels: string[],
    blockInstance: FieldBlock
  ): void {
    const parsedFieldLabels = blockInstance.parsedFieldLabels;
    const blockName = blockInstance.name;
    const fieldLabelsSet = new Set(parsedFieldLabels);

    // Check for overlap with existing labels
    const overlap = new Set(
      Array.from(fieldLabelsSet).filter((label) =>
        this.allParsedLabels.has(label)
      )
    );

    if (overlap.size > 0) {
      logger.fatal(
        `An overlap found between field string: ${fieldLabels.join(', ')} in block '${blockName}' and existing labels: ${Array.from(this.allParsedLabels).join(', ')}`
      );
      throw new OMRCheckerError(
        `The field strings for field block ${blockName} overlap with other existing fields: ${Array.from(overlap).join(', ')}`,
        {
          block_name: blockName,
          field_labels: fieldLabels,
          overlap: Array.from(overlap),
        }
      );
    }

    // Update all parsed labels
    parsedFieldLabels.forEach((label) => this.allParsedLabels.add(label));

    // Validate bounding box is within template dimensions
    const [pageWidth, pageHeight] = this.templateDimensions;

    if (!blockInstance.boundingBoxDimensions || !blockInstance.boundingBoxOrigin) {
      // Bounding box not calculated yet, skip validation
      return;
    }

    const [blockWidth, blockHeight] = blockInstance.boundingBoxDimensions;
    const [blockStartX, blockStartY] = blockInstance.boundingBoxOrigin;

    const blockEndX = blockStartX + blockWidth;
    const blockEndY = blockStartY + blockHeight;

    if (
      blockEndX >= pageWidth ||
      blockEndY >= pageHeight ||
      blockStartX < 0 ||
      blockStartY < 0
    ) {
      throw new OMRCheckerError(
        `Overflowing field block '${blockName}' with origin [${blockStartX}, ${blockStartY}] and dimensions [${blockWidth}, ${blockHeight}] in template with dimensions [${pageWidth}, ${pageHeight}]`,
        {
          block_name: blockName,
          bounding_box_origin: blockInstance.boundingBoxOrigin,
          bounding_box_dimensions: blockInstance.boundingBoxDimensions,
          template_dimensions: this.templateDimensions,
        }
      );
    }
  }

  /**
   * Convert template layout to string representation.
   *
   * Port of Python's __str__ method.
   *
   * @returns String representation (template path)
   */
  toString(): string {
    // In TypeScript, we don't have a path property on TemplateLayout
    // Return a descriptive string instead
    return `TemplateLayout(${this.fieldBlocks.length} blocks, ${this.allFields.length} fields)`;
  }

  /**
   * Convert template layout to JSON (for serialization).
   *
   * Port of Python's to_json method.
   * Uses defaultDump to handle serialization of complex objects.
   *
   * @returns JSON representation of template layout
   */
  toJSON(): Record<string, any> {
    return {
      template_dimensions: defaultDump(this.templateDimensions),
      field_blocks: defaultDump(this.fieldBlocks),
      // Note: Following Python's to_json, we only include these keys:
      // - template_dimensions
      // - field_blocks
      // Other properties (bubble_dimensions, global_empty_val, etc.) are not included
      // as they are considered local props that are overridden
    };
  }
}

