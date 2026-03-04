/**
 * Template class for OMRChecker TypeScript port.
 *
 * Migrated from:
 *   src/processors/layout/template_layout.py (TemplateLayout)
 *   src/schemas/models/template.py (TemplateConfig)
 *
 * Scope: BUBBLES_THRESHOLD field detection only.
 * Excluded: OCR, BARCODE, alignment, drawing, directory handling, ProcessingPipeline.
 */

import { OMRCheckerError } from '../utils/exceptions';
import { BUILTIN_BUBBLE_FIELD_TYPES, BubbleFieldTypeData } from './constants';
import { parseFields } from './parseFields';
import { FieldBlock, FieldBlockJson } from './FieldBlock';
import { BubbleField } from './BubbleField';

/**
 * JSON shape of a preProcessor entry in template.json.
 */
export interface PreProcessorConfig {
  name: string;
  options: Record<string, any>;
}

/**
 * JSON shape of the root template.json object.
 * Supports both camelCase (from JSON) and snake_case (for compatibility).
 */
export interface TemplateJson {
  templateDimensions: [number, number];
  bubbleDimensions: [number, number];
  processingImageShape?: [number, number];
  processing_image_shape?: [number, number];
  fieldBlocksOffset?: [number, number];
  field_blocks_offset?: [number, number];
  emptyValue?: string;
  empty_value?: string;
  customBubbleFieldTypes?: Record<string, { bubbleValues: string[]; direction: string } | { bubble_values: string[]; direction: string }>;
  custom_bubble_field_types?: Record<string, { bubble_values: string[]; direction: string }>;
  customLabels?: Record<string, string[]>;
  custom_labels?: Record<string, string[]>;
  fieldBlocks: Record<string, FieldBlockJson>;
  field_blocks?: Record<string, FieldBlockJson>;
  preProcessors?: PreProcessorConfig[];
  pre_processors?: PreProcessorConfig[];
}

/**
 * Parsed and validated template data model.
 *
 * Ported from Python:
 *   src/processors/layout/template_layout.py::TemplateLayout (setup_layout, prefill_field_block, etc.)
 *   src/processors/layout/template/template.py::Template (re-exports)
 */
export class Template {
  templateDimensions: [number, number];
  processingImageShape: [number, number];
  bubbleDimensions: [number, number];
  globalEmptyVal: string;
  fieldBlocksOffset: [number, number];
  fieldBlocks: FieldBlock[];
  allFields: BubbleField[];
  allParsedLabels: Set<string>;
  customLabels: Record<string, string[]>;
  nonCustomLabels: Set<string>;
  outputColumns: string[];
  preProcessorsConfig: PreProcessorConfig[];
  bubbleFieldTypesData: Record<string, BubbleFieldTypeData>;

  constructor(templateJson: TemplateJson) {
    // Required properties
    this.templateDimensions = templateJson.templateDimensions;
    this.bubbleDimensions = templateJson.bubbleDimensions;

    // Properties with defaults
    const [pageWidth, pageHeight] = this.templateDimensions;
    const processingImageShapeRaw =
      templateJson.processingImageShape ??
      templateJson.processing_image_shape ??
      [pageHeight, pageWidth];
    this.processingImageShape = processingImageShapeRaw as [number, number];

    this.fieldBlocksOffset = (
      templateJson.fieldBlocksOffset ??
      templateJson.field_blocks_offset ??
      [0, 0]
    ) as [number, number];

    this.globalEmptyVal =
      templateJson.emptyValue ?? templateJson.empty_value ?? '';

    this.preProcessorsConfig =
      templateJson.preProcessors ??
      templateJson.pre_processors ??
      [];

    // Custom bubble field types: merge builtins with user-defined types
    this.parseCustomBubbleFieldTypes(
      templateJson.customBubbleFieldTypes ??
      templateJson.custom_bubble_field_types,
    );

    // Custom labels (user-defined output groupings)
    const customLabelsRaw =
      templateJson.customLabels ??
      templateJson.custom_labels ??
      {};

    // Parse field blocks
    const fieldBlocksRaw =
      templateJson.fieldBlocks ??
      templateJson.field_blocks ??
      {};

    // Initialize tracking sets/arrays
    this.allParsedLabels = new Set<string>();
    this.fieldBlocks = [];
    this.allFields = [];

    // Validate field blocks reference valid bubble field types
    this.validateFieldBlocks(fieldBlocksRaw);

    // Build field blocks (enriches each block with type data then constructs)
    this.setupLayout(fieldBlocksRaw);

    // Parse custom labels and compute non-custom labels
    this.parseCustomLabels(customLabelsRaw);

    // Default output columns: alphabetically sorted non-custom + custom labels
    this.outputColumns = this.buildDefaultOutputColumns();
  }

  /**
   * Merge builtin bubble field types with any user-defined custom types.
   *
   * Ported from Python: TemplateLayout.parse_custom_bubble_field_types
   */
  private parseCustomBubbleFieldTypes(
    customBubbleFieldTypes?: Record<string, any>,
  ): void {
    if (!customBubbleFieldTypes || Object.keys(customBubbleFieldTypes).length === 0) {
      this.bubbleFieldTypesData = { ...BUILTIN_BUBBLE_FIELD_TYPES };
      return;
    }

    const converted: Record<string, BubbleFieldTypeData> = {};
    for (const [typeName, typeData] of Object.entries(customBubbleFieldTypes)) {
      // Accept both camelCase (bubbleValues) and snake_case (bubble_values) from JSON
      const bubbleValues: string[] =
        (typeData as any).bubbleValues ?? (typeData as any).bubble_values ?? [];
      const direction: 'vertical' | 'horizontal' =
        ((typeData as any).direction ?? 'horizontal') as 'vertical' | 'horizontal';
      converted[typeName] = { bubble_values: bubbleValues, direction };
    }

    this.bubbleFieldTypesData = {
      ...BUILTIN_BUBBLE_FIELD_TYPES,
      ...converted,
    };
  }

  /**
   * Validate that all field blocks reference known bubble field types.
   *
   * Ported from Python: TemplateLayout.validate_field_blocks
   */
  private validateFieldBlocks(fieldBlocksObject: Record<string, FieldBlockJson>): void {
    for (const [blockName, fieldBlockObject] of Object.entries(fieldBlocksObject)) {
      const fieldDetectionType =
        (fieldBlockObject as any).fieldDetectionType ??
        (fieldBlockObject as any).field_detection_type ??
        'BUBBLES_THRESHOLD';

      if (fieldDetectionType === 'BUBBLES_THRESHOLD') {
        const bubbleFieldType =
          (fieldBlockObject as any).bubbleFieldType ??
          (fieldBlockObject as any).bubble_field_type;

        if (!bubbleFieldType || !(bubbleFieldType in this.bubbleFieldTypesData)) {
          throw new OMRCheckerError(
            `Invalid bubble field type: ${bubbleFieldType} in block ${blockName}. Have you defined customBubbleFieldTypes?`,
            { bubble_field_type: bubbleFieldType, block_name: blockName },
          );
        }

        const fieldLabels: string[] =
          (fieldBlockObject as any).fieldLabels ??
          (fieldBlockObject as any).field_labels ??
          [];
        const labelsGap =
          (fieldBlockObject as any).labelsGap ??
          (fieldBlockObject as any).labels_gap;

        if (fieldLabels.length > 1 && labelsGap == null) {
          throw new OMRCheckerError(
            `More than one fieldLabels(${fieldLabels}) provided, but labels_gap not present for block ${blockName}`,
            { field_labels: fieldLabels, block_name: blockName },
          );
        }
      }
    }
  }

  /**
   * Build all FieldBlock instances from the raw JSON.
   *
   * Ported from Python: TemplateLayout.setup_layout
   */
  private setupLayout(fieldBlocksObject: Record<string, FieldBlockJson>): void {
    for (const [blockName, fieldBlockObject] of Object.entries(fieldBlocksObject)) {
      const blockInstance = this.parseAndAddFieldBlock(blockName, fieldBlockObject);
      this.allFields.push(...blockInstance.fields);
    }
  }

  /**
   * Enrich a raw field block object with type data, then construct a FieldBlock.
   *
   * Ported from Python: TemplateLayout.parse_and_add_field_block + prefill_field_block
   */
  private parseAndAddFieldBlock(
    blockName: string,
    fieldBlockObject: FieldBlockJson,
  ): FieldBlock {
    const enriched = this.prefillFieldBlock(fieldBlockObject);
    const blockInstance = new FieldBlock(blockName, enriched, this.fieldBlocksOffset);

    this.fieldBlocks.push(blockInstance);
    this.validateParsedFieldBlock(fieldBlockObject, blockInstance);

    return blockInstance;
  }

  /**
   * Enrich a field block object with defaults from the template and the bubble field type data.
   *
   * Ported from Python: TemplateLayout.prefill_field_block
   */
  private prefillFieldBlock(
    fieldBlockObject: FieldBlockJson,
  ): FieldBlockJson & Record<string, any> {
    const fieldDetectionType =
      (fieldBlockObject as any).fieldDetectionType ??
      (fieldBlockObject as any).field_detection_type ??
      'BUBBLES_THRESHOLD';

    if (fieldDetectionType !== 'BUBBLES_THRESHOLD') {
      // Non-bubbles types are out of scope; pass through unchanged
      return { ...fieldBlockObject };
    }

    const bubbleFieldType =
      (fieldBlockObject as any).bubbleFieldType ??
      (fieldBlockObject as any).bubble_field_type;

    const fieldTypeData = this.bubbleFieldTypesData[bubbleFieldType];

    // Merge order (matching Python's prefill_field_block):
    // 1. Start with bubble_field_type default
    // 2. Set empty_value and bubble_dimensions from template defaults
    // 3. Override with field_block_object values (user can override direction, etc.)
    // 4. Override with field_type_data (bubble_values, direction from the type definition)
    return {
      bubble_field_type: bubbleFieldType,
      empty_value: this.globalEmptyVal,
      bubble_dimensions: this.bubbleDimensions,
      ...fieldBlockObject,
      ...fieldTypeData,
    };
  }

  /**
   * Validate that the parsed field labels don't overlap with previously seen labels,
   * and that the bounding box doesn't overflow the template dimensions.
   *
   * Ported from Python: TemplateLayout.validate_parsed_field_block
   */
  private validateParsedFieldBlock(
    fieldBlockObjectRaw: FieldBlockJson,
    blockInstance: FieldBlock,
  ): void {
    const { parsedFieldLabels, name: blockName } = blockInstance;
    const fieldLabelsSet = new Set(parsedFieldLabels);

    // Check for overlap with previously parsed labels
    const overlap = [...fieldLabelsSet].filter((l) => this.allParsedLabels.has(l));
    if (overlap.length > 0) {
      const fieldLabels =
        (fieldBlockObjectRaw as any).fieldLabels ??
        (fieldBlockObjectRaw as any).field_labels ??
        [];
      throw new OMRCheckerError(
        `The field strings for field block ${blockName} overlap with other existing fields: ${overlap}`,
        { block_name: blockName, field_labels: fieldLabels, overlap },
      );
    }

    for (const label of fieldLabelsSet) {
      this.allParsedLabels.add(label);
    }

    // Bounding box overflow check
    const [pageWidth, pageHeight] = this.templateDimensions;
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
        `Overflowing field block '${blockName}' with origin ${blockInstance.boundingBoxOrigin} and dimensions ${blockInstance.boundingBoxDimensions} in template with dimensions ${this.templateDimensions}`,
        {
          block_name: blockName,
          bounding_box_origin: blockInstance.boundingBoxOrigin,
          bounding_box_dimensions: blockInstance.boundingBoxDimensions,
          template_dimensions: this.templateDimensions,
        },
      );
    }
  }

  /**
   * Parse custom labels from the template (user-defined output groupings).
   *
   * Ported from Python: TemplateLayout.parse_custom_labels
   */
  private parseCustomLabels(customLabelsObject: Record<string, string[]>): void {
    const allParsedCustomLabels = new Set<string>();
    this.customLabels = {};

    for (const [customLabel, labelStrings] of Object.entries(customLabelsObject)) {
      const parsedLabels = parseFields(`Custom Label: ${customLabel}`, labelStrings);
      const parsedLabelsSet = new Set(parsedLabels);
      this.customLabels[customLabel] = parsedLabels;

      // All referenced labels must exist in the template
      const missingCustomLabels = [...parsedLabelsSet].filter(
        (l) => !this.allParsedLabels.has(l),
      );
      if (missingCustomLabels.length > 0) {
        throw new OMRCheckerError(
          `Missing field block label(s) in the given template for ${missingCustomLabels} from '${customLabel}'`,
          { custom_label: customLabel, missing_labels: missingCustomLabels },
        );
      }

      // Custom labels must not overlap with each other
      const overlap = [...parsedLabelsSet].filter((l) => allParsedCustomLabels.has(l));
      if (overlap.length > 0) {
        throw new OMRCheckerError(
          `The field strings for custom label '${customLabel}' overlap with other existing custom labels`,
          { custom_label: customLabel, label_strings: labelStrings },
        );
      }

      for (const l of parsedLabelsSet) {
        allParsedCustomLabels.add(l);
      }
    }

    this.nonCustomLabels = new Set(
      [...this.allParsedLabels].filter((l) => !allParsedCustomLabels.has(l)),
    );
  }

  /**
   * Build default output columns by sorting non-custom and custom label names alphanumerically.
   *
   * Ported from Python: TemplateLayout.fill_output_columns
   */
  private buildDefaultOutputColumns(): string[] {
    const nonCustomColumns = [...this.nonCustomLabels];
    const allCustomColumns = Object.keys(this.customLabels);
    const allColumns = [...nonCustomColumns, ...allCustomColumns];
    return allColumns.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  }

  resetAllShifts(): void {
    for (const fieldBlock of this.fieldBlocks) {
      fieldBlock.resetAllShifts();
    }
  }

  /**
   * Construct a Template from a parsed JSON object.
   */
  static fromJSON(json: TemplateJson): Template {
    return new Template(json);
  }

  /**
   * Construct a Template from a JSON string.
   */
  static fromJSONString(jsonString: string): Template {
    const json = JSON.parse(jsonString) as TemplateJson;
    return new Template(json);
  }
}
