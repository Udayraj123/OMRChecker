/**
 * Comprehensive tests for TemplateLayout class.
 *
 * Tests all methods of TemplateLayout to ensure high coverage.
 * Ported from Python test_template_layout.py
 */

const cv = global.cv;
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TemplateLayout } from '../TemplateLayout';
import { TemplateLoader } from '../TemplateLoader';
import type { TemplateConfig } from '../types';
import { ImageUtils } from '../../utils/ImageUtils';

/**
 * Minimal valid template JSON for testing.
 */
function createMinimalTemplateConfig(): TemplateConfig {
  return {
    templateDimensions: [1000, 800],
    bubbleDimensions: [20, 20],
    emptyValue: '',
    fieldBlocksOffset: [0, 0],
    fieldBlocks: {
      block1: {
        fieldDetectionType: 'BUBBLES_THRESHOLD',
        origin: [100, 100],
        fieldLabels: ['q1', 'q2'],
        bubbleFieldType: 'QTYPE_MCQ4',
        bubblesGap: 30,
        labelsGap: 50,
      },
    },
    preProcessors: [],
    alignment: { margins: { top: 0, right: 0, bottom: 0, left: 0 } },
    customBubbleFieldTypes: {},
    customLabels: {},
    outputColumns: { sortType: 'ALPHANUMERIC', customOrder: [] },
  };
}

describe('TemplateLayout', () => {
  describe('Initialization', () => {
    it('should initialize with minimal template', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(layout.templateDimensions).toEqual([1000, 800]);
      expect(layout.bubbleDimensions).toEqual([20, 20]);
      expect(layout.globalEmptyValue).toBe('');
      expect(layout.fieldBlocks.length).toBe(1);
      expect(layout.allFields.length).toBe(2); // q1 and q2
    });

    it('should initialize with custom processing shape', () => {
      const config = createMinimalTemplateConfig();
      config.processingImageShape = [600, 400];
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(layout.processingImageShape).toEqual([600, 400]);
    });

    it('should initialize with preprocessors', () => {
      const config = createMinimalTemplateConfig();
      config.preProcessors = [{ name: 'GaussianBlur', options: {} }];
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(layout.preProcessors.length).toBe(1);
    });

    it('should initialize with alignment reference image', async () => {
      const config = createMinimalTemplateConfig();
      config.alignment = {
        referenceImage: 'ref_image.jpg',
        margins: { top: 0, right: 0, bottom: 0, left: 0 },
      };

      // Mock image reading
      const mockGrayImage = new cv.Mat(100, 100, cv.CV_8UC1);
      vi.spyOn(ImageUtils, 'readImageUtil').mockResolvedValue([mockGrayImage, null]);

      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Alignment setup is async, wait for it
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(layout.alignment.reference_image_path).toBeDefined();
      mockGrayImage.delete();
    });
  });

  describe('getExcludeFiles', () => {
    it('should return empty array without alignment reference image', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const excluded = layout.getExcludeFiles();
      expect(excluded).toEqual([]);
    });

    it('should return reference image path with alignment', async () => {
      const config = createMinimalTemplateConfig();
      config.alignment = {
        referenceImage: 'ref_image.jpg',
        margins: { top: 0, right: 0, bottom: 0, left: 0 },
      };

      const mockGrayImage = new cv.Mat(100, 100, cv.CV_8UC1);
      vi.spyOn(ImageUtils, 'readImageUtil').mockResolvedValue([mockGrayImage, null]);

      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      await new Promise((resolve) => setTimeout(resolve, 100));

      const excluded = layout.getExcludeFiles();
      expect(excluded.length).toBe(1);
      expect(excluded[0]).toContain('ref_image.jpg');

      mockGrayImage.delete();
    });
  });

  describe('getCopyForShifting', () => {
    it('should create shallow copy', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const copyLayout = layout.getCopyForShifting();

      expect(copyLayout).not.toBe(layout);
      expect(copyLayout.templateDimensions).toEqual(layout.templateDimensions);
      expect(copyLayout.bubbleDimensions).toEqual(layout.bubbleDimensions);
    });

    it('should deep copy field blocks', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const copyLayout = layout.getCopyForShifting();

      expect(copyLayout.fieldBlocks).not.toBe(layout.fieldBlocks);
      expect(copyLayout.fieldBlocks.length).toBe(layout.fieldBlocks.length);

      // Modifying copy should not affect original
      if (copyLayout.fieldBlocks.length > 0) {
        const originalOrigin = [...layout.fieldBlocks[0].origin];
        copyLayout.fieldBlocks[0].origin = [999, 999];
        expect(layout.fieldBlocks[0].origin).toEqual(originalOrigin);
      }
    });
  });

  describe('applyPreprocessors', () => {
    it('should apply no preprocessors', async () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const grayImage = new cv.Mat(800, 1000, cv.CV_8UC1);
      const coloredImage = new cv.Mat(800, 1000, cv.CV_8UC3);

      const [processedGray, processedColored, updatedLayout] =
        await layout.applyPreprocessors('test.jpg', grayImage, coloredImage);

      expect(processedGray).toBeDefined();
      expect(processedColored).toBeDefined();
      expect(updatedLayout).toBeDefined();

      // Delete input images
      grayImage.delete();
      coloredImage.delete();
      
      // Only delete processed images if they're different from inputs
      if (processedGray !== grayImage && !processedGray.isDeleted()) {
        processedGray.delete();
      }
      if (processedColored !== coloredImage && !processedColored.isDeleted()) {
        processedColored.delete();
      }
    });

    it('should apply GaussianBlur preprocessor', async () => {
      const config = createMinimalTemplateConfig();
      config.preProcessors = [{ name: 'GaussianBlur', options: {} }];
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const grayImage = new cv.Mat(800, 1000, cv.CV_8UC1);
      const coloredImage = new cv.Mat(800, 1000, cv.CV_8UC3);

      const [processedGray, processedColored, updatedLayout] =
        await layout.applyPreprocessors('test.jpg', grayImage, coloredImage);

      expect(processedGray).toBeDefined();
      expect(processedColored).toBeDefined();
      expect(updatedLayout).toBeDefined();

      // Delete input images
      grayImage.delete();
      coloredImage.delete();
      
      // Only delete processed images if they're different from inputs
      if (processedGray !== grayImage && !processedGray.isDeleted()) {
        processedGray.delete();
      }
      if (processedColored !== coloredImage && !processedColored.isDeleted()) {
        processedColored.delete();
      }
    });
  });

  describe('parseOutputColumns', () => {
    it('should parse custom sort order', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'CUSTOM',
        customOrder: ['q2', 'q1'],
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(layout.outputColumns).toEqual(['q2', 'q1']);
    });

    it('should parse alphanumeric sort', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'ALPHANUMERIC',
        customOrder: [],
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Should be sorted alphanumerically
      expect(layout.outputColumns).toContain('q1');
      expect(layout.outputColumns).toContain('q2');
    });
  });

  describe('parseCustomBubbleFieldTypes', () => {
    it('should parse custom bubble field types', () => {
      const config = createMinimalTemplateConfig();
      config.customBubbleFieldTypes = {
        CUSTOM_1: {
          bubbleValues: ['A', 'B', 'C', 'D', 'E'],
          direction: 'horizontal',
        },
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(layout.bubbleFieldTypesData['CUSTOM_1']).toBeDefined();
      expect(layout.bubbleFieldTypesData['CUSTOM_1'].bubbleValues).toEqual([
        'A',
        'B',
        'C',
        'D',
        'E',
      ]);
      expect(layout.bubbleFieldTypesData['CUSTOM_1'].direction).toBe('horizontal');
    });
  });

  describe('parseCustomLabels', () => {
    it('should parse custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {
        Combined: ['q1', 'q2'],
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(layout.customLabels['Combined']).toEqual(['q1', 'q2']);
    });

    it('should handle empty custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {};
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(Object.keys(layout.customLabels).length).toBe(0);
    });
  });

  describe('getConcatenatedOmrResponse', () => {
    it('should concatenate OMR response with custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {
        Combined: ['q1', 'q2'],
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const rawOmrResponse = {
        q1: 'A',
        q2: 'B',
      };

      const result = layout.getConcatenatedOmrResponse(rawOmrResponse);
      expect(result).toBeDefined();
    });

    it('should handle empty OMR response', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const rawOmrResponse: Record<string, string> = {};

      const result = layout.getConcatenatedOmrResponse(rawOmrResponse);
      expect(result).toBeDefined();
    });

    it('should handle missing keys in OMR response', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const rawOmrResponse = {
        q1: 'A',
        // q2 is missing
      };

      const result = layout.getConcatenatedOmrResponse(rawOmrResponse);
      expect(result).toBeDefined();
    });
  });

  describe('validateFieldBlocks', () => {
    it('should validate valid field blocks', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Should not throw
      expect(() => {
        layout.validateFieldBlocks();
      }).not.toThrow();
    });
  });

  describe('resetAllShifts', () => {
    it('should reset all shifts', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Apply some shifts
      if (layout.fieldBlocks.length > 0) {
        layout.fieldBlocks[0].origin = [200, 200];
      }

      layout.resetAllShifts();

      // Shifts should be reset
      if (layout.fieldBlocks.length > 0) {
        expect(layout.fieldBlocks[0].origin).toEqual([100, 100]);
      }
    });
  });

  describe('fillOutputColumns', () => {
    it('should auto-fill output columns', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'ALPHANUMERIC',
        customOrder: [],
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Should auto-fill with field labels
      expect(layout.outputColumns.length).toBeGreaterThan(0);
      expect(layout.outputColumns).toContain('q1');
      expect(layout.outputColumns).toContain('q2');
    });

    it('should handle empty columns', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'ALPHANUMERIC',
        customOrder: [],
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Should not be empty after auto-fill
      expect(layout.outputColumns.length).toBeGreaterThan(0);
    });
  });

  describe('validateTemplateColumns', () => {
    it('should validate valid columns', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'CUSTOM',
        customOrder: ['q1', 'q2'],
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Should not throw
      expect(layout).toBeDefined();
    });

    it('should throw error for missing columns', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'CUSTOM',
        customOrder: ['q1', 'q2', 'q99'], // q99 doesn't exist
      };
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);

      // Should throw error during validation
      expect(() => {
        new TemplateLayout(layoutData, config);
      }).toThrow();
    });
  });

  describe('parseAndAddFieldBlock', () => {
    it('should parse and add a new field block', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const initialCount = layout.fieldBlocks.length;
      const newBlock = {
        fieldDetectionType: 'BUBBLES_THRESHOLD',
        origin: [200, 200],
        fieldLabels: ['q3'],
        bubbleFieldType: 'QTYPE_MCQ4',
        bubblesGap: 30,
        labelsGap: 50,
      };

      const blockInstance = layout.parseAndAddFieldBlock('block2', newBlock);

      expect(layout.fieldBlocks.length).toBe(initialCount + 1);
      expect(blockInstance).toBeDefined();
      expect(blockInstance.name).toBe('block2');
    });
  });

  describe('prefillFieldBlock', () => {
    it('should prefill a field block', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const fieldBlockObject = {
        fieldDetectionType: 'BUBBLES_THRESHOLD',
        origin: [200, 200],
        fieldLabels: ['q3'],
        bubbleFieldType: 'QTYPE_MCQ4',
        bubblesGap: 30,
        labelsGap: 50,
      };

      const filled = layout.prefillFieldBlock(fieldBlockObject);

      expect(filled).toBeDefined();
      expect(filled.bubbleDimensions).toBeDefined();
      expect(filled.emptyValue).toBeDefined();
      expect(filled.bubbleFieldType).toBe('QTYPE_MCQ4');
    });
  });

  describe('validateParsedFieldBlock', () => {
    it('should validate valid parsed field block', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Create a mock field block
      const mockFieldBlock = layout.fieldBlocks[0];
      const fieldLabels = ['q1', 'q2'];

      // Should not throw
      expect(() => {
        layout.validateParsedFieldBlock(fieldLabels, mockFieldBlock);
      }).not.toThrow();
    });
  });

  describe('toString', () => {
    it('should return string representation', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const strRepr = layout.toString();

      expect(typeof strRepr).toBe('string');
      expect(strRepr.length).toBeGreaterThan(0);
    });
  });

  describe('toJSON', () => {
    it('should serialize to JSON', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const json = layout.toJSON();

      expect(json).toBeDefined();
      expect(json.template_dimensions).toBeDefined();
      expect(json.field_blocks).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty field blocks', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {};
      // This should throw during loadLayoutFromJSON, but test the error
      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should handle zero template dimensions', () => {
      const config = createMinimalTemplateConfig();
      config.templateDimensions = [0, 0];
      // This should throw during loadLayoutFromJSON
      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should handle zero bubble dimensions', () => {
      const config = createMinimalTemplateConfig();
      config.bubbleDimensions = [0, 0];
      // This should throw during loadLayoutFromJSON
      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should handle empty custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {};
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      expect(Object.keys(layout.customLabels).length).toBe(0);
    });

    it('should handle missing keys in OMR response', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const rawOmrResponse = {
        q1: 'A',
        // q2 is missing
      };

      const result = layout.getConcatenatedOmrResponse(rawOmrResponse);
      expect(result).toBeDefined();
    });

    it('should handle reset all shifts with no fields', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Should not throw
      expect(() => {
        layout.resetAllShifts();
      }).not.toThrow();
    });
  });
});

