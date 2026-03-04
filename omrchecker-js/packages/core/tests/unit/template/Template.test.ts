import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { Template } from '../../../src/template/Template';
import { FieldBlock } from '../../../src/template/FieldBlock';
import { BubbleField } from '../../../src/template/BubbleField';
import { BubblesScanBox } from '../../../src/template/BubblesScanBox';

const TEMPLATE_PATH = join(
  __dirname,
  '../../../../../..',
  'samples/1-mobile-camera/template.json',
);

describe('Template (sample1 mobile-camera)', () => {
  let template: Template;

  beforeAll(() => {
    const jsonString = readFileSync(TEMPLATE_PATH, 'utf-8');
    template = Template.fromJSONString(jsonString);
  });

  it('parses templateDimensions correctly', () => {
    expect(template.templateDimensions).toEqual([1846, 1500]);
  });

  it('parses bubbleDimensions correctly', () => {
    expect(template.bubbleDimensions).toEqual([40, 40]);
  });

  it('has a non-empty fieldBlocks array', () => {
    expect(template.fieldBlocks.length).toBeGreaterThan(0);
  });

  it('has a non-empty allFields array', () => {
    expect(template.allFields.length).toBeGreaterThan(0);
  });

  it('all field blocks are FieldBlock instances', () => {
    for (const fb of template.fieldBlocks) {
      expect(fb).toBeInstanceOf(FieldBlock);
    }
  });

  it('all fields are BubbleField instances', () => {
    for (const field of template.allFields) {
      expect(field).toBeInstanceOf(BubbleField);
    }
  });

  it('each field has scanBoxes with correct structure', () => {
    for (const field of template.allFields) {
      expect(field.scanBoxes.length).toBeGreaterThan(0);
      for (const scanBox of field.scanBoxes) {
        expect(scanBox).toBeInstanceOf(BubblesScanBox);
        expect(typeof scanBox.x).toBe('number');
        expect(typeof scanBox.y).toBe('number');
        expect(typeof scanBox.bubbleValue).toBe('string');
        expect(scanBox.bubbleValue.length).toBeGreaterThan(0);
      }
    }
  });

  it('parses the Roll field block with 9 fields', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    expect(rollBlock).toBeDefined();
    expect(rollBlock!.parsedFieldLabels).toHaveLength(9);
    expect(rollBlock!.parsedFieldLabels[0]).toBe('roll1');
    expect(rollBlock!.parsedFieldLabels[8]).toBe('roll9');
    expect(rollBlock!.fields).toHaveLength(9);
  });

  it('Roll block uses QTYPE_INT (10 digit values per field)', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    expect(rollBlock).toBeDefined();
    // QTYPE_INT has 10 bubble values (0..9)
    for (const field of rollBlock!.fields) {
      expect(field.scanBoxes).toHaveLength(10);
    }
  });

  it('Roll block scan boxes have correct bubble values', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    const field = rollBlock!.fields[0];
    const bubbleValues = field.scanBoxes.map((sb) => sb.bubbleValue);
    expect(bubbleValues).toEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
  });

  it('MCQ_Block_Q1 block has 4 fields', () => {
    const mcqBlock = template.fieldBlocks.find((fb) => fb.name === 'MCQ_Block_Q1');
    expect(mcqBlock).toBeDefined();
    expect(mcqBlock!.parsedFieldLabels).toEqual(['q1', 'q2', 'q3', 'q4']);
    expect(mcqBlock!.fields).toHaveLength(4);
  });

  it('MCQ_Block_Q1 fields have 4 scan boxes each (QTYPE_MCQ4)', () => {
    const mcqBlock = template.fieldBlocks.find((fb) => fb.name === 'MCQ_Block_Q1');
    for (const field of mcqBlock!.fields) {
      expect(field.scanBoxes).toHaveLength(4);
      const values = field.scanBoxes.map((sb) => sb.bubbleValue);
      expect(values).toEqual(['A', 'B', 'C', 'D']);
    }
  });

  it('MCQ_Block_Q1 fields have horizontal direction', () => {
    const mcqBlock = template.fieldBlocks.find((fb) => fb.name === 'MCQ_Block_Q1');
    for (const field of mcqBlock!.fields) {
      expect(field.direction).toBe('horizontal');
    }
  });

  it('Roll block fields have vertical direction', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    for (const field of rollBlock!.fields) {
      expect(field.direction).toBe('vertical');
    }
  });

  it('MCQ_Block_Q1 scan boxes have incrementing Y coordinates (horizontal direction = labels stack vertically)', () => {
    // For horizontal direction: v=1, so labels gap applies to Y axis
    const mcqBlock = template.fieldBlocks.find((fb) => fb.name === 'MCQ_Block_Q1');
    const fields = mcqBlock!.fields;
    // Each subsequent field should have a larger Y origin than the previous
    for (let i = 1; i < fields.length; i++) {
      expect(fields[i].origin[1]).toBeGreaterThan(fields[i - 1].origin[1]);
    }
  });

  it('Roll block fields have incrementing X coordinates (vertical direction = labels stack horizontally)', () => {
    // For vertical direction: v=0, so labels gap applies to X axis
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    const fields = rollBlock!.fields;
    for (let i = 1; i < fields.length; i++) {
      expect(fields[i].origin[0]).toBeGreaterThan(fields[i - 1].origin[0]);
    }
  });

  it('Roll block scan boxes have incrementing Y coordinates (vertical direction = bubbles stack on Y)', () => {
    // For vertical direction: h=1, so bubbles gap applies to Y axis
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    const scanBoxes = rollBlock!.fields[0].scanBoxes;
    for (let i = 1; i < scanBoxes.length; i++) {
      expect(scanBoxes[i].y).toBeGreaterThan(scanBoxes[i - 1].y);
    }
  });

  it('MCQ scan boxes have incrementing X coordinates (horizontal direction = bubbles stack on X)', () => {
    // For horizontal direction: h=0, so bubbles gap applies to X axis
    const mcqBlock = template.fieldBlocks.find((fb) => fb.name === 'MCQ_Block_Q1');
    const scanBoxes = mcqBlock!.fields[0].scanBoxes;
    for (let i = 1; i < scanBoxes.length; i++) {
      expect(scanBoxes[i].x).toBeGreaterThan(scanBoxes[i - 1].x);
    }
  });

  it('Roll block first field first scan box has correct origin (225, 282)', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    const firstScanBox = rollBlock!.fields[0].scanBoxes[0];
    expect(firstScanBox.x).toBe(225);
    expect(firstScanBox.y).toBe(282);
  });

  it('Roll block second field is offset by labelsGap (58) on X axis', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    const field0 = rollBlock!.fields[0];
    const field1 = rollBlock!.fields[1];
    expect(field1.origin[0] - field0.origin[0]).toBe(58);
  });

  it('Roll block scan boxes within one field are offset by bubblesGap (46) on Y axis', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    const scanBoxes = rollBlock!.fields[0].scanBoxes;
    expect(scanBoxes[1].y - scanBoxes[0].y).toBe(46);
  });

  it('MCQ_Block_Q1 scan boxes within one field are offset by bubblesGap (59) on X axis', () => {
    const mcqBlock = template.fieldBlocks.find((fb) => fb.name === 'MCQ_Block_Q1');
    const scanBoxes = mcqBlock!.fields[0].scanBoxes;
    expect(scanBoxes[1].x - scanBoxes[0].x).toBe(59);
  });

  it('Medium block uses CUSTOM_MEDIUM type with E/H bubble values', () => {
    const mediumBlock = template.fieldBlocks.find((fb) => fb.name === 'Medium');
    expect(mediumBlock).toBeDefined();
    const values = mediumBlock!.fields[0].scanBoxes.map((sb) => sb.bubbleValue);
    expect(values).toEqual(['E', 'H']);
  });

  it('has correct allParsedLabels (no duplicates)', () => {
    // Should have unique labels from all blocks
    expect(template.allParsedLabels.size).toBeGreaterThan(0);
    // No duplicate labels
    const labelsArray = [...template.allParsedLabels];
    expect(labelsArray.length).toBe(new Set(labelsArray).size);
  });

  it('has customLabels parsed from template.json', () => {
    expect(template.customLabels).toBeDefined();
    expect('Roll' in template.customLabels).toBe(true);
    // Roll custom label combines Medium + roll1..9
    expect(template.customLabels['Roll']).toContain('Medium');
    expect(template.customLabels['Roll']).toContain('roll1');
  });

  it('has preProcessorsConfig with 2 entries', () => {
    expect(template.preProcessorsConfig).toHaveLength(2);
    expect(template.preProcessorsConfig[0].name).toBe('CropPage');
    expect(template.preProcessorsConfig[1].name).toBe('CropOnMarkers');
  });

  it('scan box names follow the pattern fieldLabel_bubbleValue', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    const firstField = rollBlock!.fields[0];
    const firstScanBox = firstField.scanBoxes[0];
    expect(firstScanBox.name).toBe('roll1_0'); // field=roll1, bubbleValue=0
  });

  it('field ids follow the pattern blockName::fieldLabel', () => {
    const rollBlock = template.fieldBlocks.find((fb) => fb.name === 'Roll');
    expect(rollBlock!.fields[0].id).toBe('Roll::roll1');
  });

  it('field blocks have valid bounding box dimensions', () => {
    for (const fb of template.fieldBlocks) {
      const [w, h] = fb.boundingBoxDimensions;
      expect(w).toBeGreaterThan(0);
      expect(h).toBeGreaterThan(0);
    }
  });

  it('outputColumns are sorted and include all non-custom and custom labels', () => {
    expect(template.outputColumns.length).toBeGreaterThan(0);
    // Should include custom labels like Roll, q5, q6...
    expect(template.outputColumns).toContain('Roll');
    expect(template.outputColumns).toContain('q5');
  });
});

describe('Template.fromJSON', () => {
  it('creates a template from a plain object', () => {
    const json = {
      templateDimensions: [500, 500] as [number, number],
      bubbleDimensions: [20, 20] as [number, number],
      fieldBlocks: {
        TestBlock: {
          origin: [10, 10] as [number, number],
          fieldLabels: ['q1..3'],
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          bubbleFieldType: 'QTYPE_MCQ4',
          labelsGap: 50,
          bubblesGap: 30,
        },
      },
    };

    const template = Template.fromJSON(json);
    expect(template.templateDimensions).toEqual([500, 500]);
    expect(template.fieldBlocks).toHaveLength(1);
    expect(template.allFields).toHaveLength(3);
    expect(template.allFields[0].scanBoxes).toHaveLength(4);
  });

  it('uses globalEmptyVal correctly', () => {
    const json = {
      templateDimensions: [500, 500] as [number, number],
      bubbleDimensions: [20, 20] as [number, number],
      emptyValue: 'X',
      fieldBlocks: {
        TestBlock: {
          origin: [10, 10] as [number, number],
          fieldLabels: ['q1'],
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          bubbleFieldType: 'QTYPE_MCQ4',
          bubblesGap: 30,
        },
      },
    };

    const template = Template.fromJSON(json);
    expect(template.globalEmptyVal).toBe('X');
    expect(template.allFields[0].emptyValue).toBe('X');
  });

  it('applies fieldBlocksOffset to all field origins', () => {
    const json = {
      templateDimensions: [1000, 1000] as [number, number],
      bubbleDimensions: [20, 20] as [number, number],
      fieldBlocksOffset: [100, 50] as [number, number],
      fieldBlocks: {
        TestBlock: {
          origin: [10, 10] as [number, number],
          fieldLabels: ['q1'],
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          bubbleFieldType: 'QTYPE_MCQ4',
          bubblesGap: 30,
        },
      },
    };

    const template = Template.fromJSON(json);
    // Origin should be [10+100, 10+50] = [110, 60]
    expect(template.fieldBlocks[0].origin).toEqual([110, 60]);
  });

  it('merges custom bubble field types with builtins', () => {
    const json = {
      templateDimensions: [500, 500] as [number, number],
      bubbleDimensions: [20, 20] as [number, number],
      customBubbleFieldTypes: {
        MY_TYPE: {
          bubbleValues: ['X', 'Y', 'Z'],
          direction: 'horizontal' as const,
        },
      },
      fieldBlocks: {
        TestBlock: {
          origin: [10, 10] as [number, number],
          fieldLabels: ['q1'],
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          bubbleFieldType: 'MY_TYPE',
          bubblesGap: 30,
        },
      },
    };

    const template = Template.fromJSON(json);
    expect(template.allFields[0].scanBoxes).toHaveLength(3);
    const values = template.allFields[0].scanBoxes.map((sb) => sb.bubbleValue);
    expect(values).toEqual(['X', 'Y', 'Z']);
  });

  it('throws for unknown bubble field type', () => {
    const json = {
      templateDimensions: [500, 500] as [number, number],
      bubbleDimensions: [20, 20] as [number, number],
      fieldBlocks: {
        TestBlock: {
          origin: [10, 10] as [number, number],
          fieldLabels: ['q1'],
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          bubbleFieldType: 'NONEXISTENT_TYPE',
          bubblesGap: 30,
        },
      },
    };

    expect(() => Template.fromJSON(json)).toThrow();
  });

  it('throws for overlapping field labels across blocks', () => {
    const json = {
      templateDimensions: [1000, 1000] as [number, number],
      bubbleDimensions: [20, 20] as [number, number],
      fieldBlocks: {
        Block1: {
          origin: [10, 10] as [number, number],
          fieldLabels: ['q1..3'],
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          bubbleFieldType: 'QTYPE_MCQ4',
          labelsGap: 50,
          bubblesGap: 30,
        },
        Block2: {
          origin: [10, 400] as [number, number],
          fieldLabels: ['q2..4'],
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          bubbleFieldType: 'QTYPE_MCQ4',
          labelsGap: 50,
          bubblesGap: 30,
        },
      },
    };

    expect(() => Template.fromJSON(json)).toThrow();
  });
});
