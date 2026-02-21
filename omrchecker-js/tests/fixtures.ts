// Auto-generated test fixtures
// Converted from pytest fixtures in conftest.py

/**
 * Mock template fixture
 */
export const mockTemplate = {
  path: '/test/template.json',
  fieldBlocks: {},
  preProcessors: [],
  tuningConfig: {
    thresholding: {
      gammaLow: 0.7,
      minGapTwoBubbles: 30,
    },
    outputs: {
      showImageLevel: 0,
    },
  },
};

/**
 * Minimal args fixture
 */
export const minimalArgs = {
  debug: false,
  outputMode: 'default',
  setLayout: false,
};

/**
 * Minimal template JSON fixture
 */
export const minimalTemplateJson = {
  templateDimensions: [1000, 800],
  bubbleDimensions: [20, 20],
  fieldBlocks: {},
  preProcessors: [],
};

/**
 * Minimal evaluation JSON fixture
 */
export const minimalEvaluationJson = {
  sourceType: 'local',
  options: {
    questionsInOrder: ['q1', 'q2'],
    answersInOrder: ['A', 'B'],
  },
  markingSchemes: {
    DEFAULT: {
      correct: 4,
      incorrect: -1,
      unmarked: 0,
    },
  },
};

// Add more fixtures as needed
