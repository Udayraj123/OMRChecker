/**
 * Tests for EvaluationProcessor.
 *
 * Tests the evaluation and scoring functionality.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  EvaluationProcessor,
  EvaluationConfig,
  EvaluationConfigForResponse,
  MarkingScheme,
} from '../EvaluationProcessor';
import { createProcessingContext } from '../../base';
import * as cv from '@techstark/opencv-js';

// Mock evaluation config for testing
class MockEvaluationConfigForResponse implements EvaluationConfigForResponse {
  questionsInOrder = ['Q1', 'Q2', 'Q3'];
  private shouldExplain = false;

  getShouldExplainScoring(): boolean {
    return this.shouldExplain;
  }

  getFormattedAnswersSummary(_formatString: string): [string, ...any[]] {
    return ['Correct: 2, Incorrect: 1'];
  }

  prepareAndValidateOmrResponse(
    _concatenatedResponse: Record<string, string>,
    _allowStreak: boolean
  ): void {
    // Mock implementation
  }

  matchAnswerForQuestion(
    _currentScore: number,
    question: string,
    markedAnswer: string
  ): [number, string, any, string] {
    // Simple mock: Q1 and Q2 are correct, Q3 is incorrect
    const isCorrect = question === 'Q1' || question === 'Q2';
    const delta = isCorrect ? 1 : -0.25;
    const verdict = isCorrect ? 'correct' : 'incorrect';

    return [
      delta,
      verdict,
      { answerItem: markedAnswer, answerType: 'single' },
      verdict,
    ];
  }

  getMarkingSchemeForQuestion(_question: string): MarkingScheme {
    return {
      getBonusType: () => null,
    };
  }

  conditionallyPrintExplanation(): void {
    // Mock implementation
  }
}

class MockEvaluationConfig implements EvaluationConfig {
  private shouldReturnConfig = true;

  getEvaluationConfigForResponse(
    _concatenatedResponse: Record<string, string>,
    _filePath: string
  ): EvaluationConfigForResponse | null {
    return this.shouldReturnConfig ? new MockEvaluationConfigForResponse() : null;
  }

  setReturnNull(value: boolean): void {
    this.shouldReturnConfig = !value;
  }
}

describe('EvaluationProcessor', () => {
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;
  let mockTemplate: any;

  beforeEach(() => {
    // Create mock images
    mockGrayImage = new cv.Mat();
    mockColoredImage = new cv.Mat();
    mockTemplate = { name: 'test-template' };
  });

  describe('constructor', () => {
    it('should initialize with evaluation config', () => {
      const config = new MockEvaluationConfig();
      const processor = new EvaluationProcessor(config);

      expect(processor).toBeDefined();
      expect(processor.getName()).toBe('Evaluation');
    });

    it('should initialize with null config', () => {
      const processor = new EvaluationProcessor(null);

      expect(processor).toBeDefined();
    });
  });

  describe('getName', () => {
    it('should return "Evaluation"', () => {
      const processor = new EvaluationProcessor(null);

      expect(processor.getName()).toBe('Evaluation');
    });
  });

  describe('process - with no config', () => {
    it('should skip evaluation when no config provided', () => {
      const processor = new EvaluationProcessor(null);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = { Q1: 'A', Q2: 'B' };

      const result = processor.process(context);

      expect(result.score).toBe(0);
      expect(result.evaluationMeta).toBeNull();
    });
  });

  describe('process - with no matching config', () => {
    it('should skip evaluation when no matching config for response', () => {
      const config = new MockEvaluationConfig();
      config.setReturnNull(true);

      const processor = new EvaluationProcessor(config);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = { Q1: 'A', Q2: 'B' };

      const result = processor.process(context);

      expect(result.score).toBe(0);
      expect(result.evaluationMeta).toBeNull();
    });
  });

  describe('process - with valid config', () => {
    it('should evaluate response and compute score', () => {
      const config = new MockEvaluationConfig();
      const processor = new EvaluationProcessor(config);

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = {
        Q1: 'A', // Correct: +1
        Q2: 'B', // Correct: +1
        Q3: 'D', // Incorrect: -0.25
      };

      const result = processor.process(context);

      expect(result.score).toBe(1.75); // 1 + 1 - 0.25
      expect(result.evaluationMeta).toBeDefined();
      expect(result.evaluationMeta?.score).toBe(1.75);
    });

    it('should populate questions metadata', () => {
      const config = new MockEvaluationConfig();
      const processor = new EvaluationProcessor(config);

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = {
        Q1: 'A',
        Q2: 'B',
        Q3: 'D',
      };

      const result = processor.process(context);

      expect(result.evaluationMeta).toBeDefined();
      expect(result.evaluationMeta?.questionsMeta).toBeDefined();
      expect(Object.keys(result.evaluationMeta?.questionsMeta || {})).toHaveLength(3);

      // Check Q1 metadata
      const q1Meta = result.evaluationMeta?.questionsMeta['Q1'];
      expect(q1Meta?.question).toBe('Q1');
      expect(q1Meta?.markedAnswer).toBe('A');
      expect(q1Meta?.questionVerdict).toBe('correct');
      expect(q1Meta?.delta).toBe(1);
    });

    it('should store evaluation config and answers summary', () => {
      const config = new MockEvaluationConfig();
      const processor = new EvaluationProcessor(config);

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = { Q1: 'A', Q2: 'B', Q3: 'D' };

      const result = processor.process(context);

      expect(result.evaluationConfigForResponse).toBeDefined();
      expect(result.defaultAnswersSummary).toBeDefined();
      expect(result.defaultAnswersSummary).toBe('Correct: 2, Incorrect: 1');
    });

    it('should preserve existing context properties', () => {
      const config = new MockEvaluationConfig();
      const processor = new EvaluationProcessor(config);

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = { Q1: 'A', Q2: 'B', Q3: 'D' };
      context.isMultiMarked = true;
      context.metadata = { custom: 'value' };

      const result = processor.process(context);

      expect(result.isMultiMarked).toBe(true);
      expect(result.metadata.custom).toBe('value');
    });
  });

  describe('evaluateConcatenatedResponse', () => {
    it('should accumulate score correctly', () => {
      const config = new MockEvaluationConfig();
      const processor = new EvaluationProcessor(config);

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = {
        Q1: 'A', // +1, cumulative: 1
        Q2: 'B', // +1, cumulative: 2
        Q3: 'D', // -0.25, cumulative: 1.75
      };

      const result = processor.process(context);

      // Check cumulative scores in metadata
      expect(result.evaluationMeta?.questionsMeta['Q1'].currentScore).toBe(1);
      expect(result.evaluationMeta?.questionsMeta['Q2'].currentScore).toBe(2);
      expect(result.evaluationMeta?.questionsMeta['Q3'].currentScore).toBe(1.75);
    });
  });

  describe('inheritance', () => {
    it('should be instance of Processor', () => {
      const processor = new EvaluationProcessor(null);

      expect(typeof processor.process).toBe('function');
      expect(typeof processor.getName).toBe('function');
    });
  });
});

