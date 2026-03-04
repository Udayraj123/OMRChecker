/**
 * Unit tests for BubbleReader.ts
 *
 * Translated / inspired by Python:
 *   src/processors/detection/bubbles_threshold/detection.py
 *   src/processors/detection/threshold/local_threshold.py
 *   src/processors/detection/bubbles_threshold/interpretation.py
 *
 * Uses Vitest with a manual mock of @techstark/opencv-js so no real OpenCV
 * WASM is needed. The mock exposes:
 *   cv.Rect     — plain constructor (x, y, w, h)
 *   cv.mean     — returns [roi.__mean ?? 200, 0, 0, 0]
 *   cv.Mat      — stub rows=500, cols=500 + roi() method
 */

import { vi, describe, it, expect, beforeEach } from 'vitest';

// ── OpenCV mock ───────────────────────────────────────────────────────────────
//
// BubbleReader uses:
//   new cv.Rect(x, y, w, h)
//   grayImage.roi(rect)
//   cv.mean(roi)[0]
//   roi.delete()

vi.mock('@techstark/opencv-js', () => {
  class MockRect {
    constructor(
      public x: number,
      public y: number,
      public w: number,
      public h: number,
    ) {}
  }

  class MockMat {
    rows = 500;
    cols = 500;
    /**
     * Return a stub ROI whose __mean can be controlled by the test via
     * rect.__mean (set by the test helper below).
     */
    roi(rect: any) {
      return {
        __mean: rect.__mean ?? 200,
        delete() {},
      };
    }
    delete() {}
  }

  return {
    default: {
      Rect: MockRect,
      mean: (roi: any) => [roi.__mean ?? 200, 0, 0, 0],
      Mat: MockMat,
    },
  };
});

// ── Import after mock ─────────────────────────────────────────────────────────

import { BubbleReader } from '../../../src/detection/BubbleReader';
import type { OMRResponse } from '../../../src/detection/BubbleReader';

// ── Test helpers ──────────────────────────────────────────────────────────────

/**
 * Create a fake grayscale cv.Mat whose roi() method returns a stub
 * with the given mean values.
 *
 * @param means - Pixel mean per scanBox, in order. Each call to roi() pops the next value.
 */
function makeMockMat(means: number[]): any {
  let callIdx = 0;
  return {
    rows: 500,
    cols: 500,
    roi(rect: any) {
      const mean = means[callIdx++] ?? 200;
      return {
        __mean: mean,
        delete() {},
      };
    },
    delete() {},
  };
}

/**
 * Create a minimal BubbleField-like object with one or more scan boxes.
 *
 * @param fieldLabel   - Field label (e.g. 'q1')
 * @param emptyValue   - Value to return when no / all bubbles are marked
 * @param bubbleValues - Array of bubble value strings (e.g. ['A','B','C','D'])
 */
function makeField(fieldLabel: string, emptyValue: string, bubbleValues: string[]): any {
  const origin: [number, number] = [10, 10];
  const gap = 50;
  const scanBoxes = bubbleValues.map((value, idx) => ({
    bubbleValue: value,
    bubbleDimensions: [40, 40] as [number, number],
    getShiftedPosition: () => [origin[0] + idx * gap, origin[1]] as [number, number],
  }));
  return { fieldLabel, emptyValue, scanBoxes };
}

/**
 * Create a minimal Template-like object with a given list of fields.
 */
function makeTemplate(fields: any[]): any {
  return { allFields: fields };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('BubbleReader', () => {
  let reader: BubbleReader;

  beforeEach(() => {
    reader = new BubbleReader();
  });

  // ── readBubbles ─────────────────────────────────────────────────────────────

  describe('readBubbles', () => {
    it('returns emptyValue when no bubble is marked (all means above threshold)', () => {
      // Bubble means: [200, 200, 200, 200] — all well above default threshold 127.5
      const field = makeField('q1', '', ['A', 'B', 'C', 'D']);
      const mat = makeMockMat([200, 200, 200, 200]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      expect(response['q1']).toBe('');
    });

    it('returns emptyValue when all bubbles are marked (multi-mark protection)', () => {
      // Bubble means: [50, 50, 50, 50] — all clearly below threshold 127.5
      const field = makeField('q1', '', ['A', 'B', 'C', 'D']);
      const mat = makeMockMat([50, 50, 50, 50]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      expect(response['q1']).toBe('');
    });

    it('returns bubbleValue when exactly one bubble is clearly marked', () => {
      // B (idx=1) is dark (mean=50), others are light (mean=210)
      // Sorted: [50, 210, 210, 210]
      // Jump analysis (i=1): 210-50=160 > confidentJump(40)
      // threshold = 50 + 160/2 = 130
      // Only B (50 < 130) is marked
      const field = makeField('q1', '', ['A', 'B', 'C', 'D']);
      const mat = makeMockMat([210, 50, 210, 210]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      expect(response['q1']).toBe('B');
    });

    it('returns concatenated values when multiple bubbles are marked in a multi-answer field', () => {
      // q5 is an integer field with 2 digit columns
      // q5_1: digit '3' marked (mean=60), others unmarked (mean=200)
      // q5_2: digit '7' marked (mean=60), others unmarked (mean=200)
      const field1 = makeField('q5_1', '', ['0','1','2','3','4','5','6','7','8','9']);
      const field2 = makeField('q5_2', '', ['0','1','2','3','4','5','6','7','8','9']);
      // means for q5_1: digit[3]=60, rest=200
      const mat1 = makeMockMat([200,200,200,60,200,200,200,200,200,200, 200,200,200,200,200,200,200,60,200,200]);
      const template = makeTemplate([field1, field2]);

      const response: OMRResponse = reader.readBubbles(mat1, template);

      expect(response['q5_1']).toBe('3');
      expect(response['q5_2']).toBe('7');
    });

    it('handles multiple fields in the template', () => {
      const fields = [
        makeField('Medium', '', ['E', 'H']),
        makeField('q1', '', ['A', 'B', 'C', 'D']),
      ];
      // Medium: E (idx=0) mean=60, H (idx=1) mean=200 → E is marked
      // Two bubbles: gap = 200-60 = 140 >= minGapTwoBubbles(20) → threshold = (60+200)/2 = 130
      // q1: A (idx=0) mean=200, B mean=200, C mean=200, D mean=200 → none marked
      const mat = makeMockMat([60, 200, 200, 200, 200, 200]);
      const template = makeTemplate(fields);

      const response: OMRResponse = reader.readBubbles(mat, template);

      expect(response['Medium']).toBe('E');
      expect(response['q1']).toBe('');
    });

    it('pushes 255 (unmarked) for out-of-bounds scan boxes', () => {
      // field origin is at x=490, y=490 — with bubbleDimensions [40,40]
      // safeW = min(40, 500-490) = 10, safeH = 10 → still valid (goes to roi path)
      // Force the scan box way out of bounds so safeW <= 0
      const scanBox = {
        bubbleValue: 'A',
        bubbleDimensions: [40, 40] as [number, number],
        getShiftedPosition: () => [600, 600] as [number, number], // beyond 500x500 mat
      };
      const field = { fieldLabel: 'q1', emptyValue: '-', scanBoxes: [scanBox] };
      // With single bubble, localThreshold returns globalFallback (127.5).
      // bubbleMeans[0] = 255 (out-of-bounds fallback), 255 >= 127.5 → not marked
      const mat = makeMockMat([]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      // Single bubble, out-of-bounds → emptyValue (localThreshold returns fallback for len<2)
      expect(response['q1']).toBe('-');
    });
  });

  // ── localThreshold (tested indirectly via readBubbles) ──────────────────────

  describe('localThreshold edge cases (via readBubbles)', () => {
    it('falls back to globalFallbackThreshold for a single bubble', () => {
      // Single bubble with mean=100 < default globalFallback 127.5 → marked
      const field = makeField('q1', '', ['A']);
      const mat = makeMockMat([100]);
      const template = makeTemplate([field]);

      // Single bubble → local threshold = globalFallback = 127.5
      // mean=100 < 127.5 → marked
      // BUT all bubbles marked (1 of 1) → emptyValue
      const response: OMRResponse = reader.readBubbles(mat, template);
      expect(response['q1']).toBe('');
    });

    it('falls back to globalFallbackThreshold when 2-bubble gap is too small', () => {
      // Two bubbles with means [100, 115] — gap=15 < minGapTwoBubbles(20) → fallback
      // fallback = 127.5; both 100 and 115 are < 127.5 → both marked → emptyValue
      const field = makeField('q1', '', ['A', 'B']);
      const mat = makeMockMat([100, 115]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      // Both below fallback → all marked → emptyValue
      expect(response['q1']).toBe('');
    });

    it('uses midpoint threshold for 2-bubble field with sufficient gap', () => {
      // Two bubbles: A=50, B=200 — gap=150 >= minGapTwoBubbles(20)
      // threshold = (50+200)/2 = 125
      // A(50) < 125 → marked; B(200) >= 125 → unmarked
      const field = makeField('q1', '', ['A', 'B']);
      const mat = makeMockMat([50, 200]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      expect(response['q1']).toBe('A');
    });

    it('falls back to globalFallbackThreshold for 3+ bubbles with small maxJump', () => {
      // Three bubbles with similar means [120, 125, 130] — maxJump is tiny
      // confidentJump = minJump(30) + minJumpSurplusForGlobalFallback(10) = 40
      // All jumps << 40 → fallback = 127.5
      // 120 < 127.5 → A marked; 125 < 127.5 → B marked; 130 >= 127.5 → C not
      // 2 of 3 marked → fieldValue = 'AB'
      const field = makeField('q1', '', ['A', 'B', 'C']);
      const mat = makeMockMat([120, 125, 130]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      expect(response['q1']).toBe('AB');
    });

    it('uses local threshold for 3+ bubbles with large jump', () => {
      // Four bubbles: A=40, B=200, C=210, D=220
      // Sorted: [40, 200, 210, 220]
      // i=1: jump = sorted[2]-sorted[0] = 210-40 = 170 → threshold = 40 + 170/2 = 125
      // i=2: jump = sorted[3]-sorted[1] = 220-200 = 20
      // maxJump=170 >= confidentJump(40) → local threshold = 125
      // A(40) < 125 → marked; others >= 125 → not marked
      const field = makeField('q1', '', ['A', 'B', 'C', 'D']);
      const mat = makeMockMat([40, 200, 210, 220]);
      const template = makeTemplate([field]);

      const response: OMRResponse = reader.readBubbles(mat, template);

      expect(response['q1']).toBe('A');
    });

    it('respects custom globalFallbackThreshold config', () => {
      // Custom reader with globalFallbackThreshold=80
      const customReader = new BubbleReader({ globalFallbackThreshold: 80 });

      // Single bubble with mean=90; fallback=80; 90 >= 80 → not marked
      // (single bubble → fallback threshold)
      const field = makeField('q1', 'EMPTY', ['A']);
      const mat = makeMockMat([90]);
      const template = makeTemplate([field]);

      const response: OMRResponse = customReader.readBubbles(mat, template);

      // mean=90 >= globalFallback=80 → not marked, but single bubble + all not marked → emptyValue
      // Wait: single bubble → localThreshold returns fallback=80; mean=90 >= 80 → not marked (0 of 1)
      // 0 marked → emptyValue
      expect(response['q1']).toBe('EMPTY');
    });

    it('respects custom minGapTwoBubbles config', () => {
      // Custom reader with a very large minGapTwoBubbles=200 (almost never satisfied)
      // Two bubbles: A=50, B=200 — gap=150 < 200 → fallback = 127.5
      const customReader = new BubbleReader({ minGapTwoBubbles: 200 });
      const field = makeField('q1', '', ['A', 'B']);
      const mat = makeMockMat([50, 200]);
      const template = makeTemplate([field]);

      const response: OMRResponse = customReader.readBubbles(mat, template);

      // gap=150 < minGapTwoBubbles=200 → fallback=127.5
      // A(50) < 127.5 → marked; B(200) >= 127.5 → not
      expect(response['q1']).toBe('A');
    });
  });
});
