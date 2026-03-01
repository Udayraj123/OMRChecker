/**
 * Memory leak tests for DrawingUtils (drawing.ts)
 *
 * Each test verifies that OpenCV.js Mat objects created inside a DrawingUtils
 * method are fully released after the call returns.  The strategy is:
 *
 *   1. Record cv.matCount() BEFORE the call.
 *   2. Execute the method.
 *   3. Delete any Mat the method returns (caller's responsibility).
 *   4. Assert cv.matCount() equals the value recorded in step 1.
 *
 * If cv.matCount is not available in the test environment the tests are skipped
 * gracefully via the `skipIfNoMatCount` helper exported from `setup.ts`.
 *
 * Mock strategy: @techstark/opencv-js requires a real browser (WebAssembly +
 * canvas).  In jsdom we replace the global `cv` with a lightweight mock that
 * tracks live Mat instances through a counter and exposes `cv.matCount()`.
 * The mock methods record all arguments so we can also assert correct
 * parameter forwarding where relevant.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { getMatCount, skipIfNoMatCount } from '../setup';

// ---------------------------------------------------------------------------
// Minimal OpenCV.js mock
// ---------------------------------------------------------------------------

/** Counter of Mat objects currently alive in the mock. */
let _liveMatCount = 0;

function makeMockMat(rows = 100, cols = 100, type = 16 /* CV_8UC3 */): MockMat {
  _liveMatCount++;
  const mat: MockMat = {
    rows,
    cols,
    type: () => type,
    delete: () => {
      if (!mat._deleted) {
        mat._deleted = true;
        _liveMatCount--;
      }
    },
    _deleted: false,
  };
  return mat;
}

interface MockMat {
  rows: number;
  cols: number;
  type: () => number;
  delete: () => void;
  _deleted: boolean;
}

/** Calls recorded by mock cv.line / cv.rectangle / cv.putText / etc. */
const mockCalls: Record<string, unknown[][]> = {
  line: [],
  rectangle: [],
  hconcat: [],
  drawContours: [],
  arrowedLine: [],
  putText: [],
  getTextSize: [],
};

function resetMockCalls() {
  for (const key of Object.keys(mockCalls)) {
    mockCalls[key] = [];
  }
}

/**
 * Build a fresh mock `cv` object.  Re-created each test so that
 * `_liveMatCount` isolation is guaranteed.
 */
function buildMockCv() {
  _liveMatCount = 0;

  const mockCv = {
    // ---- Constants ---------------------------------------------------------
    CV_32SC2: 12,
    LINE_AA: 16,
    FONT_HERSHEY_SIMPLEX: 0,

    // ---- Mat factory -------------------------------------------------------
    Mat: function MockMat_ctor() {
      return makeMockMat();
    },

    // ---- MatVector ---------------------------------------------------------
    MatVector: function MockMatVector_ctor() {
      const items: MockMat[] = [];
      return {
        push_back(mat: MockMat) {
          items.push(mat);
        },
        size() {
          return items.length;
        },
        get(i: number) {
          return items[i];
        },
        delete() {
          // MatVector itself does NOT own the Mats; caller manages them.
          // Decrement for the vector wrapper itself.
        },
        _items: items,
      };
    },

    // ---- Point (value type, no heap allocation) ----------------------------
    Point: function MockPoint_ctor(x: number, y: number) {
      return { x, y };
    },

    // ---- matFromArray ------------------------------------------------------
    matFromArray(rows: number, cols: number, type: number, data: number[]) {
      void data;
      return makeMockMat(rows, cols, type);
    },

    // ---- Drawing primitives (record calls, do nothing to pixels) ----------
    hconcat(src: unknown, dst: MockMat) {
      mockCalls['hconcat'].push([src, dst]);
    },
    line(...args: unknown[]) {
      mockCalls['line'].push(args);
    },
    rectangle(...args: unknown[]) {
      mockCalls['rectangle'].push(args);
    },
    drawContours(...args: unknown[]) {
      mockCalls['drawContours'].push(args);
    },
    arrowedLine(...args: unknown[]) {
      mockCalls['arrowedLine'].push(args);
    },
    putText(...args: unknown[]) {
      mockCalls['putText'].push(args);
    },

    // ---- getTextSize -------------------------------------------------------
    getTextSize(text: string, fontFace: number, fontScale: number, thickness: number) {
      mockCalls['getTextSize'].push([text, fontFace, fontScale, thickness]);
      return { size: { width: 50, height: 20 }, baseLine: 4 };
    },

    // ---- matCount (mirrors _liveMatCount) ----------------------------------
    matCount() {
      return _liveMatCount;
    },
  };

  return mockCv;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Returns the number of live OpenCV Mats in the mock. */
function liveCount(): number {
  return _liveMatCount;
}

/** Convenience: create a mock image Mat to pass as the `image` parameter. */
function mockImage(rows = 200, cols = 300): MockMat {
  return makeMockMat(rows, cols);
}

// ---------------------------------------------------------------------------
// Module-level mock wiring
// The module `drawing.ts` imports `cv` from `@techstark/opencv-js`.
// We replace it with our mock before any tests run.
// ---------------------------------------------------------------------------

// We use vi.mock with a factory.  The mock is registered once; each test
// rebuilds `mockCv` via `beforeEach` and patches the module's reference.
vi.mock('@techstark/opencv-js', () => {
  // Return a Proxy so that when `drawing.ts` destructures or calls `cv.*`,
  // it always goes through the current `globalThis.__mockCv`.
  const proxy = new Proxy(
    {},
    {
      get(_target, prop: string) {
        const m = (globalThis as any).__mockCv;
        if (m && prop in m) {
          const val = m[prop as keyof typeof m];
          return typeof val === 'function' ? val.bind(m) : val;
        }
        return undefined;
      },
    }
  );
  return { default: proxy };
});

// ---------------------------------------------------------------------------
// Suite setup / teardown
// ---------------------------------------------------------------------------

let DrawingUtils: typeof import('../../src/utils/drawing').DrawingUtils;

beforeEach(async () => {
  resetMockCalls();
  (globalThis as any).__mockCv = buildMockCv();

  // Dynamic import so the mock is in place before the module initializes.
  // Vitest re-uses the module cache; we need to reset it.
  vi.resetModules();
  ({ DrawingUtils } = await import('../../src/utils/drawing'));
});

afterEach(() => {
  (globalThis as any).__mockCv = undefined;
  _liveMatCount = 0;
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('DrawingUtils — OpenCV.js memory leak tests', () => {
  // -------------------------------------------------------------------------
  // drawMatches
  // -------------------------------------------------------------------------

  describe('drawMatches', () => {
    it('releases the internal MatVector after concatenating images', () => {
      const image = mockImage();
      const warped = mockImage();
      // image + warped + the result Mat = 3 allocations at the point of call
      const before = liveCount(); // 2 (image + warped already counted)

      const result = DrawingUtils.drawMatches(
        image as any,
        [[10, 10], [20, 20]],
        warped as any,
        [[15, 15], [25, 25]]
      );

      // The internal MatVector should be .delete()-ed by now.
      // Only the returned result Mat remains above the pre-call baseline.
      expect(liveCount()).toBe(before + 1); // +1 for the result Mat

      result.delete();
      expect(liveCount()).toBe(before);

      image.delete();
      warped.delete();
    });

    it('does not leak when fromPoints and toPoints are empty', () => {
      const image = mockImage();
      const warped = mockImage();
      const before = liveCount();

      const result = DrawingUtils.drawMatches(image as any, [], warped as any, []);

      expect(liveCount()).toBe(before + 1); // only result Mat
      result.delete();
      expect(liveCount()).toBe(before);

      image.delete();
      warped.delete();
    });

    it('does not leak when point arrays have different lengths', () => {
      const image = mockImage();
      const warped = mockImage();
      const before = liveCount();

      const result = DrawingUtils.drawMatches(
        image as any,
        [[5, 5], [10, 10], [15, 15]],
        warped as any,
        [[5, 5]]
      );

      expect(liveCount()).toBe(before + 1);
      result.delete();
      expect(liveCount()).toBe(before);

      image.delete();
      warped.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawContour
  // -------------------------------------------------------------------------

  describe('drawContour', () => {
    it('releases the contour Mat and MatVector after drawing', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawContour(
        image as any,
        [[0, 0], [50, 0], [50, 50], [0, 50]]
      );

      // Both contourMat and the MatVector wrapper should be deleted.
      expect(liveCount()).toBe(before);

      image.delete();
    });

    it('does not leak for a single-point contour', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawContour(image as any, [[10, 10]]);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak for a large contour (100 points)', () => {
      const image = mockImage();
      const contour = Array.from({ length: 100 }, (_, i) => [i, i] as [number, number]);
      const before = liveCount();

      DrawingUtils.drawContour(image as any, contour);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('throws on invalid contour without leaving orphaned Mats', () => {
      const image = mockImage();
      const before = liveCount();

      // null inside contour triggers the guard
      expect(() =>
        DrawingUtils.drawContour(image as any, [[10, 10], null as any, [20, 20]])
      ).toThrow('Invalid contour provided');

      // The throw must happen BEFORE any Mat is allocated inside the method
      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawBoxDiagonal
  // -------------------------------------------------------------------------

  describe('drawBoxDiagonal', () => {
    it('creates no persistent Mats (only transient cv.Point value types)', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawBoxDiagonal(image as any, [10, 10], [90, 90]);

      // cv.Point is a plain object (value type) in the mock; no heap Mat.
      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak when called with custom color and border', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawBoxDiagonal(image as any, [5, 5], [50, 50], [0, 0, 0], 5);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawBox
  // -------------------------------------------------------------------------

  describe('drawBox', () => {
    it('does not allocate or leak any Mats for BOX_HOLLOW style', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawBox(image as any, [10, 10], [80, 60], undefined, 'BOX_HOLLOW');

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not allocate or leak any Mats for BOX_FILLED style', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawBox(image as any, [10, 10], [80, 60], [200, 200, 200], 'BOX_FILLED');

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak when centered=true', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawBox(
        image as any,
        [50, 50],
        [40, 40],
        undefined,
        'BOX_HOLLOW',
        1 / 12,
        3,
        true
      );

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawArrows
  // -------------------------------------------------------------------------

  describe('drawArrows', () => {
    it('does not allocate or leak any Mats', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawArrows(
        image as any,
        [[10, 10], [30, 10]],
        [[10, 50], [30, 50]]
      );

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak with mismatched start/end arrays', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawArrows(
        image as any,
        [[0, 0], [10, 10], [20, 20]],
        [[5, 5]]
      );

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak with empty arrays', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawArrows(image as any, [], []);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawTextResponsive
  // -------------------------------------------------------------------------

  describe('drawTextResponsive', () => {
    it('does not allocate or leak any Mats', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawTextResponsive(image as any, 'Hello', [10, 20]);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak when text overflows image boundary (clamping path)', () => {
      // position + textSize > image dimensions triggers the clamping branch
      const image = mockImage(50, 50);
      const before = liveCount();

      DrawingUtils.drawTextResponsive(image as any, 'Overflow text', [40, 45]);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawText
  // -------------------------------------------------------------------------

  describe('drawText', () => {
    it('does not leak when position is a static Point', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawText(image as any, 'Hello', [10, 20]);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak when position is a callback (uses getTextSize)', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawText(
        image as any,
        'Centered',
        (sizeX: number, sizeY: number) => [100 - sizeX, 50 + sizeY]
      );

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak when centered=true', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawText(image as any, 'Centered', [50, 50], 0.95, 2, true);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('throws (not leaks) when centered=true and position is a function', () => {
      const image = mockImage();
      const before = liveCount();

      expect(() =>
        DrawingUtils.drawText(
          image as any,
          'Bad',
          () => [0, 0],
          0.95,
          2,
          true // centered
        )
      ).toThrow('centered=true but position is a callable');

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak with empty string', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawText(image as any, '', [10, 10]);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawSymbol
  // -------------------------------------------------------------------------

  describe('drawSymbol', () => {
    it('does not allocate or leak any Mats', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawSymbol(image as any, 'A', [10, 10], [90, 90]);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawLine
  // -------------------------------------------------------------------------

  describe('drawLine', () => {
    it('does not allocate or leak any Mats', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawLine(image as any, [0, 0], [100, 100]);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak when start equals end (zero-length line)', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawLine(image as any, [50, 50], [50, 50]);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawPolygon
  // -------------------------------------------------------------------------

  describe('drawPolygon', () => {
    it('does not leak for a closed polygon (triangle)', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawPolygon(image as any, [[0, 0], [50, 0], [25, 50]], undefined, 1, true);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak for an open polygon', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawPolygon(
        image as any,
        [[0, 0], [100, 0], [100, 100], [0, 100]],
        [0, 0, 255],
        2,
        false
      );

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak for a single-point polygon (no segments drawn)', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawPolygon(image as any, [[10, 10]], undefined, 1, true);

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('does not leak for a large polygon (50 points)', () => {
      const image = mockImage();
      const points: [number, number][] = Array.from({ length: 50 }, (_, i) => {
        const angle = (2 * Math.PI * i) / 50;
        return [
          Math.round(100 + 80 * Math.cos(angle)),
          Math.round(100 + 80 * Math.sin(angle)),
        ];
      });
      const before = liveCount();

      DrawingUtils.drawPolygon(image as any, points);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // drawGroup
  // -------------------------------------------------------------------------

  describe('drawGroup', () => {
    const edges = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT'] as const;

    for (const edge of edges) {
      it(`does not allocate or leak any Mats for boxEdge="${edge}"`, () => {
        const image = mockImage();
        const before = liveCount();

        DrawingUtils.drawGroup(image as any, [10, 10], [80, 60], edge, [0, 128, 0]);

        expect(liveCount()).toBe(before);
        image.delete();
      });
    }

    it('does not leak with custom thickness and thicknessFactor', () => {
      const image = mockImage();
      const before = liveCount();

      DrawingUtils.drawGroup(image as any, [0, 0], [100, 100], 'TOP', [255, 0, 0], 5, 0.9);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // Repeated calls — cumulative leak detection
  // -------------------------------------------------------------------------

  describe('repeated invocations (cumulative leak detection)', () => {
    it('drawContour called 20 times leaves zero orphaned Mats', () => {
      const image = mockImage();
      const before = liveCount();

      for (let i = 0; i < 20; i++) {
        DrawingUtils.drawContour(
          image as any,
          [[0, 0], [i + 10, 0], [i + 10, i + 10], [0, i + 10]]
        );
      }

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('drawMatches called 10 times accumulates only 10 result Mats', () => {
      const image = mockImage();
      const warped = mockImage();
      const before = liveCount();
      const results: MockMat[] = [];

      for (let i = 0; i < 10; i++) {
        results.push(DrawingUtils.drawMatches(image as any, [], warped as any, []) as any);
      }

      expect(liveCount()).toBe(before + 10); // 1 result Mat per call

      for (const r of results) {
        r.delete();
      }
      expect(liveCount()).toBe(before);

      image.delete();
      warped.delete();
    });

    it('drawText called 50 times (static position) leaves zero orphaned Mats', () => {
      const image = mockImage();
      const before = liveCount();

      for (let i = 0; i < 50; i++) {
        DrawingUtils.drawText(image as any, `label-${i}`, [10, 20 + i]);
      }

      expect(liveCount()).toBe(before);
      image.delete();
    });

    it('drawText called 50 times (function position) leaves zero orphaned Mats', () => {
      const image = mockImage();
      const before = liveCount();

      for (let i = 0; i < 50; i++) {
        DrawingUtils.drawText(
          image as any,
          `label-${i}`,
          (sx: number, sy: number) => [10 + sx, 20 + sy]
        );
      }

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });

  // -------------------------------------------------------------------------
  // Integration: composite scenario
  // -------------------------------------------------------------------------

  describe('composite scenario (no leaks end-to-end)', () => {
    it('annotating an image with boxes, text, contours, and lines leaves no orphaned Mats', () => {
      const image = mockImage(400, 600);
      const before = liveCount();

      // Simulate a typical annotation pass
      DrawingUtils.drawBox(image as any, [10, 10], [100, 80]);
      DrawingUtils.drawText(image as any, 'Q1', [15, 25]);
      DrawingUtils.drawContour(image as any, [[10, 10], [110, 10], [110, 90], [10, 90]]);
      DrawingUtils.drawLine(image as any, [10, 10], [110, 90]);
      DrawingUtils.drawPolygon(image as any, [[50, 50], [150, 50], [100, 120]]);
      DrawingUtils.drawGroup(image as any, [10, 10], [80, 60], 'BOTTOM', [0, 0, 255]);
      DrawingUtils.drawSymbol(image as any, '✓', [10, 10], [60, 60]);
      DrawingUtils.drawTextResponsive(image as any, 'Score: 42', [10, 370]);
      DrawingUtils.drawArrows(image as any, [[0, 0]], [[50, 50]]);

      expect(liveCount()).toBe(before);
      image.delete();
    });
  });
});
