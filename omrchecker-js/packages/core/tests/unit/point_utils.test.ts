import { describe, it, expect } from 'vitest';
import {
  PointParser,
  WarpedDimensionsCalculator,
  orderFourPoints,
  computePointDistances,
  computeBoundingBox,
} from '../../src/processors/image/point_utils';

// ---------------------------------------------------------------------------
// TestPointParser — parsePoints
// ---------------------------------------------------------------------------

describe('PointParser', () => {
  describe('parsePoints', () => {
    it('test_parse_simple_array: parses a plain Nx2 array', () => {
      const [control, dest] = PointParser.parsePoints([
        [10, 20],
        [30, 40],
        [50, 60],
        [70, 80],
      ]);
      expect(control.length).toBe(4);
      expect(control[0].length).toBe(2);
      expect(dest.length).toBe(4);
      expect(dest[0].length).toBe(2);
      expect(control).toEqual(dest);
    });

    it('test_parse_numpy_array: parses a float-valued Nx2 array', () => {
      const input = [
        [1.5, 2.5],
        [3.5, 4.5],
        [5.5, 6.5],
        [7.5, 8.5],
      ];
      const [control, dest] = PointParser.parsePoints(input);
      expect(control).toEqual(input);
      expect(dest).toEqual(input);
    });

    it('test_parse_tuple_of_arrays: parses a [control, dest] pair with different arrays', () => {
      const controlInput = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ];
      const destInput = [
        [10, 10],
        [90, 10],
        [90, 90],
        [10, 90],
      ];
      const [control, dest] = PointParser.parsePoints([controlInput, destInput]);
      expect(control).toEqual(controlInput);
      expect(dest).toEqual(destInput);
      // They must differ
      expect(control).not.toEqual(dest);
    });

    it('test_parse_template_dimensions_reference: resolves "template.dimensions"', () => {
      const [control, dest] = PointParser.parsePoints('template.dimensions', {
        templateDimensions: [800, 1200],
      });
      const expected = [
        [0, 0],
        [799, 0],
        [799, 1199],
        [0, 1199],
      ];
      expect(control).toEqual(expected);
      expect(dest).toEqual(expected);
    });

    it('test_parse_page_dimensions_reference: resolves "page_dimensions"', () => {
      const [control, dest] = PointParser.parsePoints('page_dimensions', {
        pageDimensions: [600, 800],
      });
      const expected = [
        [0, 0],
        [599, 0],
        [599, 799],
        [0, 799],
      ];
      expect(control).toEqual(expected);
      expect(dest).toEqual(expected);
    });

    it('test_parse_context_reference: resolves a named key from context', () => {
      const contextPoints = [
        [5, 10],
        [50, 10],
        [50, 80],
        [5, 80],
      ];
      const [control] = PointParser.parsePoints('my_points', {
        context: { my_points: contextPoints },
      });
      expect(control.length).toBe(4);
      expect(control[0].length).toBe(2);
      expect(control).toEqual(contextPoints);
    });

    it('test_missing_template_dimensions_raises_error: missing templateDimensions throws', () => {
      expect(() =>
        PointParser.parsePoints('template.dimensions')
      ).toThrow('requires template_dimensions');
    });

    it('test_missing_page_dimensions_raises_error: missing pageDimensions throws', () => {
      expect(() =>
        PointParser.parsePoints('page_dimensions')
      ).toThrow('requires page_dimensions');
    });

    it('test_unknown_reference_raises_error: unknown string reference throws', () => {
      expect(() =>
        PointParser.parsePoints('unknown_reference')
      ).toThrow('Unknown point reference');
    });

    it('test_invalid_type_raises_error: non-array/string input throws', () => {
      expect(() =>
        PointParser.parsePoints(42 as unknown as never)
      ).toThrow('Invalid points specification type');
    });
  });

  // -------------------------------------------------------------------------
  // TestPointParser — validatePoints
  // -------------------------------------------------------------------------

  describe('validatePoints', () => {
    it('test_validate_points_valid: valid Nx2 arrays do not throw', () => {
      const control = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ];
      const dest = [
        [10, 10],
        [20, 10],
        [20, 20],
        [10, 20],
      ];
      expect(() => PointParser.validatePoints(control, dest)).not.toThrow();
    });

    it('test_validate_points_wrong_shape: 1D control array throws "must be Nx2 array"', () => {
      // Python: np.array([0,1,2,3]) is 1D; in TS we pass a flat number[]
      expect(() =>
        PointParser.validatePoints(
          [0, 1, 2, 3] as unknown as number[][],
          [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
          ]
        )
      ).toThrow('must be Nx2 array');
    });

    it('test_validate_points_mismatch_count: different lengths throw "Mismatch"', () => {
      const control = [
        [0, 0],
        [1, 0],
        [1, 1],
      ];
      const dest = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ];
      expect(() => PointParser.validatePoints(control, dest)).toThrow('Mismatch');
    });

    it('test_validate_points_too_few: fewer than minPoints throws "At least 4 points required"', () => {
      const control = [
        [0, 0],
        [1, 0],
      ];
      const dest = [
        [0, 0],
        [1, 0],
      ];
      expect(() => PointParser.validatePoints(control, dest, 4)).toThrow(
        'At least 4 points required'
      );
    });
  });
});

// ---------------------------------------------------------------------------
// TestWarpedDimensionsCalculator
// ---------------------------------------------------------------------------

describe('WarpedDimensionsCalculator', () => {
  it('test_calculate_from_points_simple: [[0,0],[100,0],[100,200],[0,200]] → 101 × 201', () => {
    const points = [
      [0, 0],
      [100, 0],
      [100, 200],
      [0, 200],
    ];
    const [width, height] = WarpedDimensionsCalculator.calculateFromPoints(points);
    expect(width).toBe(101);
    expect(height).toBe(201);
  });

  it('test_calculate_from_points_with_padding: padding=10 adds 20 to each dimension', () => {
    const points = [
      [0, 0],
      [100, 100],
    ];
    const [width, height] = WarpedDimensionsCalculator.calculateFromPoints(points, 10);
    // width = ceil(100-0)+1+20 = 121; height = ceil(100-0)+1+20 = 121
    expect(width).toBe(121);
    expect(height).toBe(121);
  });

  it('test_calculate_from_points_with_max_dimension: scales down so max(w,h) <= maxDimension', () => {
    const points = [
      [0, 0],
      [2000, 0],
      [2000, 3000],
      [0, 3000],
    ];
    const [width, height] = WarpedDimensionsCalculator.calculateFromPoints(
      points,
      0,
      1000
    );
    expect(Math.max(width, height)).toBeLessThanOrEqual(1000);
    // Aspect ratio should be roughly preserved (~2001/3001)
    const ratio = width / height;
    expect(ratio).toBeCloseTo(2001 / 3001, 1);
  });

  it('test_calculate_from_dimensions: (800,1200) scale=1.0 → 800 × 1200', () => {
    const [w, h] = WarpedDimensionsCalculator.calculateFromDimensions([800, 1200], 1.0);
    expect(w).toBe(800);
    expect(h).toBe(1200);
  });

  it('test_calculate_from_dimensions_with_scale: (800,1200) scale=0.5 → 400 × 600', () => {
    const [w, h] = WarpedDimensionsCalculator.calculateFromDimensions([800, 1200], 0.5);
    expect(w).toBe(400);
    expect(h).toBe(600);
  });
});

// ---------------------------------------------------------------------------
// TestOrderFourPoints
// ---------------------------------------------------------------------------

describe('orderFourPoints', () => {
  it('test_order_already_ordered: already ordered points stay the same', () => {
    const points = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ];
    const result = orderFourPoints(points);
    expect(result).toEqual([
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ]);
  });

  it('test_order_random_order: unsorted points are sorted to [tl, tr, br, bl]', () => {
    const points = [
      [100, 100],
      [0, 0],
      [0, 100],
      [100, 0],
    ];
    const result = orderFourPoints(points);
    expect(result).toEqual([
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ]);
  });

  it('test_order_tilted_rectangle: top 2 have lower y, left points have lower x', () => {
    // A rectangle tilted slightly; just validate invariants
    const points = [
      [10, 5],
      [110, 15],
      [100, 105],
      [0, 95],
    ];
    const [tl, tr, br, bl] = orderFourPoints(points);
    // Top two points have y <= bottom two points
    expect(Math.max(tl[1], tr[1])).toBeLessThanOrEqual(Math.min(br[1], bl[1]));
    // Left points have x <= right points
    expect(tl[0]).toBeLessThanOrEqual(tr[0]);
    expect(bl[0]).toBeLessThanOrEqual(br[0]);
  });

  it('test_order_requires_four_points: throws for non-4-point input', () => {
    expect(() =>
      orderFourPoints([
        [0, 0],
        [1, 0],
        [1, 1],
      ])
    ).toThrow('exactly 4 points');
  });
});

// ---------------------------------------------------------------------------
// TestComputePointDistances
// ---------------------------------------------------------------------------

describe('computePointDistances', () => {
  it('test_zero_distance: identical points yield distance 0', () => {
    const points = [
      [5, 10],
      [20, 30],
    ];
    const distances = computePointDistances(points, points);
    distances.forEach(d => expect(d).toBeCloseTo(0, 5));
  });

  it('test_horizontal_distance: [0,0] → [3,0] gives distance 3', () => {
    const distances = computePointDistances([[0, 0]], [[3, 0]]);
    expect(distances[0]).toBeCloseTo(3.0, 5);
  });

  it('test_diagonal_distance: [0,0] → [3,4] gives distance 5 (3-4-5 triangle)', () => {
    const distances = computePointDistances([[0, 0]], [[3, 4]]);
    expect(distances[0]).toBeCloseTo(5.0, 5);
  });

  it('test_multiple_distances: three pairs give [0.0, 5.0, 5.0]', () => {
    const points1 = [
      [0, 0],
      [0, 0],
      [0, 0],
    ];
    const points2 = [
      [0, 0],
      [3, 4],
      [4, 3],
    ];
    const distances = computePointDistances(points1, points2);
    const expected = [0.0, 5.0, 5.0];
    distances.forEach((d, i) => expect(d).toBeCloseTo(expected[i], 5));
  });

  it('test_mismatched_length_raises_error: different array lengths throw', () => {
    expect(() =>
      computePointDistances(
        [
          [0, 0],
          [1, 1],
        ],
        [[0, 0]]
      )
    ).toThrow('same length');
  });
});

// ---------------------------------------------------------------------------
// TestComputeBoundingBox
// ---------------------------------------------------------------------------

describe('computeBoundingBox', () => {
  it('test_simple_rectangle: [[10,20],[100,20],[100,200],[10,200]] → (10,20,100,200)', () => {
    const points = [
      [10, 20],
      [100, 20],
      [100, 200],
      [10, 200],
    ];
    expect(computeBoundingBox(points)).toEqual([10, 20, 100, 200]);
  });

  it('test_single_point: [[50,75]] → (50,75,50,75)', () => {
    expect(computeBoundingBox([[50, 75]])).toEqual([50, 75, 50, 75]);
  });

  it('test_scattered_points: [[30,50],[10,80],[90,20],[50,100]] → (10,20,90,100)', () => {
    const points = [
      [30, 50],
      [10, 80],
      [90, 20],
      [50, 100],
    ];
    expect(computeBoundingBox(points)).toEqual([10, 20, 90, 100]);
  });

  it('test_negative_coordinates: [[-10,-20],[10,20]] → (-10,-20,10,20)', () => {
    expect(computeBoundingBox([[-10, -20], [10, 20]])).toEqual([-10, -20, 10, 20]);
  });
});
