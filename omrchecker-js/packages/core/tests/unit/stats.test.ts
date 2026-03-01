import { describe, it, expect, beforeEach } from 'vitest';
import { StatsByLabel, NumberAggregate } from '../../src/utils/stats';

describe('StatsByLabel', () => {
  let stats: StatsByLabel;

  beforeEach(() => {
    stats = new StatsByLabel('labelA', 'labelB', 'labelC');
  });

  describe('constructor', () => {
    it('should initialize with given labels set to 0', () => {
      const counts = stats.getLabelCounts();
      expect(counts).toEqual({
        labelA: 0,
        labelB: 0,
        labelC: 0,
      });
    });

    it('should work with no labels', () => {
      const emptyStats = new StatsByLabel();
      expect(emptyStats.getLabelCounts()).toEqual({});
    });

    it('should work with single label', () => {
      const singleStats = new StatsByLabel('single');
      expect(singleStats.getLabelCounts()).toEqual({ single: 0 });
    });
  });

  describe('push', () => {
    it('should increment count by 1 when no number provided', () => {
      stats.push('labelA');
      expect(stats.getLabelCounts()['labelA']).toBe(1);
    });

    it('should increment count by specified amount', () => {
      stats.push('labelB', 5);
      expect(stats.getLabelCounts()['labelB']).toBe(5);
    });

    it('should accumulate multiple pushes', () => {
      stats.push('labelC', 3);
      stats.push('labelC', 2);
      stats.push('labelC');
      expect(stats.getLabelCounts()['labelC']).toBe(6);
    });

    it('should throw error for unknown label', () => {
      expect(() => stats.push('unknownLabel')).toThrow(
        'Unknown label passed to stats by label: unknownLabel'
      );
    });

    it('should include allowed labels in error message', () => {
      try {
        stats.push('invalid');
      } catch (error: any) {
        expect(error.message).toContain('labelA, labelB, labelC');
      }
    });

    it('should handle negative increments', () => {
      stats.push('labelA', 10);
      stats.push('labelA', -3);
      expect(stats.getLabelCounts()['labelA']).toBe(7);
    });

    it('should handle zero increment', () => {
      stats.push('labelA', 0);
      expect(stats.getLabelCounts()['labelA']).toBe(0);
    });

    it('should handle floating point increments', () => {
      stats.push('labelB', 2.5);
      stats.push('labelB', 1.5);
      expect(stats.getLabelCounts()['labelB']).toBe(4);
    });
  });

  describe('getLabelCounts', () => {
    it('should return copy of label counts', () => {
      stats.push('labelA', 5);
      const counts1 = stats.getLabelCounts();
      const counts2 = stats.getLabelCounts();
      
      // Should be equal but not the same object
      expect(counts1).toEqual(counts2);
      expect(counts1).not.toBe(counts2);
    });

    it('should not allow external modification', () => {
      const counts = stats.getLabelCounts();
      counts['labelA'] = 999;
      
      expect(stats.getLabelCounts()['labelA']).toBe(0);
    });
  });

  describe('toJSON', () => {
    it('should return JSON-serializable object', () => {
      stats.push('labelA', 1);
      stats.push('labelB', 2);
      
      const json = stats.toJSON();
      expect(json).toEqual({
        label_counts: {
          labelA: 1,
          labelB: 2,
          labelC: 0,
        },
      });
    });

    it('should be serializable to JSON string', () => {
      stats.push('labelA', 5);
      const jsonString = JSON.stringify(stats.toJSON());
      const parsed = JSON.parse(jsonString);
      
      expect(parsed.label_counts.labelA).toBe(5);
    });
  });

  describe('toString', () => {
    it('should return JSON string representation', () => {
      stats.push('labelA', 3);
      const str = stats.toString();
      const parsed = JSON.parse(str);
      
      expect(parsed.label_counts.labelA).toBe(3);
    });
  });
});

describe('NumberAggregate', () => {
  let aggregate: NumberAggregate;

  beforeEach(() => {
    aggregate = new NumberAggregate();
  });

  describe('constructor', () => {
    it('should initialize with empty state', () => {
      expect(aggregate.getCollection()).toEqual([]);
      expect(aggregate.getRunningSum()).toBe(0);
      expect(aggregate.getRunningAverage()).toBe(0);
    });
  });

  describe('push', () => {
    it('should add number with label', () => {
      aggregate.push(10, 'labelA');
      
      expect(aggregate.getCollection()).toEqual([[10, 'labelA']]);
      expect(aggregate.getRunningSum()).toBe(10);
      expect(aggregate.getRunningAverage()).toBe(10);
    });

    it('should accumulate multiple values', () => {
      aggregate.push(10, 'labelA');
      aggregate.push(20, 'labelB');
      aggregate.push(30, 'labelC');
      
      expect(aggregate.getCollection()).toHaveLength(3);
      expect(aggregate.getRunningSum()).toBe(60);
      expect(aggregate.getRunningAverage()).toBe(20);
    });

    it('should handle negative numbers', () => {
      aggregate.push(10, 'a');
      aggregate.push(-5, 'b');
      
      expect(aggregate.getRunningSum()).toBe(5);
      expect(aggregate.getRunningAverage()).toBe(2.5);
    });

    it('should handle zero', () => {
      aggregate.push(0, 'zero');
      aggregate.push(10, 'ten');
      
      expect(aggregate.getRunningSum()).toBe(10);
      expect(aggregate.getRunningAverage()).toBe(5);
    });

    it('should handle floating point numbers', () => {
      aggregate.push(1.5, 'a');
      aggregate.push(2.5, 'b');
      
      expect(aggregate.getRunningSum()).toBe(4);
      expect(aggregate.getRunningAverage()).toBe(2);
    });

    it('should update running average correctly', () => {
      aggregate.push(10, 'a');
      expect(aggregate.getRunningAverage()).toBe(10);
      
      aggregate.push(20, 'b');
      expect(aggregate.getRunningAverage()).toBe(15);
      
      aggregate.push(30, 'c');
      expect(aggregate.getRunningAverage()).toBe(20);
    });
  });

  describe('merge', () => {
    it('should merge two aggregates', () => {
      aggregate.push(10, 'a');
      aggregate.push(20, 'b');
      
      const other = new NumberAggregate();
      other.push(30, 'c');
      other.push(40, 'd');
      
      aggregate.merge(other);
      
      expect(aggregate.getCollection()).toHaveLength(4);
      expect(aggregate.getRunningSum()).toBe(100);
      expect(aggregate.getRunningAverage()).toBe(25);
    });

    it('should merge empty aggregate', () => {
      aggregate.push(10, 'a');
      
      const empty = new NumberAggregate();
      aggregate.merge(empty);
      
      expect(aggregate.getCollection()).toHaveLength(1);
      expect(aggregate.getRunningSum()).toBe(10);
      expect(aggregate.getRunningAverage()).toBe(10);
    });

    it('should merge into empty aggregate', () => {
      const other = new NumberAggregate();
      other.push(5, 'x');
      other.push(15, 'y');
      
      aggregate.merge(other);
      
      expect(aggregate.getCollection()).toHaveLength(2);
      expect(aggregate.getRunningSum()).toBe(20);
      expect(aggregate.getRunningAverage()).toBe(10);
    });

    it('should maintain correct order after merge', () => {
      aggregate.push(1, 'a');
      aggregate.push(2, 'b');
      
      const other = new NumberAggregate();
      other.push(3, 'c');
      other.push(4, 'd');
      
      aggregate.merge(other);
      
      const collection = aggregate.getCollection();
      expect(collection[0]).toEqual([1, 'a']);
      expect(collection[1]).toEqual([2, 'b']);
      expect(collection[2]).toEqual([3, 'c']);
      expect(collection[3]).toEqual([4, 'd']);
    });

    it('should handle multiple merges', () => {
      aggregate.push(10, 'a');
      
      const other1 = new NumberAggregate();
      other1.push(20, 'b');
      
      const other2 = new NumberAggregate();
      other2.push(30, 'c');
      
      aggregate.merge(other1);
      aggregate.merge(other2);
      
      expect(aggregate.getRunningSum()).toBe(60);
      expect(aggregate.getRunningAverage()).toBe(20);
    });
  });

  describe('getCollection', () => {
    it('should return copy of collection', () => {
      aggregate.push(10, 'a');
      const collection1 = aggregate.getCollection();
      const collection2 = aggregate.getCollection();
      
      expect(collection1).toEqual(collection2);
      expect(collection1).not.toBe(collection2);
    });

    it('should not allow external modification', () => {
      aggregate.push(10, 'a');
      const collection = aggregate.getCollection();
      collection.push([999, 'hacked']);
      
      expect(aggregate.getCollection()).toHaveLength(1);
    });
  });

  describe('getRunningSum', () => {
    it('should return current sum', () => {
      aggregate.push(5, 'a');
      aggregate.push(10, 'b');
      aggregate.push(15, 'c');
      
      expect(aggregate.getRunningSum()).toBe(30);
    });
  });

  describe('getRunningAverage', () => {
    it('should return current average', () => {
      aggregate.push(10, 'a');
      aggregate.push(20, 'b');
      aggregate.push(30, 'c');
      
      expect(aggregate.getRunningAverage()).toBe(20);
    });

    it('should handle single value', () => {
      aggregate.push(42, 'answer');
      expect(aggregate.getRunningAverage()).toBe(42);
    });
  });

  describe('toJSON', () => {
    it('should return JSON-serializable object', () => {
      aggregate.push(10, 'a');
      aggregate.push(20, 'b');
      
      const json = aggregate.toJSON();
      expect(json).toEqual({
        collection: [[10, 'a'], [20, 'b']],
        running_sum: 30,
        running_average: 15,
      });
    });

    it('should be serializable to JSON string', () => {
      aggregate.push(5, 'test');
      const jsonString = JSON.stringify(aggregate.toJSON());
      const parsed = JSON.parse(jsonString);
      
      expect(parsed.running_sum).toBe(5);
      expect(parsed.running_average).toBe(5);
    });
  });

  describe('toString', () => {
    it('should return JSON string representation', () => {
      aggregate.push(100, 'value');
      const str = aggregate.toString();
      const parsed = JSON.parse(str);
      
      expect(parsed.running_sum).toBe(100);
      expect(parsed.collection[0]).toEqual([100, 'value']);
    });
  });
});
