/**
 * Tests for stats utilities.
 *
 * Tests for StatsByLabel and NumberAggregate classes.
 */

import { ConfigError } from '../../core/exceptions';
import { NumberAggregate, StatsByLabel } from '../stats';

describe('StatsByLabel', () => {
  describe('constructor', () => {
    it('should initialize with zero counts for all labels', () => {
      const stats = new StatsByLabel('processed', 'skipped', 'failed');
      expect(stats.labelCounts).toEqual({
        processed: 0,
        skipped: 0,
        failed: 0,
      });
    });

    it('should handle empty labels', () => {
      const stats = new StatsByLabel();
      expect(stats.labelCounts).toEqual({});
    });
  });

  describe('push', () => {
    it('should increment count for valid label', () => {
      const stats = new StatsByLabel('processed', 'skipped');
      stats.push('processed');
      expect(stats.labelCounts.processed).toBe(1);
      expect(stats.labelCounts.skipped).toBe(0);
    });

    it('should increment by custom amount', () => {
      const stats = new StatsByLabel('processed');
      stats.push('processed', 5);
      expect(stats.labelCounts.processed).toBe(5);
    });

    it('should throw ConfigError for unknown label', () => {
      const stats = new StatsByLabel('processed');
      expect(() => stats.push('unknown')).toThrow(ConfigError);
      expect(() => stats.push('unknown')).toThrow('Unknown label passed to stats by label');
    });

    it('should accumulate counts', () => {
      const stats = new StatsByLabel('processed');
      stats.push('processed', 2);
      stats.push('processed', 3);
      expect(stats.labelCounts.processed).toBe(5);
    });
  });

  describe('toJSON', () => {
    it('should serialize to JSON', () => {
      const stats = new StatsByLabel('processed', 'skipped');
      stats.push('processed', 2);
      stats.push('skipped', 1);

      const json = stats.toJSON();
      expect(json).toEqual({
        label_counts: {
          processed: 2,
          skipped: 1,
        },
      });
    });
  });

  describe('toString', () => {
    it('should return JSON string', () => {
      const stats = new StatsByLabel('processed');
      stats.push('processed', 1);
      const str = stats.toString();
      expect(str).toContain('label_counts');
      expect(str).toContain('processed');
      expect(str).toContain('1');
    });
  });
});

describe('NumberAggregate', () => {
  describe('constructor', () => {
    it('should initialize with empty collection', () => {
      const agg = new NumberAggregate();
      expect(agg.collection).toEqual([]);
      expect(agg.runningSum).toBe(0);
      expect(agg.runningAverage).toBe(0);
    });
  });

  describe('push', () => {
    it('should add number with label', () => {
      const agg = new NumberAggregate();
      agg.push(10, 'field1');
      expect(agg.collection).toEqual([[10, 'field1']]);
      expect(agg.runningSum).toBe(10);
      expect(agg.runningAverage).toBe(10);
    });

    it('should calculate running average correctly', () => {
      const agg = new NumberAggregate();
      agg.push(10, 'field1');
      agg.push(20, 'field2');
      agg.push(30, 'field3');

      expect(agg.runningSum).toBe(60);
      expect(agg.runningAverage).toBe(20);
      expect(agg.collection.length).toBe(3);
    });

    it('should handle zero values', () => {
      const agg = new NumberAggregate();
      agg.push(0, 'field1');
      expect(agg.runningSum).toBe(0);
      expect(agg.runningAverage).toBe(0);
    });

    it('should handle negative values', () => {
      const agg = new NumberAggregate();
      agg.push(-10, 'field1');
      agg.push(20, 'field2');
      expect(agg.runningSum).toBe(10);
      expect(agg.runningAverage).toBe(5);
    });
  });

  describe('merge', () => {
    it('should merge two aggregates', () => {
      const agg1 = new NumberAggregate();
      agg1.push(10, 'field1');
      agg1.push(20, 'field2');

      const agg2 = new NumberAggregate();
      agg2.push(30, 'field3');
      agg2.push(40, 'field4');

      agg1.merge(agg2);

      expect(agg1.collection.length).toBe(4);
      expect(agg1.runningSum).toBe(100);
      expect(agg1.runningAverage).toBe(25);
    });

    it('should handle merging empty aggregate', () => {
      const agg1 = new NumberAggregate();
      agg1.push(10, 'field1');

      const agg2 = new NumberAggregate();
      agg1.merge(agg2);

      expect(agg1.collection.length).toBe(1);
      expect(agg1.runningSum).toBe(10);
      expect(agg1.runningAverage).toBe(10);
    });
  });

  describe('toJSON', () => {
    it('should serialize to JSON', () => {
      const agg = new NumberAggregate();
      agg.push(10, 'field1');
      agg.push(20, 'field2');

      const json = agg.toJSON();
      expect(json).toEqual({
        collection: [
          [10, 'field1'],
          [20, 'field2'],
        ],
        running_sum: 30,
        running_average: 15,
      });
    });
  });
});

