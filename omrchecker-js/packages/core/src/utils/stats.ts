/**
 * Statistics utilities for OMRChecker.
 *
 * TypeScript port of src/utils/stats.py
 * Provides label-based statistics tracking and numeric aggregation.
 */

import { ConfigError } from '../core/exceptions';
import { defaultDump } from './parsing';

/**
 * Label-based statistics tracking.
 *
 * Tracks counts for predefined labels (e.g., "processed", "skipped").
 */
export class StatsByLabel {
  public labelCounts: Record<string, number>;

  constructor(...labels: string[]) {
    this.labelCounts = Object.fromEntries(labels.map((label) => [label, 0]));
  }

  /**
   * Increment count for a label.
   *
   * @param label - Label to increment
   * @param number - Amount to increment (default: 1)
   */
  push(label: string, number = 1): void {
    if (!(label in this.labelCounts)) {
      throw new ConfigError(
        `Unknown label passed to stats by label: ${label}, allowed labels: ${Object.keys(this.labelCounts).join(', ')}`,
        {
          label,
          allowed_labels: Object.keys(this.labelCounts),
        }
      );
    }

    this.labelCounts[label] += number;
  }

  /**
   * Serialize to JSON.
   */
  toJSON(): Record<string, unknown> {
    return {
      label_counts: defaultDump(this.labelCounts),
    };
  }

  toString(): string {
    return JSON.stringify(this.toJSON());
  }
}

/**
 * Numeric aggregation with running averages.
 *
 * Tracks a collection of numbers with labels and maintains running statistics.
 */
export class NumberAggregate {
  public collection: Array<[number, string]>;
  public runningSum: number;
  public runningAverage: number;

  constructor() {
    this.collection = [];
    this.runningSum = 0;
    this.runningAverage = 0;
  }

  /**
   * Add a number with a label.
   *
   * @param numberLike - Number to add
   * @param label - Label for this number
   */
  push(numberLike: number, label: string): void {
    this.collection.push([numberLike, label]);
    this.runningSum += numberLike;
    this.runningAverage = this.runningSum / this.collection.length;
  }

  /**
   * Merge another NumberAggregate into this one.
   *
   * @param otherAggregate - Another NumberAggregate to merge
   */
  merge(otherAggregate: NumberAggregate): void {
    this.collection.push(...otherAggregate.collection);
    this.runningSum += otherAggregate.runningSum;
    this.runningAverage = this.runningSum / this.collection.length;
  }

  /**
   * Serialize to JSON.
   */
  toJSON(): Record<string, unknown> {
    return {
      collection: defaultDump(this.collection),
      running_sum: this.runningSum,
      running_average: this.runningAverage,
    };
  }
}

