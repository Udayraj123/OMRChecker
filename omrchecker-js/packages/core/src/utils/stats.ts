/**
 * Migrated from Python: src/utils/stats.py
 * Agent: Foundation-Alpha
 * Phase: 1
 *
 * Statistics tracking utilities for aggregating detection results
 */

/**
 * Track counts by label
 * Used for tracking statistics grouped by label (e.g., bubble detection counts per field)
 */
export class StatsByLabel {
  private labelCounts: Record<string, number>;

  constructor(...labels: string[]) {
    this.labelCounts = {};
    for (const label of labels) {
      this.labelCounts[label] = 0;
    }
  }

  /**
   * Increment count for a label
   * 
   * @param label - Label to increment
   * @param number - Amount to increment by (default: 1)
   * @throws Error if label is not in allowed labels
   */
  push(label: string, number: number = 1): void {
    if (!(label in this.labelCounts)) {
      const allowedLabels = Object.keys(this.labelCounts);
      throw new Error(
        `Unknown label passed to stats by label: ${label}, ` +
        `allowed labels: ${allowedLabels.join(', ')}`
      );
    }

    this.labelCounts[label] += number;
  }

  /**
   * Get label counts
   */
  getLabelCounts(): Record<string, number> {
    return { ...this.labelCounts };
  }

  /**
   * Convert to JSON-serializable object
   */
  toJSON(): Record<string, any> {
    return {
      label_counts: this.labelCounts,
    };
  }

  /**
   * Convert to string representation
   */
  toString(): string {
    return JSON.stringify(this.toJSON());
  }
}

/**
 * Aggregate numbers with running statistics
 * Used for collecting and averaging numeric values (e.g., confidence scores)
 */
export class NumberAggregate {
  private collection: Array<[number, string]>;
  private runningSum: number;
  private runningAverage: number;

  constructor() {
    this.collection = [];
    this.runningSum = 0;
    this.runningAverage = 0;
  }

  /**
   * Add a number to the aggregate
   * 
   * @param numberLike - Number to add
   * @param label - Label associated with this number
   */
  push(numberLike: number, label: string): void {
    this.collection.push([numberLike, label]);
    this.runningSum += numberLike;
    this.runningAverage = this.runningSum / this.collection.length;
  }

  /**
   * Merge another aggregate into this one
   * 
   * @param otherAggregate - Another NumberAggregate to merge
   */
  merge(otherAggregate: NumberAggregate): void {
    this.collection.push(...otherAggregate.collection);
    this.runningSum += otherAggregate.runningSum;
    this.runningAverage = this.runningSum / this.collection.length;
  }

  /**
   * Get collection of all values
   */
  getCollection(): Array<[number, string]> {
    return [...this.collection];
  }

  /**
   * Get running sum
   */
  getRunningSum(): number {
    return this.runningSum;
  }

  /**
   * Get running average
   */
  getRunningAverage(): number {
    return this.runningAverage;
  }

  /**
   * Convert to JSON-serializable object
   */
  toJSON(): Record<string, any> {
    return {
      collection: this.collection,
      running_sum: this.runningSum,
      running_average: this.runningAverage,
    };
  }

  /**
   * Convert to string representation
   */
  toString(): string {
    return JSON.stringify(this.toJSON());
  }
}
