/**
 * Typed models for detection results.
 *
 * Port of Python's src/processors/detection/models/detection_results.py
 * Provides strongly-typed, validated models to replace dictionary-based aggregates.
 */

/**
 * Quality assessment of scanned field.
 */
export enum ScanQuality {
  EXCELLENT = 'excellent',  // Clear contrast, high confidence
  GOOD = 'good',           // Acceptable quality
  ACCEPTABLE = 'acceptable', // Marginal, may need review
  POOR = 'poor'            // Likely to have errors
}

/**
 * Represents a bubble location with dimensions and label.
 * Simplified interface for TypeScript (maps to Python's BubblesScanBox).
 */
export interface BubbleLocation {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

/**
 * Single bubble mean intensity value with metadata.
 */
export class BubbleMeanValue {
  constructor(
    public meanValue: number,
    public unitBubble: BubbleLocation,
    public position: [number, number] = [0, 0]
  ) {}

  /**
   * Enable comparison for sorting.
   */
  compareTo(other: BubbleMeanValue): number {
    return this.meanValue - other.meanValue;
  }

  /**
   * String representation for debugging.
   */
  toString(): string {
    return `BubbleMean(${this.meanValue.toFixed(1)})`;
  }
}

/**
 * Typed result for bubble field detection.
 *
 * Replaces nested dictionary structure with strongly-typed model.
 * Includes auto-calculated properties to eliminate utility functions.
 */
export class BubbleFieldDetectionResult {
  constructor(
    public fieldId: string,
    public fieldLabel: string,
    public bubbleMeans: BubbleMeanValue[],
    public timestamp: Date = new Date()
  ) {}

  /**
   * Calculate standard deviation from bubble means.
   */
  get stdDeviation(): number {
    if (this.bubbleMeans.length === 0) {
      return 0.0;
    }

    const values = this.bubbleMeans.map(bm => bm.meanValue);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;

    return Math.sqrt(variance);
  }

  /**
   * Automatically assess scan quality based on std deviation.
   */
  get scanQuality(): ScanQuality {
    const std = this.stdDeviation;

    if (std > 50) return ScanQuality.EXCELLENT;
    if (std > 30) return ScanQuality.GOOD;
    if (std > 15) return ScanQuality.ACCEPTABLE;
    return ScanQuality.POOR;
  }

  /**
   * Check if detection is reliable enough for interpretation.
   */
  get isReliable(): boolean {
    return this.scanQuality === ScanQuality.EXCELLENT ||
           this.scanQuality === ScanQuality.GOOD;
  }

  /**
   * Get bubble means sorted by value.
   */
  get sortedBubbleMeans(): BubbleMeanValue[] {
    return [...this.bubbleMeans].sort((a, b) => a.compareTo(b));
  }

  /**
   * Get just the mean values as numbers.
   */
  get meanValues(): number[] {
    return this.bubbleMeans.map(bm => bm.meanValue);
  }

  /**
   * Get sorted mean values as numbers.
   */
  get sortedMeanValues(): number[] {
    return this.sortedBubbleMeans.map(bm => bm.meanValue);
  }

  /**
   * Calculate jumps between consecutive sorted bubble means.
   *
   * Returns list of [jump_size, bubble_after_jump] tuples.
   * Replaces get_jumps_in_bubble_means utility function.
   */
  get jumps(): Array<[number, BubbleMeanValue]> {
    const sortedMeans = this.sortedBubbleMeans;

    if (sortedMeans.length < 2) {
      return [];
    }

    const jumpsList: Array<[number, BubbleMeanValue]> = [];

    for (let i = 1; i < sortedMeans.length; i++) {
      const jump = sortedMeans[i].meanValue - sortedMeans[i - 1].meanValue;
      jumpsList.push([Math.round(jump * 100) / 100, sortedMeans[i - 1]]);
    }

    return jumpsList;
  }

  /**
   * Get maximum jump between consecutive sorted means.
   */
  get maxJump(): number {
    if (this.bubbleMeans.length < 2) {
      return 0.0;
    }

    const jumps = this.jumps;

    if (jumps.length === 0) {
      return 0.0;
    }

    return Math.max(...jumps.map(([jump]) => jump));
  }

  /**
   * Get minimum bubble mean value.
   */
  get minMean(): number {
    if (this.bubbleMeans.length === 0) {
      return 0.0;
    }
    return Math.min(...this.meanValues);
  }

  /**
   * Get maximum bubble mean value.
   */
  get maxMean(): number {
    if (this.bubbleMeans.length === 0) {
      return 255.0;
    }
    return Math.max(...this.meanValues);
  }

  /**
   * Return number of bubbles.
   */
  get length(): number {
    return this.bubbleMeans.length;
  }

  /**
   * Readable representation.
   */
  toString(): string {
    return `BubbleFieldDetectionResult(field=${this.fieldLabel}, ` +
           `bubbles=${this.length}, std=${this.stdDeviation.toFixed(1)}, ` +
           `quality=${this.scanQuality})`;
  }
}

/**
 * Typed result for OCR field detection.
 */
export class OCRFieldDetectionResult {
  constructor(
    public fieldId: string,
    public fieldLabel: string,
    public detections: any[], // OCRDetection objects
    public confidence: number = 0.0,
    public timestamp: Date = new Date()
  ) {}

  toString(): string {
    return `OCRFieldDetectionResult(field=${this.fieldLabel}, ` +
           `detections=${this.detections.length}, confidence=${this.confidence.toFixed(2)})`;
  }
}

/**
 * Typed result for barcode field detection.
 */
export class BarcodeFieldDetectionResult {
  constructor(
    public fieldId: string,
    public fieldLabel: string,
    public detections: any[], // BarcodeDetection objects
    public timestamp: Date = new Date()
  ) {}

  toString(): string {
    return `BarcodeFieldDetectionResult(field=${this.fieldLabel}, ` +
           `detections=${this.detections.length})`;
  }
}

/**
 * All detection results for a single file.
 *
 * Replaces nested file_level_aggregates dictionary structure.
 */
export class FileDetectionResults {
  public bubbleFields: Map<string, BubbleFieldDetectionResult> = new Map();
  public ocrFields: Map<string, OCRFieldDetectionResult> = new Map();
  public barcodeFields: Map<string, BarcodeFieldDetectionResult> = new Map();

  constructor(
    public filePath: string,
    public timestamp: Date = new Date()
  ) {}

  /**
   * Get result for any field type.
   */
  getFieldResult(fieldId: string): BubbleFieldDetectionResult | OCRFieldDetectionResult | BarcodeFieldDetectionResult {
    if (this.bubbleFields.has(fieldId)) {
      return this.bubbleFields.get(fieldId)!;
    }
    if (this.ocrFields.has(fieldId)) {
      return this.ocrFields.get(fieldId)!;
    }
    if (this.barcodeFields.has(fieldId)) {
      return this.barcodeFields.get(fieldId)!;
    }

    throw new Error(`Field ${fieldId} not found in detection results`);
  }

  /**
   * Get all bubble means across all bubble fields.
   */
  get allBubbleMeans(): BubbleMeanValue[] {
    const means: BubbleMeanValue[] = [];

    for (const fieldResult of this.bubbleFields.values()) {
      means.push(...fieldResult.bubbleMeans);
    }

    return means;
  }

  /**
   * Get all bubble mean values as numbers.
   */
  get allBubbleMeanValues(): number[] {
    return this.allBubbleMeans.map(bm => bm.meanValue);
  }

  /**
   * Total number of fields detected.
   */
  get numFields(): number {
    return this.bubbleFields.size + this.ocrFields.size + this.barcodeFields.size;
  }

  toString(): string {
    return `FileDetectionResults(file=${this.filePath}, fields=${this.numFields})`;
  }
}

