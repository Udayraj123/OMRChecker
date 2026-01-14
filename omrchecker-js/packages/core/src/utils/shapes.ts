/**
 * Shape utilities for scan zone manipulation and geometry operations.
 *
 * TypeScript port of shape-related utilities from Python implementation.
 * Provides functions for working with scan zones, margins, and image extraction.
 */

import * as cv from '@techstark/opencv-js';
import { Point, Rectangle, MathUtils } from './math';
import { Logger } from './logger';

const logger = new Logger('ShapeUtils');

/**
 * Zone description interface matching processor requirements
 */
export interface ZoneDescription {
  label: string;
  origin?: Point;
  dimensions?: [number, number];
  margins?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

/**
 * Utility class for shape and zone manipulation.
 */
export class ShapeUtils {
  /**
   * Compute scan zone rectangle from zone description.
   *
   * @param zoneDescription - Zone configuration with origin, dimensions, and margins
   * @param includeMargins - Whether to include margins in the rectangle
   * @returns Rectangle with 4 corners [TL, TR, BR, BL]
   */
  static computeScanZoneRectangle(
    zoneDescription: ZoneDescription,
    includeMargins: boolean = true
  ): Rectangle {
    const origin = zoneDescription.origin || [0, 0];
    const dimensions = zoneDescription.dimensions || [0, 0];
    const margins = zoneDescription.margins || { top: 0, right: 0, bottom: 0, left: 0 };

    let [x, y] = origin;
    let [w, h] = dimensions;

    if (includeMargins) {
      x -= margins.left;
      y -= margins.top;
      w += margins.left + margins.right;
      h += margins.top + margins.bottom;
    }

    // Return corners: TL, TR, BR, BL
    return [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ];
  }

  /**
   * Extract image region from zone description.
   *
   * Returns both the extracted image zone and the rectangle coordinates.
   *
   * @param image - Source image
   * @param zoneDescription - Zone configuration
   * @returns Tuple of [extracted zone image, zone rectangle]
   */
  static extractImageFromZoneDescription(
    image: cv.Mat,
    zoneDescription: ZoneDescription
  ): [cv.Mat, Rectangle] {
    const rectangle = this.computeScanZoneRectangle(zoneDescription, true);

    // Get origin and dimensions from rectangle
    const [tl, , br] = rectangle;
    const x = Math.floor(tl[0]);
    const y = Math.floor(tl[1]);
    const w = Math.floor(br[0] - tl[0]);
    const h = Math.floor(br[1] - tl[1]);

    // Validate bounds
    if (x < 0 || y < 0 || x + w > image.cols || y + h > image.rows) {
      logger.warn(
        `Zone ${zoneDescription.label} exceeds image bounds: ` +
        `[${x}, ${y}, ${w}, ${h}] vs image [${image.cols}, ${image.rows}]`
      );

      // Clamp to image bounds
      const clampedX = Math.max(0, x);
      const clampedY = Math.max(0, y);
      const clampedW = Math.min(w, image.cols - clampedX);
      const clampedH = Math.min(h, image.rows - clampedY);

      const rect = new cv.Rect(clampedX, clampedY, clampedW, clampedH);
      const zone = image.roi(rect);

      // Update rectangle with clamped values
      const clampedRectangle: Rectangle = [
        [clampedX, clampedY],
        [clampedX + clampedW, clampedY],
        [clampedX + clampedW, clampedY + clampedH],
        [clampedX, clampedY + clampedH],
      ];

      return [zone, clampedRectangle];
    }

    // Extract zone
    const rect = new cv.Rect(x, y, w, h);
    const zone = image.roi(rect);

    return [zone, rectangle];
  }

  /**
   * Compute scan zone with margins applied.
   *
   * Similar to computeScanZoneRectangle but returns different format
   * for compatibility with legacy code.
   *
   * @param zoneDescription - Zone configuration
   * @returns Object with origin and dimensions
   */
  static computeScanZone(zoneDescription: ZoneDescription): {
    origin: Point;
    dimensions: [number, number];
  } {
    const rectangle = this.computeScanZoneRectangle(zoneDescription, true);
    const [tl, , br] = rectangle;

    return {
      origin: tl,
      dimensions: [br[0] - tl[0], br[1] - tl[1]],
    };
  }

  /**
   * Get bounding box dimensions from a zone description.
   *
   * @param zoneDescription - Zone configuration
   * @param includeMargins - Whether to include margins
   * @returns [width, height] tuple
   */
  static getZoneDimensions(
    zoneDescription: ZoneDescription,
    includeMargins: boolean = true
  ): [number, number] {
    const rectangle = this.computeScanZoneRectangle(zoneDescription, includeMargins);
    const [tl, , br] = rectangle;

    return [br[0] - tl[0], br[1] - tl[1]];
  }

  /**
   * Check if a point is within a zone.
   *
   * @param point - Point to check
   * @param zoneDescription - Zone configuration
   * @param includeMargins - Whether to include margins in check
   * @returns True if point is inside zone
   */
  static isPointInZone(
    point: Point,
    zoneDescription: ZoneDescription,
    includeMargins: boolean = true
  ): boolean {
    const rectangle = this.computeScanZoneRectangle(zoneDescription, includeMargins);
    const [tl, , br] = rectangle;

    return (
      point[0] >= tl[0] &&
      point[0] <= br[0] &&
      point[1] >= tl[1] &&
      point[1] <= br[1]
    );
  }

  /**
   * Compute the center point of a zone.
   *
   * @param zoneDescription - Zone configuration
   * @returns Center point [x, y]
   */
  static getZoneCenter(zoneDescription: ZoneDescription): Point {
    const rectangle = this.computeScanZoneRectangle(zoneDescription, false);
    const [tl, , br] = rectangle;

    return [
      Math.floor((tl[0] + br[0]) / 2),
      Math.floor((tl[1] + br[1]) / 2),
    ];
  }

  /**
   * Apply margins to a zone description (creates new object).
   *
   * @param zoneDescription - Original zone configuration
   * @param margins - Margins to apply
   * @returns New zone description with margins
   */
  static applyMarginsToZone(
    zoneDescription: ZoneDescription,
    margins: { top: number; right: number; bottom: number; left: number }
  ): ZoneDescription {
    return {
      ...zoneDescription,
      margins: {
        ...(zoneDescription.margins || { top: 0, right: 0, bottom: 0, left: 0 }),
        ...margins,
      },
    };
  }

  /**
   * Merge two zone descriptions (right overrides left).
   *
   * @param base - Base zone configuration
   * @param override - Override zone configuration
   * @returns Merged zone description
   */
  static mergeZoneDescriptions(
    base: ZoneDescription,
    override: Partial<ZoneDescription>
  ): ZoneDescription {
    return {
      ...base,
      ...override,
      margins: {
        ...(base.margins || { top: 0, right: 0, bottom: 0, left: 0 }),
        ...(override.margins || {}),
      },
    };
  }

  /**
   * Validate that a zone description is properly formed.
   *
   * @param zoneDescription - Zone to validate
   * @returns True if valid
   * @throws Error if invalid
   */
  static validateZoneDescription(zoneDescription: ZoneDescription): boolean {
    if (!zoneDescription.label) {
      throw new Error('Zone description must have a label');
    }

    if (zoneDescription.origin) {
      const [x, y] = zoneDescription.origin;
      if (x < 0 || y < 0) {
        throw new Error(`Zone ${zoneDescription.label} has negative origin: [${x}, ${y}]`);
      }
    }

    if (zoneDescription.dimensions) {
      const [w, h] = zoneDescription.dimensions;
      if (w <= 0 || h <= 0) {
        throw new Error(
          `Zone ${zoneDescription.label} has invalid dimensions: [${w}, ${h}]`
        );
      }
    }

    return true;
  }

  /**
   * Create a default zone description for an entire image.
   *
   * @param image - Source image
   * @param label - Label for the zone
   * @returns Zone description covering the whole image
   */
  static createFullImageZone(image: cv.Mat, label: string = 'full_image'): ZoneDescription {
    return {
      label,
      origin: [0, 0],
      dimensions: [image.cols, image.rows],
      margins: { top: 0, right: 0, bottom: 0, left: 0 },
    };
  }

  /**
   * Scale a zone description by a factor.
   *
   * @param zoneDescription - Zone to scale
   * @param scale - Scale factor
   * @returns Scaled zone description
   */
  static scaleZoneDescription(
    zoneDescription: ZoneDescription,
    scale: number
  ): ZoneDescription {
    const scaled: ZoneDescription = {
      ...zoneDescription,
    };

    if (zoneDescription.origin) {
      scaled.origin = [
        Math.floor(zoneDescription.origin[0] * scale),
        Math.floor(zoneDescription.origin[1] * scale),
      ];
    }

    if (zoneDescription.dimensions) {
      scaled.dimensions = [
        Math.floor(zoneDescription.dimensions[0] * scale),
        Math.floor(zoneDescription.dimensions[1] * scale),
      ];
    }

    if (zoneDescription.margins) {
      scaled.margins = {
        top: Math.floor(zoneDescription.margins.top * scale),
        right: Math.floor(zoneDescription.margins.right * scale),
        bottom: Math.floor(zoneDescription.margins.bottom * scale),
        left: Math.floor(zoneDescription.margins.left * scale),
      };
    }

    return scaled;
  }
}

