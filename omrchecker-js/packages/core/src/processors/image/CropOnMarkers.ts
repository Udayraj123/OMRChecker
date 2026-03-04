/**
 * CropOnMarkers processor — TypeScript port of Python:
 *   src/processors/image/CropOnMarkers.py  (thin wrapper)
 *   src/processors/image/crop_on_patches/custom_markers.py  (CropOnCustomMarkers — main logic)
 *
 * Handles the FOUR_MARKERS layout type: detects 4 corner markers via template matching
 * and perspective-warps the image to align the sheet.
 *
 * Browser-compatible: accepts marker reference image as a pre-decoded grayscale cv.Mat
 * via the `assetMats` map. For convenience, use the static async factory method
 * `CropOnMarkers.fromBase64(options, assets)` which decodes the image via canvas.
 *
 * Extends WarpOnPointsCommon and implements the 3 abstract methods:
 *   - validateAndRemapOptionsSchema
 *   - prepareImageBeforeExtraction
 *   - extractControlDestinationPoints
 *
 * Port of Python: omr-1lv
 */

import cv from '@techstark/opencv-js';
import { WarpOnPointsCommon } from './WarpOnPointsCommon';
import { prepareMarkerTemplate, detectMarkerInPatch } from './crop_on_patches/marker_detection';
import type { ReferenceZone } from './crop_on_patches/marker_detection';
import { ImageUtils } from '../../utils/image';
import { ImageProcessingError } from '../../utils/exceptions';
import { WarpMethod } from '../constants';

// ── Zone order (matches Python MARKER_ZONE_TYPES_IN_ORDER) ───────────────────

const FOUR_MARKERS_ZONE_ORDER = [
  'topLeftMarker',
  'topRightMarker',
  'bottomRightMarker',
  'bottomLeftMarker',
] as const;

type MarkerZoneType = (typeof FOUR_MARKERS_ZONE_ORDER)[number];

// ── Options interface ─────────────────────────────────────────────────────────

export interface CropOnMarkersOptions {
  /** Must be 'FOUR_MARKERS' (the only supported type). */
  type: 'FOUR_MARKERS';
  /** Key into the `assetMats` map for the reference marker image. */
  reference_image: string;
  /** Optional [width, height] to resize the extracted marker template. */
  marker_dimensions?: [number, number];
  tuning_options?: {
    warp_method?: string;
    min_matching_threshold?: number;
    marker_rescale_range?: [number, number];
    marker_rescale_steps?: number;
    apply_erode_subtract?: boolean;
  };
  [key: string]: unknown;
}

// ── CropOnMarkers ─────────────────────────────────────────────────────────────

/**
 * Detects 4 corner markers in an image and perspective-warps it to align the sheet.
 *
 * Usage (synchronous, caller pre-decodes the marker image):
 * ```ts
 * const markerMat = /* grayscale cv.Mat decoded from omr_marker.jpg *\/;
 * const processor = new CropOnMarkers(
 *   { type: 'FOUR_MARKERS', reference_image: 'omr_marker.jpg', marker_dimensions: [35, 35] },
 *   { 'omr_marker.jpg': markerMat }
 * );
 * const [warped, warpedColored, template] = processor.applyFilter(gray, null, null, 'sheet.jpg');
 * processor.dispose();
 * ```
 *
 * Usage (async factory — decodes base64 via canvas/Image API):
 * ```ts
 * const processor = await CropOnMarkers.fromBase64(
 *   { type: 'FOUR_MARKERS', reference_image: 'omr_marker.jpg' },
 *   { 'omr_marker.jpg': base64String }
 * );
 * ```
 */
export class CropOnMarkers extends WarpOnPointsCommon {
  private readonly referenceImageKey: string;
  private readonly markerDimensions: [number, number] | undefined;
  private readonly minMatchingThreshold: number;
  private readonly markerRescaleRange: [number, number];
  private readonly markerRescaleSteps: number;
  private readonly applyErodeSubtract: boolean;

  /**
   * Prepared marker template Mats keyed by zone type.
   * Populated in initResizedMarkers(); each Mat must be deleted via dispose().
   */
  private markerTemplates: Map<MarkerZoneType, cv.Mat> = new Map();

  /**
   * @param options   - CropOnMarkers options
   * @param assetMats - Map of filename key → pre-decoded grayscale cv.Mat
   *                    The caller retains ownership of the input Mats;
   *                    this class clones/processes them internally.
   */
  constructor(
    options: CropOnMarkersOptions,
    assetMats: Record<string, cv.Mat>
  ) {
    // super() calls validateAndRemapOptionsSchema() via polymorphism.
    super(options as Record<string, any>);

    if (options.type !== 'FOUR_MARKERS') {
      throw new ImageProcessingError(
        `CropOnMarkers: unsupported type '${options.type}'. Only 'FOUR_MARKERS' is supported.`
      );
    }

    this.referenceImageKey = options.reference_image;
    this.markerDimensions = options.marker_dimensions;

    const tuning = options.tuning_options ?? {};
    this.minMatchingThreshold = tuning.min_matching_threshold ?? 0.3;
    this.markerRescaleRange = tuning.marker_rescale_range ?? [85, 115];
    this.markerRescaleSteps = tuning.marker_rescale_steps ?? 5;
    this.applyErodeSubtract = tuning.apply_erode_subtract ?? true;

    const refMat = assetMats[this.referenceImageKey];
    if (!refMat) {
      throw new ImageProcessingError(
        `CropOnMarkers: asset Mat not found for key '${this.referenceImageKey}'`,
        { key: this.referenceImageKey, availableKeys: Object.keys(assetMats) }
      );
    }

    this.initResizedMarkers(refMat);
  }

  /**
   * Async factory that decodes a base64-encoded image via the browser's
   * Image/canvas API and constructs a CropOnMarkers instance.
   *
   * @param options - CropOnMarkers options (reference_image key must exist in assets)
   * @param assets  - Map of filename key → base64 string (data URL or raw base64)
   * @returns Initialized CropOnMarkers instance (call .dispose() when done)
   */
  static async fromBase64(
    options: CropOnMarkersOptions,
    assets: Record<string, string>
  ): Promise<CropOnMarkers> {
    const key = options.reference_image;
    const b64 = assets[key];
    if (!b64) {
      throw new ImageProcessingError(
        `CropOnMarkers.fromBase64: asset '${key}' not found in assets map`,
        { key, availableKeys: Object.keys(assets) }
      );
    }

    const refMat = await CropOnMarkers.decodeBase64ViaCanvas(b64);
    try {
      return new CropOnMarkers(options, { [key]: refMat });
    } finally {
      // The constructor clones/processes the Mat; we can delete the source.
      refMat.delete();
    }
  }

  // ── Abstract method implementations ──────────────────────────────────────────

  validateAndRemapOptionsSchema(options: Record<string, any>): Record<string, any> {
    const tuning = options['tuning_options'] ?? {};
    return {
      enable_cropping: true,
      points_layout: options['type'] ?? 'FOUR_MARKERS',
      tuning_options: {
        warp_method: tuning['warp_method'] ?? WarpMethod.PERSPECTIVE_TRANSFORM,
      },
    };
  }

  /**
   * Normalize the image before marker extraction.
   * Port of Python CropOnCustomMarkers.prepare_image_before_extraction.
   */
  prepareImageBeforeExtraction(image: cv.Mat): cv.Mat {
    return ImageUtils.normalizeSingle(image);
  }

  /**
   * Detect all 4 corner markers and return their centers as control points.
   *
   * Returns [controlPoints, warpedPoints, null].
   * controlPoints = 4 center points of the detected markers (one per zone).
   * warpedPoints = corresponding destination points.
   */
  extractControlDestinationPoints(
    image: cv.Mat,
    _coloredImage: cv.Mat | null,
    filePath: string
  ): [number[][], number[][], null] {
    const allCorners: number[][] = [];

    for (const zoneType of FOUR_MARKERS_ZONE_ORDER) {
      const marker = this.markerTemplates.get(zoneType);
      if (!marker) {
        throw new ImageProcessingError(
          `CropOnMarkers: marker template not initialized for zone '${zoneType}'`,
          { filePath, zoneType }
        );
      }

      const zoneDesc = this.getQuadrantZoneDescription(zoneType, image, marker);
      const markerCorners = this.findMarkerCornersInPatch(image, zoneDesc, marker, zoneType, filePath);

      // Use the center of the marker as the control point
      const centerX = (markerCorners[0][0] + markerCorners[1][0] + markerCorners[2][0] + markerCorners[3][0]) / 4;
      const centerY = (markerCorners[0][1] + markerCorners[1][1] + markerCorners[2][1] + markerCorners[3][1]) / 4;
      allCorners.push([centerX, centerY]);
    }

    // Compute destination rectangle points from ordered corners
    const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(allCorners);

    return [allCorners, warpedPoints, null];
  }

  // ── Private helpers ───────────────────────────────────────────────────────────

  /**
   * Prepare a marker template Mat for each zone from the reference image.
   * All zones in FOUR_MARKERS share the same reference image.
   *
   * @param referenceImage - pre-decoded grayscale cv.Mat (not modified)
   */
  private initResizedMarkers(referenceImage: cv.Mat): void {
    const referenceZone = CropOnMarkers.getDefaultScanZoneForImage(referenceImage);

    for (const zoneType of FOUR_MARKERS_ZONE_ORDER) {
      const marker = prepareMarkerTemplate(
        referenceImage,
        referenceZone,
        this.markerDimensions,
        [5, 5],
        this.applyErodeSubtract
      );
      this.markerTemplates.set(zoneType, marker);
    }
  }

  /**
   * Decode a base64-encoded image to a grayscale cv.Mat using the
   * browser's Image/canvas API (works without cv.imdecode).
   *
   * @param b64 - base64 string (data URL or raw base64)
   * @returns Promise<cv.Mat> grayscale image (caller must delete)
   */
  private static async decodeBase64ViaCanvas(b64: string): Promise<cv.Mat> {
    const dataUrl = b64.startsWith('data:') ? b64 : `data:image/jpeg;base64,${b64}`;

    const img = new Image();
    img.src = dataUrl;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error(`Failed to load image from base64 data`));
    });

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new ImageProcessingError('CropOnMarkers: could not create canvas 2D context');
    }
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const rgba = cv.matFromImageData(imageData);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete();
    return gray;
  }

  /**
   * Compute the default reference zone covering the full image.
   * Port of Python CropOnCustomMarkers.get_default_scan_zone_for_image.
   */
  private static getDefaultScanZoneForImage(image: cv.Mat): ReferenceZone {
    return {
      origin: [1, 1],
      dimensions: [image.cols - 1, image.rows - 1],
    };
  }

  /**
   * Compute the zone bounds (origin, dimensions, margins) for a given quadrant.
   *
   * Port of Python CropOnCustomMarkers.get_quadrant_zone_description.
   *
   * @param zoneType - Which corner marker zone
   * @param image    - The full image (for dimensions)
   * @param marker   - The marker template (for its dimensions)
   */
  private getQuadrantZoneDescription(
    zoneType: MarkerZoneType,
    image: cv.Mat,
    marker: cv.Mat
  ): {
    origin: [number, number];
    dimensions: [number, number];
    margins: { top: number; right: number; bottom: number; left: number };
  } {
    const h = image.rows;
    const w = image.cols;
    const halfHeight = Math.floor(h / 2);
    const halfWidth = Math.floor(w / 2);
    const markerH = marker.rows;
    const markerW = marker.cols;

    let zoneStart: [number, number];
    let zoneEnd: [number, number];

    if (zoneType === 'topLeftMarker') {
      zoneStart = [1, 1];
      zoneEnd = [halfWidth, halfHeight];
    } else if (zoneType === 'topRightMarker') {
      zoneStart = [halfWidth, 1];
      zoneEnd = [w, halfHeight];
    } else if (zoneType === 'bottomRightMarker') {
      zoneStart = [halfWidth, halfHeight];
      zoneEnd = [w, h];
    } else {
      // bottomLeftMarker
      zoneStart = [1, halfHeight];
      zoneEnd = [halfWidth, h];
    }

    const originX = Math.floor((zoneStart[0] + zoneEnd[0] - markerW) / 2);
    const originY = Math.floor((zoneStart[1] + zoneEnd[1] - markerH) / 2);
    const marginH = (zoneEnd[0] - zoneStart[0] - markerW) / 2 - 1;
    const marginV = (zoneEnd[1] - zoneStart[1] - markerH) / 2 - 1;

    return {
      origin: [originX, originY],
      dimensions: [markerW, markerH],
      margins: {
        top: marginV,
        right: marginH,
        bottom: marginV,
        left: marginH,
      },
    };
  }

  /**
   * Extract the patch for a zone and run detectMarkerInPatch on it.
   *
   * Port of Python CropOnCustomMarkers.find_marker_corners_in_patch + compute_scan_zone.
   *
   * @param image      - Full prepared image
   * @param zoneDesc   - Zone description (origin, dimensions, margins)
   * @param marker     - Marker template
   * @param zoneType   - Zone type label (for error messages)
   * @param filePath   - File path (for error messages)
   * @returns 4 corner points [[x,y], ...] in absolute image coordinates
   */
  private findMarkerCornersInPatch(
    image: cv.Mat,
    zoneDesc: {
      origin: [number, number];
      dimensions: [number, number];
      margins: { top: number; right: number; bottom: number; left: number };
    },
    marker: cv.Mat,
    zoneType: MarkerZoneType,
    filePath: string
  ): number[][] {
    const { origin, dimensions, margins } = zoneDesc;
    const [ox, oy] = origin;
    const [dw, dh] = dimensions;

    // Clamp margins to ≥ 0
    const marginTop = Math.max(0, Math.floor(margins.top));
    const marginRight = Math.max(0, Math.floor(margins.right));
    const marginBottom = Math.max(0, Math.floor(margins.bottom));
    const marginLeft = Math.max(0, Math.floor(margins.left));

    // Compute patch rectangle (origin minus margins)
    const patchX = Math.max(0, ox - marginLeft);
    const patchY = Math.max(0, oy - marginTop);
    const patchW = Math.min(image.cols - patchX, dw + marginLeft + marginRight);
    const patchH = Math.min(image.rows - patchY, dh + marginTop + marginBottom);

    if (patchW <= 0 || patchH <= 0) {
      throw new ImageProcessingError(
        `CropOnMarkers: degenerate patch for zone '${zoneType}'`,
        { filePath, zoneType, patchX, patchY, patchW, patchH }
      );
    }

    const rect = new cv.Rect(patchX, patchY, patchW, patchH);
    const patch = image.roi(rect).clone();

    const zoneOffset: [number, number] = [patchX, patchY];

    let corners: number[][] | null;
    try {
      corners = detectMarkerInPatch(
        patch,
        marker,
        zoneOffset,
        this.markerRescaleRange,
        this.markerRescaleSteps,
        this.minMatchingThreshold
      );
    } finally {
      patch.delete();
    }

    if (corners === null) {
      throw new ImageProcessingError(
        `CropOnMarkers: no marker found in patch for zone '${zoneType}'`,
        { filePath, zoneType }
      );
    }

    return corners;
  }

  /**
   * Release all marker template Mats to free Emscripten WASM memory.
   * Call this when the processor is no longer needed.
   */
  dispose(): void {
    for (const [, mat] of this.markerTemplates) {
      try {
        if (!mat.isDeleted()) {
          mat.delete();
        }
      } catch (_) {
        // ignore
      }
    }
    this.markerTemplates.clear();
  }
}
