/**
 * OMRChecker — main entry point for the OMRChecker JS browser port.
 *
 * Equivalent to Python's process_single_file() / OMR pipeline coordinator.
 *
 * Steps:
 *   1. Decode base64 image → grayscale cv.Mat
 *   2. Build Template from templateJson
 *   3. Run pre-processors listed in template.preProcessorsConfig
 *      (CropPage and/or CropOnMarkers)
 *   4. Resize to template.processingImageShape
 *   5. Run BubbleReader on all fields
 *   6. Return { response, processedImageDimensions }
 *
 * Ported from Python:
 *   src/omr.py / process_single_file
 *   src/processors/image/CropPage.py
 *   src/processors/image/CropOnMarkers.py
 *
 * bd issue: omr-ojy
 */

import cv from '@techstark/opencv-js';
import { Template, TemplateJson } from './template/Template';
import { BubbleReader, BubbleReaderConfig, OMRResponse } from './detection/BubbleReader';
import { CropPage } from './processors/image/CropPage';
import { CropOnMarkers, CropOnMarkersOptions } from './processors/image/CropOnMarkers';

// ── Public interfaces ─────────────────────────────────────────────────────────

/**
 * Options for processing a single OMR image.
 */
export interface ProcessSingleFileOptions {
  /**
   * Base64-encoded image (JPEG or PNG).
   * May be a raw base64 string or a data URL (data:image/jpeg;base64,…).
   */
  imageBase64: string;

  /**
   * Parsed template.json object.
   * Supports both camelCase and snake_case field names.
   */
  templateJson: TemplateJson;

  /**
   * Asset files required by pre-processors (e.g. reference marker images).
   * Maps filename key (as referenced in template.json) → base64-encoded image data.
   *
   * Required when template uses CropOnMarkers (key: template's referenceImage value).
   */
  assets?: Record<string, string>;

  /** Optional BubbleReader config overrides (threshold tuning). */
  bubbleReaderConfig?: BubbleReaderConfig;
}

/**
 * Result of processing a single OMR image.
 */
export interface OMRResult {
  /** Detected bubble values keyed by field label. */
  response: OMRResponse;
  /**
   * Dimensions of the processed image after pre-processing and resize,
   * as [width, height].
   */
  processedImageDimensions: [number, number];
}

// ── OMRChecker ────────────────────────────────────────────────────────────────

/**
 * Main OMR processing class.
 *
 * All methods are static — instantiate only if you need to share state.
 */
export class OMRChecker {
  /**
   * Process a single OMR sheet image and return the detected bubble responses.
   *
   * This is the primary entry point for the OMRChecker JS port, equivalent to
   * Python's process_single_file().
   *
   * @throws {Error} if image decoding, pre-processing, or bubble reading fails
   */
  static async processSingleFile(options: ProcessSingleFileOptions): Promise<OMRResult> {
    const { imageBase64, templateJson, assets = {}, bubbleReaderConfig = {} } = options;

    // Step 1: Decode base64 image → grayscale cv.Mat
    const sourceGray = await OMRChecker.decodeBase64ToGray(imageBase64);

    try {
      // Step 2: Build Template
      const template = Template.fromJSON(templateJson);

      // Step 3: Run pre-processors
      // Port of Python: ImageProcessorBase.apply_filter_with_context
      //   Each preprocessor optionally has its own processing_image_shape (from options).
      //   The image is resized to that shape before the preprocessor runs.
      //   After all preprocessors, the image is resized to template.templateDimensions
      //   which is the coordinate space used by all field block origins.
      let processedImage = sourceGray.clone();

      for (const ppConfig of template.preProcessorsConfig) {
        // Resize to the preprocessor's processing_image_shape if specified.
        // template.json uses camelCase (processingImageShape) or snake_case (processing_image_shape).
        const ppOptions = ppConfig.options ?? {};
        const ppShape: [number, number] | undefined =
          ppOptions.processingImageShape ??
          ppOptions.processing_image_shape ??
          undefined;

        if (ppShape) {
          const [ppH, ppW] = ppShape; // [height, width]
          const ppResized = new cv.Mat();
          cv.resize(processedImage, ppResized, new cv.Size(ppW, ppH));
          processedImage.delete();
          processedImage = ppResized;
        }

        const next = await OMRChecker.runPreprocessor(ppConfig, processedImage, assets);
        processedImage.delete();
        processedImage = next;
      }

      // Step 4: Resize to templateDimensions for bubble detection.
      // Python: ImageUtils.resize_to_dimensions(template.template_dimensions, gray_image)
      // template.templateDimensions = [width, height]
      const [templateW, templateH] = template.templateDimensions;
      const resized = new cv.Mat();
      cv.resize(processedImage, resized, new cv.Size(templateW, templateH));
      processedImage.delete();

      // Step 5: Read bubbles
      const reader = new BubbleReader(bubbleReaderConfig);
      const response = reader.readBubbles(resized, template);

      const dims: [number, number] = [resized.cols, resized.rows];
      resized.delete();

      return { response, processedImageDimensions: dims };
    } catch (err) {
      // Ensure sourceGray is released on error paths where we did not clone it yet
      try {
        if (!sourceGray.isDeleted()) {
          sourceGray.delete();
        }
      } catch {
        // ignore cleanup errors
      }
      throw err;
    }
  }

  // ── Private helpers ───────────────────────────────────────────────────────────

  /**
   * Decode a base64-encoded image to a grayscale cv.Mat via the browser's
   * Image / canvas API (works without cv.imdecode / Node file I/O).
   *
   * @param base64 - Base64 string or data URL (data:image/…;base64,…)
   * @returns Promise<cv.Mat> grayscale image — caller must delete
   */
  private static decodeBase64ToGray(base64: string): Promise<cv.Mat> {
    const dataUrl = base64.startsWith('data:')
      ? base64
      : `data:image/jpeg;base64,${base64}`;

    return new Promise<cv.Mat>((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('OMRChecker: could not create canvas 2D context'));
          return;
        }
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const rgba = cv.matFromImageData(imageData);
        const gray = new cv.Mat();
        cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
        rgba.delete();
        resolve(gray);
      };
      img.onerror = () =>
        reject(new Error('OMRChecker: failed to load image from base64 data'));
      img.src = dataUrl;
    });
  }

  /**
   * Apply a single pre-processor by name to an image.
   *
   * Supported processors:
   *   - CropPage        (no async factory needed)
   *   - CropOnMarkers   (async factory; needs base64 asset for marker image)
   *
   * Unknown processors are silently skipped with a console.warn.
   *
   * @param ppConfig      - Pre-processor config from template.preProcessorsConfig
   * @param image         - Current grayscale cv.Mat (caller retains ownership)
   * @param assets        - Base64 assets map (filename → base64)
   * @returns New grayscale cv.Mat (caller must delete)
   */
  private static async runPreprocessor(
    ppConfig: { name: string; options: Record<string, any> },
    image: cv.Mat,
    assets: Record<string, string>,
  ): Promise<cv.Mat> {
    const { name, options } = ppConfig;

    if (name === 'CropPage') {
      // CropPage accepts camelCase options natively (morphKernel, useColoredCanny)
      const cropPage = new CropPage(options);
      try {
        const [warped] = cropPage.applyFilter(image, null, null, '');
        return warped;
      } finally {
        cropPage.dispose();
      }
    }

    if (name === 'CropOnMarkers') {
      // template.json uses camelCase for CropOnMarkers options; remap to snake_case
      const normalized = OMRChecker.normalizeCropOnMarkersOptions(options);
      const cropOnMarkers = await CropOnMarkers.fromBase64(normalized, assets);
      try {
        const [warped] = cropOnMarkers.applyFilter(image, null, null, '');
        return warped;
      } finally {
        cropOnMarkers.dispose();
      }
    }

    console.warn(`OMRChecker: unknown pre-processor '${name}', skipping`);
    return image.clone();
  }

  /**
   * Normalize CropOnMarkers options from template.json (camelCase) to the
   * snake_case keys expected by the CropOnMarkers constructor.
   *
   * template.json uses:
   *   referenceImage, markerDimensions, tuningOptions.markerRescaleRange, etc.
   *
   * CropOnMarkers expects:
   *   reference_image, marker_dimensions, tuning_options.marker_rescale_range, etc.
   *
   * Falls back to already-snake_case keys when present (idempotent).
   */
  private static normalizeCropOnMarkersOptions(
    options: Record<string, any>,
  ): CropOnMarkersOptions {
    const type = options.type ?? 'FOUR_MARKERS';

    const referenceImage: string =
      options.reference_image ??
      options.referenceImage ??
      '';

    const markerDimensions: [number, number] | undefined =
      options.marker_dimensions ??
      options.markerDimensions ??
      undefined;

    // Normalize tuningOptions → tuning_options (and inner camelCase keys)
    const rawTuning: Record<string, any> =
      options.tuning_options ??
      options.tuningOptions ??
      {};

    const tuning_options: Record<string, any> = {
      warp_method: rawTuning.warp_method ?? rawTuning.warpMethod,
      min_matching_threshold: rawTuning.min_matching_threshold ?? rawTuning.minMatchingThreshold,
      marker_rescale_range: rawTuning.marker_rescale_range ?? rawTuning.markerRescaleRange,
      marker_rescale_steps: rawTuning.marker_rescale_steps ?? rawTuning.markerRescaleSteps,
      apply_erode_subtract: rawTuning.apply_erode_subtract ?? rawTuning.applyErodeSubtract,
    };

    // Remove undefined keys to avoid overriding CropOnMarkers defaults with undefined
    for (const key of Object.keys(tuning_options)) {
      if (tuning_options[key] === undefined) {
        delete tuning_options[key];
      }
    }

    return {
      type,
      reference_image: referenceImage,
      ...(markerDimensions !== undefined ? { marker_dimensions: markerDimensions } : {}),
      ...(Object.keys(tuning_options).length > 0 ? { tuning_options } : {}),
    } as CropOnMarkersOptions;
  }
}
