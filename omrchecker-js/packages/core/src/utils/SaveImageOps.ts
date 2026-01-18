/**
 * Save image operations for managing debug/intermediate image saves.
 *
 * TypeScript port of src/utils/file.py SaveImageOps class
 * Maintains 1:1 correspondence with Python implementation.
 */

import * as cv from '@techstark/opencv-js';
import { Logger } from './logger';
import { ImageUtils } from './ImageUtils';
import { MathUtils } from './math';

const logger = new Logger('SaveImageOps');

/**
 * Type for image with title: [title, image]
 */
type TitleAndImage = [string, cv.Mat];

/**
 * Save image operations class.
 *
 * Manages saving of debug/intermediate images at different levels.
 * Images are organized by level (1-7) and can be saved as stacks.
 */
export class SaveImageOps {
  private grayImages: Map<number, TitleAndImage[]>;
  private coloredImages: Map<number, TitleAndImage[]>;
  public tuningConfig: any; // TuningConfig type - made public to match interface requirements
  private saveImageLevel: number;

  constructor(tuningConfig: any) {
    this.grayImages = new Map();
    this.coloredImages = new Map();
    this.tuningConfig = tuningConfig;
    this.saveImageLevel = tuningConfig.outputs?.save_image_level || 0;

    // Initialize maps for levels 1-7
    for (let i = 1; i <= 7; i++) {
      this.grayImages.set(i, []);
      this.coloredImages.set(i, []);
    }
  }

  /**
   * Append image to save queue.
   *
   * Port of Python's append_save_image method.
   *
   * @param title - Title/name for the image
   * @param keys - Single key or array of keys (levels 1-7)
   * @param grayImage - Optional grayscale image
   * @param coloredImage - Optional colored image
   */
  appendSaveImage(
    title: string,
    keys: number | number[],
    grayImage?: cv.Mat,
    coloredImage?: cv.Mat
  ): void {
    if (typeof title !== 'string') {
      throw new TypeError(`title=${title} is not a string`);
    }

    const keyArray = Array.isArray(keys) ? keys : [keys];
    let grayImageCopy: cv.Mat | undefined;
    let coloredImageCopy: cv.Mat | undefined;

    if (grayImage !== undefined) {
      grayImageCopy = grayImage.clone();
    }
    if (coloredImage !== undefined) {
      coloredImageCopy = coloredImage.clone();
    }

    for (const key of keyArray) {
      const keyInt = Number(key);
      if (keyInt > this.saveImageLevel) {
        continue;
      }

      if (grayImageCopy !== undefined) {
        const existing = this.grayImages.get(keyInt) || [];
        existing.push([title, grayImageCopy.clone()]);
        this.grayImages.set(keyInt, existing);
      }

      if (coloredImageCopy !== undefined) {
        const existing = this.coloredImages.get(keyInt) || [];
        existing.push([title, coloredImageCopy.clone()]);
        this.coloredImages.set(keyInt, existing);
      }
    }
  }

  /**
   * Save image stacks to disk.
   *
   * Port of Python's save_image_stacks method.
   * Note: In browser environment, this may need to use ImageSaver or download mechanism.
   *
   * @param filePath - Path to the source file (for naming)
   * @param saveMarkedDir - Directory to save stacks
   * @param key - Specific level to save (defaults to saveImageLevel)
   * @param imagesPerRow - Number of images per row in stack (default: 4)
   */
  saveImageStacks(
    filePath: string,
    saveMarkedDir: string,
    key?: number,
    imagesPerRow: number = 4
  ): void {
    const targetKey = key !== undefined ? key : this.saveImageLevel;

    if (this.saveImageLevel < targetKey) {
      return;
    }

    // Extract stem from file path
    const pathParts = filePath.split('/');
    const fileName = pathParts[pathParts.length - 1];
    const stem = fileName.replace(/\.[^/.]+$/, '');

    // Save gray images stack
    const grayImagesForLevel = this.grayImages.get(targetKey) || [];
    if (grayImagesForLevel.length > 0) {
      const titles = grayImagesForLevel.map(([title]) => title);
      logger.info(`Gray Stack level: ${targetKey} - ${titles.join(', ')}`);

      const result = this.getResultHstack(grayImagesForLevel, imagesPerRow);
      const stackPath = `${saveMarkedDir}/stack/${stem}_${targetKey}_stack.jpg`;

      // TODO: Use ImageSaver or file system API to save
      // For now, we'll log the path
      logger.info(`Would save stack image to: ${stackPath}`);
      // ImageUtils.saveImage(result, 'jpg'); // When file saving is implemented

      result.delete();
    } else {
      logger.info(
        `Note: Nothing to save for gray image. Stack level: ${this.saveImageLevel}`
      );
    }

    // Save colored images stack
    const coloredImagesForLevel = this.coloredImages.get(targetKey) || [];
    if (coloredImagesForLevel.length > 0) {
      const titles = coloredImagesForLevel.map(([title]) => title);
      logger.info(`Colored Stack level: ${targetKey} - ${titles.join(', ')}`);

      const coloredResult = this.getResultHstack(
        coloredImagesForLevel,
        imagesPerRow
      );
      const coloredStackPath = `${saveMarkedDir}/stack/colored/${stem}_${targetKey}_stack.jpg`;

      // TODO: Use ImageSaver or file system API to save
      logger.info(`Would save colored stack image to: ${coloredStackPath}`);
      // ImageUtils.saveImage(coloredResult, 'jpg'); // When file saving is implemented

      coloredResult.delete();
    } else {
      logger.info(
        `Note: Nothing to save for colored image. Stack level: ${this.saveImageLevel}`
      );
    }
  }

  /**
   * Get result horizontal stack from titles and images.
   *
   * Port of Python's get_result_hstack method.
   *
   * @param titlesAndImages - Array of [title, image] tuples
   * @param imagesPerRow - Number of images per row
   * @returns Stacked image result
   */
  getResultHstack(
    titlesAndImages: TitleAndImage[],
    imagesPerRow: number
  ): cv.Mat {
    const config = this.tuningConfig;
    const displayDimensions =
      config.outputs?.display_image_dimensions || [600, 800];
    const displayWidth = displayDimensions[1];

    // TODO: attach title text as header to each stack image!
    const images = titlesAndImages.map(([_title, image]) => image);
    const resizedImages = ImageUtils.resizeMultiple(images, displayWidth) as cv.Mat[];

    // Create grid: chunk images into rows using MathUtils.chunks
    const gridImages: cv.Mat[][] = [];
    for (const chunk of MathUtils.chunks(resizedImages, imagesPerRow)) {
      gridImages.push(chunk);
    }

    // Create horizontal stacks for each row (get_padded_hstack for each)
    const hstacks: cv.Mat[] = [];
    for (const row of gridImages) {
      const hstack = ImageUtils.getPaddedHstack(row);
      hstacks.push(hstack);
    }

    // Create vertical stack from horizontal stacks (get_padded_vstack)
    const result = ImageUtils.getPaddedVstack(hstacks);

    // Resize final result
    const finalWidth = Math.min(
      titlesAndImages.length * displayWidth / 3,
      Math.floor(displayWidth * 2.5)
    );
    const finalResult = ImageUtils.resizeSingle(result, finalWidth);

    // Cleanup intermediate images
    resizedImages.forEach((img, idx) => {
      if (img !== images[idx]) img.delete();
    });
    hstacks.forEach((h) => h.delete());
    result.delete();

    return finalResult!;
  }

  /**
   * Reset all saved images.
   *
   * Port of Python's reset_all_save_img method.
   */
  resetAllSaveImg(): void {
    // Max save image level is 6 (but we store 1-7)
    for (let i = 1; i <= 7; i++) {
      // Clean up existing images
      const grayImgs = this.grayImages.get(i) || [];
      const coloredImgs = this.coloredImages.get(i) || [];

      grayImgs.forEach(([_title, img]) => img.delete());
      coloredImgs.forEach(([_title, img]) => img.delete());

      this.grayImages.set(i, []);
      this.coloredImages.set(i, []);
    }
  }

  /**
   * Get tuning config.
   */
  getTuningConfig(): any {
    return this.tuningConfig;
  }
}

