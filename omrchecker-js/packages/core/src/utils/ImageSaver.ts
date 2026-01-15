/**
 * Image saving utilities using File System Access API
 * Browser-compatible version of Python's image saving functionality
 */

import * as cv from '@techstark/opencv-js';
import { Logger } from './logger';

const logger = new Logger('ImageSaver');

/**
 * Options for saving images
 */
export interface SaveImageOptions {
  /** Output directory (for batch saves) */
  directory?: string;
  /** File name (without extension) */
  fileName?: string;
  /** Image format (png, jpeg, webp) */
  format?: 'png' | 'jpeg' | 'webp';
  /** Quality for lossy formats (0-1) */
  quality?: number;
  /** Whether to use File System Access API (requires user permission) */
  useFileSystemAPI?: boolean;
  /** Whether to auto-download (alternative to File System API) */
  autoDownload?: boolean;
}

/**
 * Image saver class for managing debug/intermediate image saves
 */
export class ImageSaver {
  private savedImages: Array<{ name: string; data: string; timestamp: Date }> = [];
  private directoryHandle: FileSystemDirectoryHandle | null = null;
  private imageCounter = 0;

  /**
   * Request directory access using File System Access API
   * User will be prompted to select a directory
   */
  async requestDirectoryAccess(): Promise<boolean> {
    try {
      if (!('showDirectoryPicker' in window)) {
        logger.warn('File System Access API not supported in this browser');
        return false;
      }

      // @ts-ignore - File System Access API
      this.directoryHandle = await window.showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'downloads',
      });

      logger.info(`Directory access granted: ${this.directoryHandle?.name || 'unknown'}`);
      return true;
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        logger.info('Directory selection cancelled by user');
      } else {
        logger.error('Failed to request directory access:', error instanceof Error ? error.message : String(error));
      }
      return false;
    }
  }

  /**
   * Save an OpenCV Mat as an image file
   */
  async saveImage(
    image: cv.Mat,
    fileName: string,
    options: SaveImageOptions = {}
  ): Promise<boolean> {
    try {
      const format = options.format || 'png';
      const quality = options.quality || 0.95;
      const fullFileName = `${fileName}.${format}`;

      // Convert Mat to canvas
      const canvas = document.createElement('canvas');
      canvas.width = image.cols;
      canvas.height = image.rows;

      // Handle different image types
      let displayImage = image;
      let needsCleanup = false;

      if (image.channels() === 1) {
        displayImage = new cv.Mat();
        cv.cvtColor(image, displayImage, cv.COLOR_GRAY2RGBA);
        needsCleanup = true;
      } else if (image.channels() === 3) {
        displayImage = new cv.Mat();
        cv.cvtColor(image, displayImage, cv.COLOR_RGB2RGBA);
        needsCleanup = true;
      }

      cv.imshow(canvas, displayImage);

      if (needsCleanup) {
        displayImage.delete();
      }

      // Convert canvas to blob
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Failed to create blob from canvas'));
            }
          },
          `image/${format}`,
          quality
        );
      });

      // Save using File System Access API if available and requested
      if (options.useFileSystemAPI && this.directoryHandle) {
        return await this.saveWithFileSystemAPI(fullFileName, blob);
      }

      // Fallback: Auto-download or store in memory
      if (options.autoDownload) {
        this.autoDownload(fullFileName, blob);
        return true;
      }

      // Store in memory for later download
      const dataUrl = await this.blobToDataUrl(blob);
      this.savedImages.push({
        name: fullFileName,
        data: dataUrl,
        timestamp: new Date(),
      });

      logger.debug(`Image stored in memory: ${fullFileName}`);
      return true;
    } catch (error) {
      logger.error(`Failed to save image ${fileName}:`, error instanceof Error ? error.message : String(error));
      return false;
    }
  }

  /**
   * Save blob using File System Access API
   */
  private async saveWithFileSystemAPI(
    fileName: string,
    blob: Blob
  ): Promise<boolean> {
    try {
      if (!this.directoryHandle) {
        logger.warn('No directory handle available');
        return false;
      }

      // Create file in selected directory
      const fileHandle = await this.directoryHandle.getFileHandle(fileName, {
        create: true,
      });

      // Write blob to file
      const writable = await fileHandle.createWritable();
      await writable.write(blob);
      await writable.close();

      logger.info(`Saved image to disk: ${fileName}`);
      return true;
    } catch (error) {
      logger.error(`Failed to save with File System API:`, error instanceof Error ? error.message : String(error));
      return false;
    }
  }

  /**
   * Auto-download image (trigger browser download)
   */
  private autoDownload(fileName: string, blob: Blob): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    logger.debug(`Auto-downloaded: ${fileName}`);
  }

  /**
   * Convert blob to data URL
   */
  private async blobToDataUrl(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Append image to save queue (for batch processing)
   * Compatible with Python's appendSaveImage
   */
  async appendSaveImage(
    key: string,
    image: cv.Mat,
    options: SaveImageOptions = {}
  ): Promise<void> {
    const fileName = options.fileName || `${key}_${this.imageCounter++}`;
    await this.saveImage(image, fileName, options);
  }

  /**
   * Download all stored images as a ZIP file
   */
  async downloadAllAsZip(_zipFileName: string = 'omr-debug-images.zip'): Promise<void> {
    try {
      // This would require a ZIP library like JSZip
      // For now, download individually
      logger.warn('ZIP download not implemented, downloading individually');
      for (const img of this.savedImages) {
        const a = document.createElement('a');
        a.href = img.data;
        a.download = img.name;
        a.click();
        await new Promise(resolve => setTimeout(resolve, 100)); // Small delay between downloads
      }
      logger.info(`Downloaded ${this.savedImages.length} images`);
    } catch (error) {
      logger.error('Failed to download images:', error instanceof Error ? error.message : String(error));
    }
  }

  /**
   * Clear all stored images
   */
  clearStoredImages(): void {
    this.savedImages = [];
    this.imageCounter = 0;
    logger.debug('Cleared stored images');
  }

  /**
   * Get list of stored images
   */
  getStoredImages(): Array<{ name: string; timestamp: Date }> {
    return this.savedImages.map(img => ({
      name: img.name,
      timestamp: img.timestamp,
    }));
  }

  /**
   * Get stored image count
   */
  getStoredImageCount(): number {
    return this.savedImages.length;
  }
}

// Singleton instance
const imageSaver = new ImageSaver();

/**
 * Save an image to disk or memory
 *
 * @example
 * ```typescript
 * // Auto-download
 * await saveImage(image, 'processed', { autoDownload: true });
 *
 * // Use File System API (requires user permission)
 * await requestDirectoryAccess();
 * await saveImage(image, 'debug1', { useFileSystemAPI: true });
 *
 * // Store in memory for batch download
 * await saveImage(image, 'step1');
 * await saveImage(image, 'step2');
 * await downloadAllStoredImages();
 * ```
 */
export async function saveImage(
  image: cv.Mat,
  fileName: string,
  options: SaveImageOptions = {}
): Promise<boolean> {
  return imageSaver.saveImage(image, fileName, options);
}

/**
 * Append image to save queue (compatible with Python's appendSaveImage)
 */
export async function appendSaveImage(
  key: string,
  image: cv.Mat,
  options: SaveImageOptions = {}
): Promise<void> {
  return imageSaver.appendSaveImage(key, image, options);
}

/**
 * Request directory access for File System API
 */
export async function requestDirectoryAccess(): Promise<boolean> {
  return imageSaver.requestDirectoryAccess();
}

/**
 * Download all stored images
 */
export async function downloadAllStoredImages(): Promise<void> {
  return imageSaver.downloadAllAsZip();
}

/**
 * Clear all stored images
 */
export function clearStoredImages(): void {
  imageSaver.clearStoredImages();
}

/**
 * Get list of stored images
 */
export function getStoredImages(): Array<{ name: string; timestamp: Date }> {
  return imageSaver.getStoredImages();
}

/**
 * Get stored image count
 */
export function getStoredImageCount(): number {
  return imageSaver.getStoredImageCount();
}

/**
 * ImageSaver utilities namespace
 */
export const ImageSaverUtils = {
  saveImage,
  appendSaveImage,
  requestDirectoryAccess,
  downloadAllStoredImages,
  clearStoredImages,
  getStoredImages,
  getStoredImageCount,
  ImageSaver,
};

export default ImageSaverUtils;

