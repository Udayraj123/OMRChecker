/**
 * Interaction utilities for displaying images and getting user input
 * Browser-compatible version of Python's InteractionUtils
 */

import cv from './opencv';
import { Logger } from './logger';

const logger = new Logger('InteractionUtils');

/**
 * Options for displaying images
 */
export interface ShowImageOptions {
  /** Window/container title */
  title?: string;
  /** Width of the display (auto if not specified) */
  width?: number;
  /** Height of the display (auto if not specified) */
  height?: number;
  /** Wait for user input (blocking) - not supported in browser, logs warning */
  waitKey?: boolean;
  /** Container element ID where image should be displayed */
  containerId?: string;
  /** Whether to resize to fit container */
  resizeToFit?: boolean;
}

/**
 * Container for debug images shown in the UI
 */
class DebugImageContainer {
  private container: HTMLElement | null = null;
  private imageCounter = 0;

  /**
   * Get or create the debug container
   */
  private getContainer(): HTMLElement {
    if (this.container && document.body.contains(this.container)) {
      return this.container;
    }

    // Create debug container if it doesn't exist
    this.container = document.getElementById('omr-debug-container');

    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = 'omr-debug-container';
      this.container.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        max-width: 400px;
        max-height: 80vh;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.9);
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 15px;
        z-index: 10000;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
      `;

      // Add header
      const header = document.createElement('div');
      header.style.cssText = `
        color: #4CAF50;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
      `;
      header.innerHTML = `
        <span>🔍 Debug Images</span>
        <button id="omr-debug-close" style="
          background: #f44336;
          color: white;
          border: none;
          border-radius: 4px;
          padding: 4px 8px;
          cursor: pointer;
          font-size: 12px;
        ">Close</button>
      `;
      this.container.appendChild(header);

      // Add images container
      const imagesContainer = document.createElement('div');
      imagesContainer.id = 'omr-debug-images';
      this.container.appendChild(imagesContainer);

      document.body.appendChild(this.container);

      // Add close button handler
      document.getElementById('omr-debug-close')?.addEventListener('click', () => {
        this.hide();
      });

      logger.debug('Created debug image container');
    }

    return this.container;
  }

  /**
   * Add an image to the debug container
   */
  addImage(canvas: HTMLCanvasElement, title: string): void {
    const container = this.getContainer();
    const imagesContainer = container.querySelector('#omr-debug-images');

    if (!imagesContainer) return;

    const imageWrapper = document.createElement('div');
    imageWrapper.style.cssText = `
      margin-bottom: 15px;
      padding: 10px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 4px;
    `;

    // Add title
    const titleElement = document.createElement('div');
    titleElement.textContent = title;
    titleElement.style.cssText = `
      color: #fff;
      font-size: 12px;
      margin-bottom: 8px;
      font-family: monospace;
    `;
    imageWrapper.appendChild(titleElement);

    // Add canvas
    canvas.style.cssText = `
      width: 100%;
      height: auto;
      border: 1px solid #666;
      border-radius: 4px;
      display: block;
    `;
    imageWrapper.appendChild(canvas);

    imagesContainer.appendChild(imageWrapper);

    // Auto-scroll to latest image
    container.scrollTop = container.scrollHeight;
  }

  /**
   * Clear all debug images
   */
  clear(): void {
    const imagesContainer = document.querySelector('#omr-debug-images');
    if (imagesContainer) {
      imagesContainer.innerHTML = '';
    }
    this.imageCounter = 0;
    logger.debug('Cleared debug images');
  }

  /**
   * Hide the debug container
   */
  hide(): void {
    if (this.container) {
      this.container.style.display = 'none';
    }
  }

  /**
   * Show the debug container
   */
  show(): void {
    const container = this.getContainer();
    container.style.display = 'block';
  }

  /**
   * Get next image number for auto-naming
   */
  getNextImageNumber(): number {
    return ++this.imageCounter;
  }
}

// Singleton instance
const debugContainer = new DebugImageContainer();

/**
 * Show an image in the debug UI (non-blocking)
 * Browser-compatible version of cv2.imshow()
 *
 * @param windowName - Title for the image window
 * @param image - OpenCV Mat to display
 * @param options - Display options
 *
 * @example
 * ```typescript
 * const image = cv.imread(canvas);
 * InteractionUtils.show('Original', image);
 * InteractionUtils.show('Processed', processedImage, { width: 300 });
 * ```
 */
export function show(
  windowName: string,
  image: cv.Mat,
  options: ShowImageOptions = {}
): void {
  try {
    // Create a canvas for this image
    const canvas = document.createElement('canvas');
    canvas.width = options.width || image.cols;
    canvas.height = options.height || image.rows;

    // Handle different image types
    let displayImage = image;
    let needsCleanup = false;

    // Convert grayscale to RGBA for display
    if (image.channels() === 1) {
      displayImage = new cv.Mat();
      cv.cvtColor(image, displayImage, cv.COLOR_GRAY2RGBA);
      needsCleanup = true;
    } else if (image.channels() === 3) {
      displayImage = new cv.Mat();
      cv.cvtColor(image, displayImage, cv.COLOR_RGB2RGBA);
      needsCleanup = true;
    }

    // Resize if requested
    if (options.resizeToFit && (options.width || options.height)) {
      const resized = new cv.Mat();
      const size = new cv.Size(
        options.width || image.cols,
        options.height || image.rows
      );
      cv.resize(displayImage, resized, size, 0, 0, cv.INTER_LINEAR);

      if (needsCleanup) {
        displayImage.delete();
      }
      displayImage = resized;
      needsCleanup = true;
    }

    // Render to canvas
    cv.imshow(canvas, displayImage);

    // Clean up temporary mat
    if (needsCleanup) {
      displayImage.delete();
    }

    // Add to debug container
    const title = options.title || windowName;
    debugContainer.addImage(canvas, title);

    // If specific container ID provided, also add there
    if (options.containerId) {
      const targetContainer = document.getElementById(options.containerId);
      if (targetContainer) {
        targetContainer.innerHTML = '';
        targetContainer.appendChild(canvas.cloneNode(true));
      }
    }

    // Log warning if waitKey requested (not supported in browser)
    if (options.waitKey) {
      logger.warn('waitKey is not supported in browser environment (non-blocking display)');
    }

    logger.debug(`Displayed image: ${windowName} (${image.cols}x${image.rows})`);
  } catch (error) {
    logger.error(`Failed to show image ${windowName}:`, error instanceof Error ? error.message : String(error));
  }
}

/**
 * Clear all debug images from the UI
 */
export function clearDebugImages(): void {
  debugContainer.clear();
}

/**
 * Hide the debug container
 */
export function hideDebugContainer(): void {
  debugContainer.hide();
}

/**
 * Show the debug container
 */
export function showDebugContainer(): void {
  debugContainer.show();
}

/**
 * Wait for a key press (browser-compatible, uses Promise)
 * Note: In browser, this just waits for specified time or returns immediately
 *
 * @param delay - Delay in milliseconds (0 = no wait in browser)
 * @returns Promise that resolves after delay
 */
export async function waitKey(delay: number = 0): Promise<void> {
  if (delay > 0) {
    return new Promise(resolve => setTimeout(resolve, delay));
  }
  return Promise.resolve();
}

/**
 * Destroy a window (browser-compatible, clears specific image)
 *
 * @param windowName - Name of window to destroy
 */
export function destroyWindow(windowName: string): void {
  logger.debug(`destroyWindow called for: ${windowName} (no-op in browser)`);
  // In browser, we don't have individual windows, so this is a no-op
  // Could implement removing specific image from debug container if needed
}

/**
 * Destroy all windows (browser-compatible, clears all debug images)
 */
export function destroyAllWindows(): void {
  debugContainer.clear();
  logger.debug('Destroyed all windows (cleared debug images)');
}

/**
 * Get the next auto-generated image number for naming
 */
export function getNextImageNumber(): number {
  return debugContainer.getNextImageNumber();
}

/**
 * InteractionUtils namespace (for compatibility with Python structure)
 */
export const InteractionUtils = {
  show,
  clearDebugImages,
  hideDebugContainer,
  showDebugContainer,
  waitKey,
  destroyWindow,
  destroyAllWindows,
  getNextImageNumber,
};

export default InteractionUtils;

