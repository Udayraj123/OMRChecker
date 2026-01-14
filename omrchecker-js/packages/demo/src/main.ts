/**
 * OMRChecker Demo - Main Application
 *
 * Browser-based OMR bubble detection using the TypeScript port.
 * Supports single/multiple files and folder upload with File System Access API.
 */

import {
  TemplateLoader,
  SimpleBubbleDetector,
  type TemplateConfig,
  type ParsedTemplate,
  type FieldDetectionResult,
} from '@omrchecker/core';

// Create global binding for OpenCV before it loads
declare global {
  interface Window {
    cv: any;
    Module: any;
    showDirectoryPicker(): Promise<FileSystemDirectoryHandle>;
  }
}

// OpenCV will be loaded from CDN and attach to window.cv
const cv: any = (window as any).cv || {};

// OpenCV type declarations
type OpenCVType = {
  Mat: any;
  imread: (element: HTMLImageElement) => any;
  cvtColor: (src: any, dst: any, code: number) => void;
  imshow: (canvas: HTMLCanvasElement, mat: any) => void;
  circle: (img: any, center: any, radius: number, color: any, thickness: number) => void;
  putText: (img: any, text: string, org: any, fontFace: number, fontScale: number, color: any, thickness: number) => void;
  Point: new (x: number, y: number) => any;
  Scalar: new (...values: number[]) => any;
  COLOR_RGBA2GRAY: number;
  COLOR_GRAY2RGBA: number;
  FONT_HERSHEY_SIMPLEX: number;
  getBuildInformation?: () => string;
  onRuntimeInitialized?: () => void;
};

// File System Access API types
interface FileSystemHandle {
  kind: 'file' | 'directory';
  name: string;
}

interface FileSystemFileHandle extends FileSystemHandle {
  kind: 'file';
  getFile(): Promise<File>;
}

interface FileSystemDirectoryHandle extends FileSystemHandle {
  kind: 'directory';
  values(): AsyncIterableIterator<FileSystemFileHandle | FileSystemDirectoryHandle>;
  getFileHandle(name: string): Promise<FileSystemFileHandle>;
  getDirectoryHandle(name: string): Promise<FileSystemDirectoryHandle>;
}

// Global state
let templateData: ParsedTemplate | null = null;
let imageFiles: File[] = [];
let allResults: Array<{ filename: string; results: Map<string, FieldDetectionResult> }> = [];

// DOM elements
const templateUpload = document.getElementById('template-upload') as HTMLInputElement;
const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
const folderUploadBtn = document.getElementById('folder-upload-btn') as HTMLButtonElement;
const detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;
const exportCsvBtn = document.getElementById('export-csv-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
const inputCanvas = document.getElementById('input-canvas') as HTMLCanvasElement;
const outputCanvas = document.getElementById('output-canvas') as HTMLCanvasElement;
const resultsSection = document.getElementById('results-section') as HTMLElement;
const loadingOverlay = document.getElementById('loading-overlay') as HTMLElement;
const loadingMessage = document.getElementById('loading-message') as HTMLElement;

/**
 * Wait for OpenCV.js to be ready
 */
function waitForOpenCV(): Promise<void> {
  return new Promise((resolve) => {
    // Check if already loaded
    if (window.cv && typeof window.cv.getBuildInformation === 'function') {
      resolve();
      return;
    }

    // Poll for OpenCV to be ready
    const checkInterval = setInterval(() => {
      if (window.cv && typeof window.cv.getBuildInformation === 'function') {
        clearInterval(checkInterval);
        resolve();
      }
    }, 100);

    // Set up timeout to prevent infinite waiting
    setTimeout(() => {
      clearInterval(checkInterval);
      if (!window.cv || !window.cv.getBuildInformation) {
        console.error('OpenCV failed to load within timeout');
      }
    }, 30000); // 30 second timeout
  });
}

/**
 * Initialize the application
 */
async function init(): Promise<void> {
  console.log('Initializing OMRChecker Demo...');

  updateStatus('⏳ Waiting for OpenCV...', 'info');

  // Wait for OpenCV to load
  await waitForOpenCV();

  console.log('OpenCV.js is ready!');
  updateStatus('✅ OpenCV loaded successfully', 'success');

  // Setup event listeners
  setupEventListeners();

  console.log('Demo initialized!');
}

/**
 * Setup all event listeners
 */
function setupEventListeners(): void {
  templateUpload.addEventListener('change', handleTemplateUpload);
  imageUpload.addEventListener('change', handleImageUpload);
  folderUploadBtn.addEventListener('click', handleFolderUpload);
  detectBtn.addEventListener('click', handleDetectBatch);
  exportCsvBtn.addEventListener('click', handleExportCSV);
  resetBtn.addEventListener('click', handleReset);
}

/**
 * Handle template file upload
 */
async function handleTemplateUpload(event: Event): Promise<void> {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;

  try {
    showLoading('Loading template...');

    const text = await file.text();
    const templateJson = JSON.parse(text) as TemplateConfig;

    templateData = TemplateLoader.loadFromJSON(templateJson);

    updateFileInfo('template-info', file.name, `${templateData.fields.size} fields detected`);
    updateStatus(`✅ Template loaded: ${templateData.fields.size} fields`, 'success');

    checkReadyToDetect();
    hideLoading();
  } catch (error) {
    console.error('Template load error:', error);
    updateStatus(`❌ Failed to load template: ${(error as Error).message}`, 'error');
    hideLoading();
  }
}

/**
 * Handle image file upload (single or multiple files)
 */
async function handleImageUpload(event: Event): Promise<void> {
  const files = Array.from((event.target as HTMLInputElement).files || []);
  if (files.length === 0) return;

  try {
    imageFiles = files.filter((f) => f.type.startsWith('image/'));

    if (imageFiles.length === 0) {
      updateStatus('❌ No image files selected', 'error');
      return;
    }

    // Show first image preview
    await displayImageFile(imageFiles[0]);

    updateFileInfo(
      'image-info',
      imageFiles.length === 1 ? imageFiles[0].name : `${imageFiles.length} images selected`,
      imageFiles.length === 1 ? '' : 'Batch mode enabled'
    );
    updateStatus(`✅ Loaded ${imageFiles.length} image(s)`, 'success');

    checkReadyToDetect();
  } catch (error) {
    console.error('Image load error:', error);
    updateStatus(`❌ Failed to load images: ${(error as Error).message}`, 'error');
  }
}

/**
 * Handle folder upload using File System Access API
 */
async function handleFolderUpload(): Promise<void> {
  // Check if File System Access API is supported
  if (!('showDirectoryPicker' in window)) {
    alert(
      'Folder upload is not supported in this browser. Please use Chrome, Edge, or another Chromium-based browser.'
    );
    return;
  }

  try {
    showLoading('Scanning folder...');

    // Request directory access
    const dirHandle = await window.showDirectoryPicker();

    // Try to auto-load template.json from the selected folder
    await tryAutoLoadTemplate(dirHandle);

    // Recursively collect image files
    imageFiles = await collectImageFilesFromDirectory(dirHandle);

    if (imageFiles.length === 0) {
      updateStatus('❌ No image files found in folder', 'error');
      hideLoading();
      return;
    }

    // Show first image preview
    await displayImageFile(imageFiles[0]);

    updateFileInfo(
      'image-info',
      `${imageFiles.length} images from folder`,
      `Root: ${dirHandle.name}`
    );
    updateStatus(`✅ Loaded ${imageFiles.length} image(s) from folder`, 'success');

    checkReadyToDetect();
    hideLoading();
  } catch (error) {
    if ((error as Error).name === 'AbortError') {
      updateStatus('Folder selection cancelled', 'info');
    } else {
      console.error('Folder upload error:', error);
      updateStatus(`❌ Failed to load folder: ${(error as Error).message}`, 'error');
    }
    hideLoading();
  }
}

/**
 * Try to automatically find and load template.json from the directory
 */
async function tryAutoLoadTemplate(dirHandle: FileSystemDirectoryHandle): Promise<void> {
  try {
    updateLoadingMessage('Searching for template.json...');

    // Try to find template.json in the selected directory
    const templateFile = await findTemplateInDirectory(dirHandle);

    if (templateFile) {
      const text = await templateFile.text();
      const templateJson = JSON.parse(text) as TemplateConfig;

      templateData = TemplateLoader.loadFromJSON(templateJson);

      updateFileInfo(
        'template-info',
        '📄 template.json (auto-detected)',
        `${templateData.fields.size} fields | Found in: ${dirHandle.name}`
      );
      updateStatus(`✅ Auto-loaded template: ${templateData.fields.size} fields`, 'success');

      console.log('Auto-loaded template from folder');
    } else {
      console.log('No template.json found in folder');

      // If no template loaded yet, show a helpful message
      if (!templateData) {
        updateFileInfo(
          'template-info',
          '⚠️ No template.json found',
          'Please upload template.json manually'
        );
      }
    }
  } catch (error) {
    console.error('Error auto-loading template:', error);
    // Don't fail the whole operation, just log it
    if (!templateData) {
      updateFileInfo(
        'template-info',
        '⚠️ Failed to auto-load template',
        'Please upload template.json manually'
      );
    }
  }
}

/**
 * Find template.json in the directory
 * Note: File System Access API doesn't allow accessing parent directories
 * for security reasons, so we can only search the selected folder.
 */
async function findTemplateInDirectory(dirHandle: FileSystemDirectoryHandle): Promise<File | null> {
  try {
    // Try to get template.json file directly
    const templateHandle = await dirHandle.getFileHandle('template.json');
    return await templateHandle.getFile();
  } catch {
    // File not found in this directory
    // Note: We cannot access parent directories due to browser security restrictions
    return null;
  }
}

/**
 * Recursively collect image files from directory
 */
async function collectImageFilesFromDirectory(
  dirHandle: FileSystemDirectoryHandle,
  path: string = ''
): Promise<File[]> {
  const files: File[] = [];
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'];

  for await (const entry of dirHandle.values()) {
    const entryPath = path ? `${path}/${entry.name}` : entry.name;

    if (entry.kind === 'file') {
      const ext = entry.name.toLowerCase().substring(entry.name.lastIndexOf('.'));
      if (imageExtensions.includes(ext)) {
        const file = await (entry as FileSystemFileHandle).getFile();
        // Store path in file object for reference
        Object.defineProperty(file, 'relativePath', { value: entryPath, writable: true });
        files.push(file);
      }
    } else if (entry.kind === 'directory') {
      // Recursively scan subdirectories
      const subFiles = await collectImageFilesFromDirectory(entry as FileSystemDirectoryHandle, entryPath);
      files.push(...subFiles);
    }
  }

  return files;
}

/**
 * Display image file on canvas
 */
async function displayImageFile(file: File): Promise<void> {
  try {
    showLoading('Loading image...');

    const imageData = await loadImage(file);

    // Display on canvas
    displayImage(inputCanvas, imageData);
    hideCanvasPlaceholder('input-placeholder');

    imageData.delete(); // Clean up
    hideLoading();
  } catch (error) {
    console.error('Display error:', error);
    hideLoading();
    throw error;
  }
}

/**
 * Load image from file
 */
function loadImage(file: File): Promise<any> {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      try {
        const mat = window.cv.imread(img);
        if (mat.empty()) {
          throw new Error('Failed to load image');
        }

        // Convert to grayscale
        let grayMat: any;
        if (mat.channels() === 3 || mat.channels() === 4) {
          grayMat = new window.cv.Mat();
          window.cv.cvtColor(mat, grayMat, window.cv.COLOR_RGBA2GRAY);
          mat.delete();
        } else {
          grayMat = mat;
        }

        resolve(grayMat);
      } catch (error) {
        reject(error);
      }
    };

    img.onerror = () => reject(new Error('Failed to load image'));

    const reader = new FileReader();
    reader.onload = (e) => {
      img.src = e.target?.result as string;
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

/**
 * Display image on canvas
 */
function displayImage(canvas: HTMLCanvasElement, mat: any): void {
  // Convert grayscale to RGBA for display
  const displayMat = new window.cv.Mat();
  window.cv.cvtColor(mat, displayMat, window.cv.COLOR_GRAY2RGBA);

  canvas.width = displayMat.cols;
  canvas.height = displayMat.rows;
  window.cv.imshow(canvas, displayMat);

  displayMat.delete();
}

/**
 * Handle batch detection for all loaded images
 */
async function handleDetectBatch(): Promise<void> {
  if (!templateData || imageFiles.length === 0) return;

  try {
    showLoading(`Processing 0/${imageFiles.length} images...`);

    allResults = [];
    const detector = new SimpleBubbleDetector();

    // Process each image
    for (let i = 0; i < imageFiles.length; i++) {
      const file = imageFiles[i];
      const filename = ((file as File & { relativePath?: string }).relativePath) || file.name;

      updateLoadingMessage(`Processing ${i + 1}/${imageFiles.length}: ${filename}`);

      // Load image
      const imageData = await loadImage(file);

      // Detect
      const results = detector.detectMultipleFields(imageData, templateData.fieldBubbles);

      // Store results
      allResults.push({ filename, results });

      // If it's the current/first image, visualize it
      if (i === 0) {
        visualizeResults(imageData, results, templateData);
      }

      // Clean up
      imageData.delete();
    }

    // Display aggregate results
    displayBatchResults(allResults, templateData);

    // Enable export button
    exportCsvBtn.disabled = false;

    updateStatus(
      `✅ Batch complete: ${imageFiles.length} images processed`,
      'success'
    );
    hideLoading();
  } catch (error) {
    console.error('Detection error:', error);
    updateStatus(`❌ Detection failed: ${(error as Error).message}`, 'error');
    hideLoading();
  }
}

/**
 * Visualize detection results on canvas
 */
function visualizeResults(
  image: any,
  results: Map<string, FieldDetectionResult>,
  template: ParsedTemplate
): void {
  // Create colored output image
  const outputMat = new window.cv.Mat();
  window.cv.cvtColor(image, outputMat, window.cv.COLOR_GRAY2RGBA);

  // Draw bubbles
  for (const [fieldId, result] of results.entries()) {
    const field = template.fields.get(fieldId);
    if (!field) continue;

    for (let i = 0; i < field.bubbles.length; i++) {
      const bubble = field.bubbles[i];
      const bubbleResult = result.bubbles[i];

      const center = new window.cv.Point(
        Math.floor(bubble.x + bubble.width / 2),
        Math.floor(bubble.y + bubble.height / 2)
      );
      const radius = Math.floor(Math.min(bubble.width, bubble.height) / 2);

      // Color based on detection
      let color: any;
      if (bubbleResult.isMarked) {
        color = new window.cv.Scalar(0, 255, 0, 255); // Green for marked
      } else {
        color = new window.cv.Scalar(100, 100, 100, 200); // Gray for unmarked
      }

      // Draw circle
      window.cv.circle(outputMat, center, radius, color, 2);

      // Draw label
      const textPos = new window.cv.Point(
        Math.floor(bubble.x + bubble.width / 2 - 5),
        Math.floor(bubble.y + bubble.height / 2 + 5)
      );
      window.cv.putText(
        outputMat,
        bubble.label,
        textPos,
        window.cv.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1
      );
    }
  }

  // Display on output canvas
  outputCanvas.width = outputMat.cols;
  outputCanvas.height = outputMat.rows;
  window.cv.imshow(outputCanvas, outputMat);

  hideCanvasPlaceholder('output-placeholder');
  outputMat.delete();
}

/**
 * Display detection results
 */
function displayResults(results: Map<string, FieldDetectionResult>, template: ParsedTemplate): void {
  // Calculate statistics
  const detector = new SimpleBubbleDetector();
  const stats = detector.getDetectionStats(results);

  // Update stat cards
  document.getElementById('stat-total')!.textContent = stats.totalFields.toString();
  document.getElementById('stat-answered')!.textContent = stats.answeredFields.toString();
  document.getElementById('stat-unanswered')!.textContent = stats.unansweredFields.toString();
  document.getElementById('stat-multimarked')!.textContent = stats.multiMarkedFields.toString();
  document.getElementById('stat-confidence')!.textContent = `${(stats.avgConfidence * 100).toFixed(1)}%`;

  // Populate table
  const tbody = document.getElementById('results-tbody') as HTMLTableSectionElement;
  tbody.innerHTML = '';

  const sortedFieldIds = TemplateLoader.getSortedFieldIds(template);

  for (const fieldId of sortedFieldIds) {
    const result = results.get(fieldId);
    if (!result) continue;

    const row = tbody.insertRow();

    // Determine status class
    if (result.isMultiMarked) {
      row.classList.add('multi-marked');
    } else if (result.detectedAnswer) {
      row.classList.add('answered');
    } else {
      row.classList.add('unanswered');
    }

    // Question
    row.insertCell(0).textContent = fieldId;

    // Answer
    const field = template.fields.get(fieldId);
    const answer = result.detectedAnswer || field?.emptyValue || '-';
    row.insertCell(1).textContent = answer;

    // Confidence
    const markedBubble = result.bubbles.find((b) => b.isMarked);
    const confidence = markedBubble ? (markedBubble.confidence * 100).toFixed(1) : '0.0';
    row.insertCell(2).textContent = `${confidence}%`;

    // Threshold
    row.insertCell(3).textContent = result.threshold.toFixed(1);

    // Status
    const statusCell = row.insertCell(4);
    let statusHtml: string;
    if (result.isMultiMarked) {
      statusHtml = '<span class="status-badge warning">⚠️ Multi-marked</span>';
    } else if (result.detectedAnswer) {
      statusHtml = '<span class="status-badge success">✅ Detected</span>';
    } else {
      statusHtml = '<span class="status-badge error">❌ Empty</span>';
    }
    statusCell.innerHTML = statusHtml;
  }

  // Show results section
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Display batch results (aggregate from multiple images)
 */
function displayBatchResults(
  batchResults: Array<{ filename: string; results: Map<string, FieldDetectionResult> }>,
  template: ParsedTemplate
): void {
  if (batchResults.length === 0) return;

  // If only one image, use regular display
  if (batchResults.length === 1) {
    displayResults(batchResults[0].results, template);
    return;
  }

  // Calculate aggregate statistics
  const detector = new SimpleBubbleDetector();
  let totalAnswered = 0;
  let totalUnanswered = 0;
  let totalMultiMarked = 0;
  let totalConfidence = 0;
  let totalFields = 0;

  for (const { results } of batchResults) {
    const stats = detector.getDetectionStats(results);
    totalAnswered += stats.answeredFields;
    totalUnanswered += stats.unansweredFields;
    totalMultiMarked += stats.multiMarkedFields;
    totalConfidence += stats.avgConfidence * stats.totalFields;
    totalFields += stats.totalFields;
  }

  const avgConfidence = totalFields > 0 ? totalConfidence / totalFields : 0;

  // Update stat cards with batch totals
  document.getElementById('stat-total')!.textContent = `${totalFields} (${batchResults.length} images)`;
  document.getElementById('stat-answered')!.textContent = totalAnswered.toString();
  document.getElementById('stat-unanswered')!.textContent = totalUnanswered.toString();
  document.getElementById('stat-multimarked')!.textContent = totalMultiMarked.toString();
  document.getElementById('stat-confidence')!.textContent = `${(avgConfidence * 100).toFixed(1)}%`;

  // Populate table with all results
  const tbody = document.getElementById('results-tbody') as HTMLTableSectionElement;
  tbody.innerHTML = '';

  for (const { filename, results } of batchResults) {
    // Add filename row
    const filenameRow = tbody.insertRow();
    filenameRow.style.backgroundColor = 'var(--bg-card)';
    filenameRow.style.fontWeight = 'bold';
    const filenameCell = filenameRow.insertCell(0);
    filenameCell.colSpan = 5;
    filenameCell.textContent = `📄 ${filename}`;

    // Add result rows for this image
    const sortedFieldIds = TemplateLoader.getSortedFieldIds(template);

    for (const fieldId of sortedFieldIds) {
      const result = results.get(fieldId);
      if (!result) continue;

      const row = tbody.insertRow();

      // Determine status class
      if (result.isMultiMarked) {
        row.classList.add('multi-marked');
      } else if (result.detectedAnswer) {
        row.classList.add('answered');
      } else {
        row.classList.add('unanswered');
      }

      // Question
      row.insertCell(0).textContent = `  ${fieldId}`;

      // Answer
      const field = template.fields.get(fieldId);
      const answer = result.detectedAnswer || field?.emptyValue || '-';
      row.insertCell(1).textContent = answer;

      // Confidence
      const markedBubble = result.bubbles.find((b) => b.isMarked);
      const confidence = markedBubble ? (markedBubble.confidence * 100).toFixed(1) : '0.0';
      row.insertCell(2).textContent = `${confidence}%`;

      // Threshold
      row.insertCell(3).textContent = result.threshold.toFixed(1);

      // Status
      const statusCell = row.insertCell(4);
      let statusHtml: string;
      if (result.isMultiMarked) {
        statusHtml = '<span class="status-badge warning">⚠️ Multi-marked</span>';
      } else if (result.detectedAnswer) {
        statusHtml = '<span class="status-badge success">✅ Detected</span>';
      } else {
        statusHtml = '<span class="status-badge error">❌ Empty</span>';
      }
      statusCell.innerHTML = statusHtml;
    }
  }

  // Show results section
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Export results to CSV
 */
function handleExportCSV(): void {
  if (allResults.length === 0 || !templateData) return;

  try {
    const csv = generateBatchCSV(allResults, templateData);
    downloadCSV(csv, 'omr-results-batch.csv');
    updateStatus('✅ CSV exported successfully', 'success');
  } catch (error) {
    console.error('Export error:', error);
    updateStatus(`❌ Export failed: ${(error as Error).message}`, 'error');
  }
}

/**
 * Generate CSV from batch results
 */
function generateBatchCSV(
  batchResults: Array<{ filename: string; results: Map<string, FieldDetectionResult> }>,
  template: ParsedTemplate
): string {
  const sortedFieldIds = TemplateLoader.getSortedFieldIds(template);

  // Header
  const header = 'Filename,' + sortedFieldIds.join(',') + ',Total,Answered,MultiMarked\n';

  // Rows (one per image)
  const rows = batchResults.map(({ filename, results }) => {
    const detector = new SimpleBubbleDetector();
    const stats = detector.getDetectionStats(results);

    const answers = sortedFieldIds.map((fieldId) => {
      const result = results.get(fieldId);
      const field = template.fields.get(fieldId);
      return result?.detectedAnswer || field?.emptyValue || '';
    });

    return `${filename},${answers.join(',')},${stats.totalFields},${stats.answeredFields},${stats.multiMarkedFields}`;
  });

  return header + rows.join('\n');
}

/**
 * Download CSV file
 */
function downloadCSV(csv: string, filename: string): void {
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Reset the application
 */
function handleReset(): void {
  // Clean up
  imageFiles = [];
  allResults = [];

  templateData = null;

  // Reset UI
  (templateUpload as HTMLInputElement).value = '';
  (imageUpload as HTMLInputElement).value = '';
  updateFileInfo('template-info', '', '');
  updateFileInfo('image-info', '', '');

  inputCanvas.width = 0;
  inputCanvas.height = 0;
  outputCanvas.width = 0;
  outputCanvas.height = 0;

  showCanvasPlaceholder('input-placeholder');
  showCanvasPlaceholder('output-placeholder');

  resultsSection.style.display = 'none';

  detectBtn.disabled = true;
  exportCsvBtn.disabled = true;

  updateStatus('🔄 Ready to start', 'info');
}

/**
 * Check if ready to detect
 */
function checkReadyToDetect(): void {
  detectBtn.disabled = !(templateData && imageFiles.length > 0);
}

/**
 * Update status bar
 */
function updateStatus(message: string, type: 'success' | 'error' | 'info'): void {
  const statusBar = document.getElementById('opencv-status')!;
  statusBar.textContent = message;
  statusBar.style.color =
    type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--error)' : 'var(--text-primary)';
}

/**
 * Update file info display
 */
function updateFileInfo(elementId: string, filename: string, details: string): void {
  const element = document.getElementById(elementId)!;
  if (filename) {
    element.innerHTML = `<strong>${filename}</strong><br/>${details}`;
  } else {
    element.innerHTML = '';
  }
}

/**
 * Show loading overlay
 */
function showLoading(message: string): void {
  loadingMessage.textContent = message;
  loadingOverlay.style.display = 'flex';
}

/**
 * Update loading message
 */
function updateLoadingMessage(message: string): void {
  loadingMessage.textContent = message;
}

/**
 * Hide loading overlay
 */
function hideLoading(): void {
  loadingOverlay.style.display = 'none';
}

/**
 * Hide canvas placeholder
 */
function hideCanvasPlaceholder(placeholderId: string): void {
  const placeholder = document.getElementById(placeholderId);
  if (placeholder) {
    placeholder.style.display = 'none';
  }
}

/**
 * Show canvas placeholder
 */
function showCanvasPlaceholder(placeholderId: string): void {
  const placeholder = document.getElementById(placeholderId);
  if (placeholder) {
    placeholder.style.display = 'block';
  }
}

// // Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  // DOM already loaded
  init();
}

