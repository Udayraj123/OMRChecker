/**
 * Complete OMR Detection Example
 *
 * Demonstrates how to use the ported TypeScript components together
 * for end-to-end OMR bubble detection.
 */

import {
  TemplateLoader,
  SimpleBubbleDetector,
  GaussianBlur,
  type TemplateConfig,
  type FieldDetectionResult,
} from '@omrchecker/core';
import * as cv from '@techstark/opencv-js';

/**
 * Complete OMR detection workflow.
 *
 * @param imageFile - Input image file
 * @param templateJson - Template configuration
 * @returns Detection results for all fields
 */
export async function detectOMRSheet(
  imageFile: File | Blob,
  templateJson: TemplateConfig
): Promise<Map<string, FieldDetectionResult>> {
  console.log('Starting OMR detection...');

  // Step 1: Load template
  console.log('1. Loading template...');
  const parsedTemplate = TemplateLoader.loadFromJSON(templateJson);
  console.log(`   Loaded ${parsedTemplate.fields.size} fields`);

  // Step 2: Load image
  console.log('2. Loading image...');
  const image = await loadImageFromFile(imageFile);
  console.log(`   Image size: ${image.cols}x${image.rows}`);

  // Step 3: Pre-process image (optional but recommended)
  console.log('3. Pre-processing image...');
  const processedImage = await preprocessImage(image, templateJson);

  // Step 4: Detect bubbles
  console.log('4. Detecting bubbles...');
  const detector = new SimpleBubbleDetector();
  const results = detector.detectMultipleFields(
    processedImage,
    parsedTemplate.fieldBubbles
  );

  // Step 5: Get statistics
  const stats = detector.getDetectionStats(results);
  console.log('5. Detection complete!');
  console.log(`   Answered: ${stats.answeredFields}/${stats.totalFields}`);
  console.log(`   Multi-marked: ${stats.multiMarkedFields}`);
  console.log(`   Avg confidence: ${(stats.avgConfidence * 100).toFixed(1)}%`);

  // Clean up
  image.delete();
  if (processedImage !== image) {
    processedImage.delete();
  }

  return results;
}

/**
 * Load image from File/Blob.
 */
async function loadImageFromFile(file: File | Blob): Promise<cv.Mat> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const mat = cv.imread(img);
        if (mat.empty()) {
          throw new Error('Failed to load image');
        }

        // Convert to grayscale if needed
        let grayMat: cv.Mat;
        if (mat.channels() === 3 || mat.channels() === 4) {
          grayMat = new cv.Mat();
          cv.cvtColor(mat, grayMat, cv.COLOR_RGBA2GRAY);
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
 * Apply pre-processors from template configuration.
 */
async function preprocessImage(
  image: cv.Mat,
  templateJson: TemplateConfig
): Promise<cv.Mat> {
  let processedImage = image.clone();

  const preProcessors = templateJson.preProcessors || [];

  for (const processorConfig of preProcessors) {
    switch (processorConfig.name) {
      case 'GaussianBlur': {
        const options = processorConfig.options || {};
        const kSize = (options.kSize as number[]) || [5, 5];
        const sigmaX = (options.sigmaX as number) || 0;

        const blurred = new cv.Mat();
        cv.GaussianBlur(
          processedImage,
          blurred,
          new cv.Size(kSize[0], kSize[1]),
          sigmaX,
          sigmaX,
          cv.BORDER_DEFAULT
        );
        processedImage.delete();
        processedImage = blurred;
        break;
      }

      case 'MedianBlur': {
        const options = processorConfig.options || {};
        const kSize = (options.kSize as number) || 5;

        const blurred = new cv.Mat();
        cv.medianBlur(processedImage, blurred, kSize);
        processedImage.delete();
        processedImage = blurred;
        break;
      }

      // Add more pre-processors as needed
      default:
        console.warn(`Unknown pre-processor: ${processorConfig.name}`);
    }
  }

  return processedImage;
}

/**
 * Export results to CSV format.
 */
export function exportToCSV(
  results: Map<string, FieldDetectionResult>,
  parsedTemplate: ReturnType<typeof TemplateLoader.loadFromJSON>
): string {
  const fieldIds = TemplateLoader.getSortedFieldIds(parsedTemplate);

  // Header
  const header = 'Question,Answer,Confidence,MultiMarked,Threshold\n';

  // Rows
  const rows = fieldIds.map((fieldId) => {
    const result = results.get(fieldId);
    const field = parsedTemplate.fields.get(fieldId);

    if (!result) {
      return `${fieldId},,,false,`;
    }

    const answer = result.detectedAnswer || field?.emptyValue || '';
    const markedBubble = result.bubbles.find((b) => b.isMarked);
    const confidence = markedBubble ? (markedBubble.confidence * 100).toFixed(1) : '0.0';

    return `${fieldId},${answer},${confidence}%,${result.isMultiMarked},${result.threshold.toFixed(1)}`;
  });

  return header + rows.join('\n');
}

/**
 * Example usage in a web app.
 */
export async function handleFileUpload(
  imageInput: HTMLInputElement,
  templateInput: HTMLInputElement
): Promise<void> {
  const imageFile = imageInput.files?.[0];
  const templateFile = templateInput.files?.[0];

  if (!imageFile || !templateFile) {
    alert('Please select both image and template files');
    return;
  }

  try {
    // Load template JSON
    const templateText = await templateFile.text();
    const templateJson = JSON.parse(templateText) as TemplateConfig;

    // Detect OMR
    const parsedTemplate = TemplateLoader.loadFromJSON(templateJson);
    const results = await detectOMRSheet(imageFile, templateJson);

    // Generate CSV
    const csv = exportToCSV(results, parsedTemplate);

    // Download CSV
    downloadCSV(csv, 'omr-results.csv');

    // Display results
    displayResults(results);
  } catch (error) {
    console.error('OMR detection failed:', error);
    alert(`Error: ${(error as Error).message}`);
  }
}

/**
 * Download CSV file.
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
 * Display results in UI.
 */
function displayResults(results: Map<string, FieldDetectionResult>): void {
  const resultsDiv = document.getElementById('results');
  if (!resultsDiv) return;

  resultsDiv.innerHTML = '<h3>Detection Results</h3>';

  const table = document.createElement('table');
  table.innerHTML = `
    <thead>
      <tr>
        <th>Question</th>
        <th>Answer</th>
        <th>Confidence</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
    </tbody>
  `;

  const tbody = table.querySelector('tbody')!;

  for (const [fieldId, result] of results.entries()) {
    const row = tbody.insertRow();

    row.insertCell(0).textContent = fieldId;
    row.insertCell(1).textContent = result.detectedAnswer || '-';

    const markedBubble = result.bubbles.find((b) => b.isMarked);
    const confidence = markedBubble ? (markedBubble.confidence * 100).toFixed(1) : '0.0';
    row.insertCell(2).textContent = `${confidence}%`;

    const status = result.isMultiMarked
      ? '⚠️ Multi-marked'
      : result.detectedAnswer
        ? '✅ Detected'
        : '❌ Empty';
    row.insertCell(3).textContent = status;

    // Color code rows
    if (result.isMultiMarked) {
      row.style.backgroundColor = '#fff3cd';
    } else if (!result.detectedAnswer) {
      row.style.backgroundColor = '#f8d7da';
    }
  }

  resultsDiv.appendChild(table);
}

