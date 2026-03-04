/**
 * E2E browser test: OMRChecker full pipeline — BubbleReader stage
 *
 * Extends the existing e2e-sample1.test.ts pipeline to add Stage 3: BubbleRead.
 * Runs the complete OMR pipeline on samples/1-mobile-camera/MobileCamera/sheet1.jpg:
 *   [1] CropPage    — detect sheet boundary, perspective-warp
 *   [2] CropOnMarkers — detect 4 corner markers, align sheet
 *   [3] BubbleReader — detect & interpret filled bubbles
 *
 * Python expected output (snapshot from test_all_samples.py):
 *   Medium=E, Roll=503110026
 *   q1=B, q2=-, q3=D, q4=B, q5=6, q6=11, q7=20, q8=7, q9=16
 *   q10=B, q11=D, q12=C, q13=D, q14=A, q15=D, q16=B, q17=A, q18=C, q19=C, q20=D
 *
 * Unlike e2e-sample1.test.ts (which uses the compiled IIFE bundle), this test
 * inlines the OMR logic as plain JS strings to exercise the algorithm code
 * directly, independent of the build pipeline.
 *
 * Run with:
 *   cd omrchecker-js/packages/core && npx playwright test tests/browser/e2e-omrchecker.test.ts --reporter=list
 */

import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from './browser-setup';
import { withMemoryTracking } from './memory-utils';
import { SHARED_UTILS_SCRIPT } from './processors/shared-browser-utils';
import { CROP_ON_MARKERS_SCRIPT } from './processors/shared-crop-on-markers-script';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Real sample files
const SHEET1_PATH = join(__dirname, '../../../../../samples/1-mobile-camera/MobileCamera/sheet1.jpg');
const MARKER_PATH = join(__dirname, '../../../../../samples/1-mobile-camera/omr_marker.jpg');
const TEMPLATE_PATH = join(__dirname, '../../../../../samples/1-mobile-camera/template.json');

// Allow extra time: CDN OpenCV.js load + WASM init + full pipeline
test.setTimeout(120_000);

// ─────────────────────────────────────────────────────────────────────────────
// BubbleReader + template parser + runOMRChecker orchestrator
//
// This script is specific to this test file — all shared infrastructure
// (WarpMethod, MathUtils, ImageUtils, WarpStrategyFactory, WarpOnPointsCommon,
// CropOnMarkers, __loadGray, __cropPage) is injected via SHARED_UTILS_SCRIPT
// and CROP_ON_MARKERS_SCRIPT before this runs.
// ─────────────────────────────────────────────────────────────────────────────

const BUBBLE_READER_SCRIPT = `
(function() {

const cv = window.cv;
const MathUtils = window.MathUtils;
const ImageUtils = window.ImageUtils;

// ── BubbleReader ──────────────────────────────────────────────────────────────
// Ports: BubbleReader.ts + LocalThresholdStrategy

window.BubbleReader = class BubbleReader {
  constructor(config={}) {
    this.minJump = config.minJump ?? 30;
    this.minGapTwoBubbles = config.minGapTwoBubbles ?? 20;
    this.minJumpSurplusForGlobalFallback = config.minJumpSurplusForGlobalFallback ?? 10;
    this.globalFallbackThreshold = config.globalFallbackThreshold ?? 127.5;
  }

  readBubbles(grayImage, template) {
    const response = {};
    const { globalFallbackThreshold } = this;

    for (const field of template.allFields) {
      const bubbleMeans = [];

      for (const scanBox of field.scanBoxes) {
        const [w, h] = scanBox.bubbleDimensions;
        const [sx, sy] = scanBox.getShiftedPosition();
        const x = Math.round(sx), y = Math.round(sy);
        const safeX = Math.max(0, Math.min(x, grayImage.cols - 1));
        const safeY = Math.max(0, Math.min(y, grayImage.rows - 1));
        const safeW = Math.min(w, grayImage.cols - safeX);
        const safeH = Math.min(h, grayImage.rows - safeY);

        if (safeW <= 0 || safeH <= 0) { bubbleMeans.push(255); continue; }

        const roi = grayImage.roi(new cv.Rect(safeX, safeY, safeW, safeH));
        try { bubbleMeans.push(cv.mean(roi)[0]); } finally { roi.delete(); }
      }

      const threshold = this._localThreshold(bubbleMeans, globalFallbackThreshold);
      const marked = field.scanBoxes.filter((_,i) => bubbleMeans[i] < threshold);
      const value = (marked.length === 0 || marked.length === field.scanBoxes.length)
        ? field.emptyValue
        : marked.map(sb => sb.bubbleValue).join('');
      response[field.fieldLabel] = value;
    }

    return response;
  }

  _localThreshold(bubbleMeans, globalFallback) {
    if (bubbleMeans.length < 2) return globalFallback;
    const sorted = [...bubbleMeans].sort((a, b) => a - b);
    if (sorted.length === 2) {
      const gap = sorted[1] - sorted[0];
      return gap < this.minGapTwoBubbles ? globalFallback : (sorted[0]+sorted[1])/2;
    }
    let maxJump = 0, localThreshold = globalFallback;
    for (let i = 1; i < sorted.length - 1; i++) {
      const jump = sorted[i+1] - sorted[i-1];
      if (jump > maxJump) { maxJump = jump; localThreshold = sorted[i-1] + jump/2; }
    }
    const confidentJump = this.minJump + this.minJumpSurplusForGlobalFallback;
    return maxJump < confidentJump ? globalFallback : localThreshold;
  }
};

// ── Template parser (matches TypeScript Template class) ───────────────────────
// Minimal version — only parses what BubbleReader needs:
//   allFields: [{ fieldLabel, emptyValue, scanBoxes: [{ bubbleDimensions, getShiftedPosition, bubbleValue }] }]

const BUILTIN_BUBBLE_FIELD_TYPES = {
  QTYPE_INT:  { bubble_values: ['0','1','2','3','4','5','6','7','8','9'], direction: 'vertical' },
  QTYPE_MCQ4: { bubble_values: ['A','B','C','D'],                        direction: 'horizontal' },
  QTYPE_MCQ5: { bubble_values: ['A','B','C','D','E'],                    direction: 'horizontal' },
  QTYPE_MCQ6: { bubble_values: ['A','B','C','D','E','F'],                direction: 'horizontal' },
  QTYPE_BOOL: { bubble_values: ['T','F'],                                direction: 'horizontal' },
};

function parseFieldLabels(labelStrings) {
  const labels = [];
  for (const str of labelStrings) {
    // Pattern: "prefix<n>..<m>" → prefix+n, prefix+(n+1), ..., prefix+m
    const rangeMatch = str.match(/^(.*?)(-?\\d+)\\.\\.(-?\\d+)$/);
    if (rangeMatch) {
      const prefix = rangeMatch[1];
      const start = parseInt(rangeMatch[2]);
      const end = parseInt(rangeMatch[3]);
      const step = start <= end ? 1 : -1;
      for (let i = start; step > 0 ? i <= end : i >= end; i += step) {
        labels.push(prefix + i);
      }
    } else {
      labels.push(str);
    }
  }
  return labels;
}

function parseTemplate(templateJson) {
  const globalEmptyVal = templateJson.emptyValue ?? templateJson.empty_value ?? '';
  const globalBubbleDimensions = templateJson.bubbleDimensions ?? templateJson.bubble_dimensions ?? [40,40];
  const fieldBlocksOffset = templateJson.fieldBlocksOffset ?? templateJson.field_blocks_offset ?? [0,0];

  // Merge custom bubble field types
  const customTypes = templateJson.customBubbleFieldTypes ?? templateJson.custom_bubble_field_types ?? {};
  const bubbleFieldTypesData = Object.assign({}, BUILTIN_BUBBLE_FIELD_TYPES);
  for (const [typeName, typeData] of Object.entries(customTypes)) {
    const vals = typeData.bubbleValues ?? typeData.bubble_values ?? [];
    const dir = typeData.direction ?? 'horizontal';
    bubbleFieldTypesData[typeName] = { bubble_values: vals, direction: dir };
  }

  const fieldBlocksRaw = templateJson.fieldBlocks ?? templateJson.field_blocks ?? {};
  const allFields = [];

  for (const [blockName, blockRaw] of Object.entries(fieldBlocksRaw)) {
    const bubbleFieldType = blockRaw.bubbleFieldType ?? blockRaw.bubble_field_type;
    const typeData = bubbleFieldTypesData[bubbleFieldType];
    if (!typeData) continue; // skip unknown types

    const bubbleValues = typeData.bubble_values;
    const direction = typeData.direction;
    const emptyValue = blockRaw.emptyValue ?? blockRaw.empty_value ?? globalEmptyVal;
    const bubbleDimensions = blockRaw.bubbleDimensions ?? blockRaw.bubble_dimensions ?? globalBubbleDimensions;
    const bubblesGap = blockRaw.bubblesGap ?? blockRaw.bubbles_gap ?? 0;
    const labelsGap = blockRaw.labelsGap ?? blockRaw.labels_gap ?? 0;
    const fieldLabels = parseFieldLabels(blockRaw.fieldLabels ?? blockRaw.field_labels ?? []);
    const originRaw = blockRaw.origin ?? [0, 0];
    const origin = [originRaw[0] + fieldBlocksOffset[0], originRaw[1] + fieldBlocksOffset[1]];

    // Direction: vertical → bubbles stack on Y-axis; horizontal → X-axis
    const isVertical = direction === 'vertical';
    const v = isVertical ? 0 : 1; // axis for label gap
    const h = isVertical ? 1 : 0; // axis for bubble gap

    const leadPoint = [origin[0], origin[1]];

    for (let li = 0; li < fieldLabels.length; li++) {
      const fieldLabel = fieldLabels[li];
      const fieldOriginX = leadPoint[0];
      const fieldOriginY = leadPoint[1];

      const scanBoxes = bubbleValues.map((value, idx) => {
        const bx = fieldOriginX + (h === 0 ? idx * bubblesGap : 0);
        const by = fieldOriginY + (h === 1 ? idx * bubblesGap : 0);
        return {
          bubbleValue: value,
          bubbleDimensions: bubbleDimensions,
          getShiftedPosition: () => [bx, by],
        };
      });

      leadPoint[v] += labelsGap;
      allFields.push({ fieldLabel, emptyValue, scanBoxes });
    }
  }

  // processingImageShape: check root, then preProcessors, then derive from templateDimensions
  let processingImageShape =
    templateJson.processingImageShape ??
    templateJson.processing_image_shape ??
    null;

  if (!processingImageShape) {
    const preProcessors = templateJson.preProcessors ?? templateJson.pre_processors ?? [];
    for (const pp of preProcessors) {
      const opts = pp.options ?? {};
      const shape = opts.processingImageShape ?? opts.processing_image_shape;
      if (shape) { processingImageShape = shape; break; }
    }
  }

  if (!processingImageShape) {
    const [pw, ph] = templateJson.templateDimensions ?? [1846, 1500];
    processingImageShape = [ph, pw];
  }

  return { allFields, processingImageShape };
}
window.parseTemplate = parseTemplate;

// ── runOMRChecker ─────────────────────────────────────────────────────────────
// Full pipeline: loadGray → CropPage → CropOnMarkers → resize → BubbleReader

window.runOMRChecker = async function(sheetB64, markerB64, templateJson) {
  const mats = [];

  // 1. Load source image
  const gray = await window.__loadGray(sheetB64);
  mats.push(gray);

  // 2. CropPage
  const { warped: croppedPage } = window.__cropPage(gray);
  mats.push(croppedPage);

  // 3. Get template (needed for dimensions)
  const template = window.parseTemplate(templateJson);

  // 3a. Resize to CropOnMarkers processingImageShape BEFORE running CropOnMarkers.
  // Python: ImageProcessorBase.apply_filter_with_context resizes to processor.processing_image_shape
  // template.json CropOnMarkers options has processingImageShape: [900, 650] (h=900, w=650)
  const cropOnMarkersShape = templateJson.preProcessors
    .find(pp => pp.name === 'CropOnMarkers')?.options?.processingImageShape;
  let preAligned = croppedPage;
  if (cropOnMarkersShape) {
    const [ppH, ppW] = cropOnMarkersShape;
    preAligned = new cv.Mat();
    cv.resize(croppedPage, preAligned, new cv.Size(ppW, ppH));
    mats.push(preAligned);
  }

  // 3b. CropOnMarkers
  const cropOnMarkersOptions = {
    type: 'FOUR_MARKERS',
    reference_image: 'omr_marker.jpg',
    marker_dimensions: [35, 35],
    tuning_options: {
      marker_rescale_range: [80, 120],
      marker_rescale_steps: 5,
      min_matching_threshold: 0.2,
      apply_erode_subtract: true,
    },
  };
  const processor = await window.CropOnMarkers.fromBase64(
    cropOnMarkersOptions,
    { 'omr_marker.jpg': markerB64 }
  );
  const [aligned] = processor.applyFilter(preAligned, null, null, 'sheet1.jpg');
  processor.dispose();
  mats.push(aligned);

  // 4. Resize to templateDimensions for bubble detection.
  const [templateW, templateH] = templateJson.templateDimensions; // [1846, 1500]
  const resized = new cv.Mat();
  cv.resize(aligned, resized, new cv.Size(templateW, templateH));
  mats.push(resized);

  // 5. BubbleReader
  const reader = new window.BubbleReader();
  const response = reader.readBubbles(resized, template);

  const dims = [resized.cols, resized.rows];
  mats.forEach(m => { try { if (!m.isDeleted()) m.delete(); } catch(_) {} });

  return { response, dims };
};

})();
`;

// Combined setup: shared base → CropOnMarkers → BubbleReader + orchestrator.
const CROP_AND_MARKER_SCRIPT =
  SHARED_UTILS_SCRIPT + '\n' + CROP_ON_MARKERS_SCRIPT + '\n' + BUBBLE_READER_SCRIPT;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test.describe('E2E: OMRChecker — Full Pipeline with BubbleReader', () => {
  const sheet1B64 = readFileSync(SHEET1_PATH).toString('base64');
  const markerB64 = readFileSync(MARKER_PATH).toString('base64');
  const templateJson = JSON.parse(readFileSync(TEMPLATE_PATH, 'utf-8'));

  // Pre-inject the large data (base64 images + templateJson) as window globals
  // to avoid Playwright's argument size limits when calling page.evaluate().
  async function injectData(page: any) {
    await page.evaluate(`window.__tplJson = ${JSON.stringify(templateJson)};`);
    await page.evaluate(`window.__sheet1B64 = "${sheet1B64}";`);
    await page.evaluate(`window.__markerB64 = "${markerB64}";`);
  }

  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
    await page.evaluate(CROP_AND_MARKER_SCRIPT);
    await injectData(page);
  });

  // ── Stage 3: BubbleReader — structural checks ───────────────────────────────

  test('stage 3 — BubbleReader returns response with all expected field labels', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(async () => {
        const w = window as any;
        return await w.runOMRChecker(w.__sheet1B64, w.__markerB64, w.__tplJson);
      });

      const { response } = result;

      const expectedLabels = [
        'Medium',
        'roll1', 'roll2', 'roll3', 'roll4', 'roll5', 'roll6', 'roll7', 'roll8', 'roll9',
        'q5_1', 'q5_2', 'q6_1', 'q6_2', 'q7_1', 'q7_2', 'q8_1', 'q8_2', 'q9_1', 'q9_2',
        'q1', 'q2', 'q3', 'q4',
        'q10', 'q11', 'q12', 'q13',
        'q14', 'q15', 'q16',
        'q17', 'q18', 'q19', 'q20',
      ];

      for (const label of expectedLabels) {
        expect(Object.prototype.hasOwnProperty.call(response, label)).toBe(true);
      }

      for (const [key, value] of Object.entries(response)) {
        expect(typeof value).toBe('string');
      }
    });
  });

  // ── Stage 3: BubbleReader — known bubble value checks ──────────────────────

  test('stage 3 — BubbleReader detects known bubble values from Python snapshot', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(async () => {
        const w = window as any;
        return await w.runOMRChecker(w.__sheet1B64, w.__markerB64, w.__tplJson);
      });

      const { response } = result;

      expect(response['Medium']).toBe('E');
      expect(response['q1']).toBe('B');
      expect(response['q3']).toBe('D');
      expect(response['q4']).toBe('B');
      expect(response['q10']).toBe('B');
      expect(response['q11']).toBe('D');
      expect(response['q12']).toBe('C');
      expect(response['q13']).toBe('D');
      expect(response['q14']).toBe('A');
      expect(response['q15']).toBe('D');
      expect(response['q16']).toBe('B');
      expect(response['q17']).toBe('A');
      expect(response['q18']).toBe('C');
      expect(response['q19']).toBe('C');
      expect(response['q20']).toBe('D');
    });
  });

  // ── Stage 3: BubbleReader — processedImageDimensions ──────────────────────

  test('stage 3 — processed image has expected dimensions after resize', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(async () => {
        const w = window as any;
        return await w.runOMRChecker(w.__sheet1B64, w.__markerB64, w.__tplJson);
      });

      expect(result.dims[0]).toBeGreaterThan(100); // width
      expect(result.dims[1]).toBeGreaterThan(100); // height
    });
  });
});
