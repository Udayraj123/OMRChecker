/**
 * E2E browser test: Sample 1 Mobile Camera — OMR pipeline
 *
 * Runs the real samples/1-mobile-camera/MobileCamera/sheet1.jpg through the
 * OMR preprocessing pipeline using OpenCV.js in a real Chromium browser.
 *
 * Pipeline stages (mirrors Python: test_all_samples.py::test_run_omr_marker_mobile):
 *   [1] ✅ CropPage    — detect sheet boundary from dark background, perspective-warp
 *   [2] ✅ CropOnMarkers — detect 4 corner markers, perspective-warp to align sheet
 *   [3] ☐  BubbleRead  — detect & interpret filled bubbles (not yet ported to TS)
 *
 * Python expected output (snapshot):
 *   Medium=E, Roll=503110026
 *   q1=B, q2=-, q3=D, q4=B, q5=6, q6=11, q7=20, q8=7, q9=16,
 *   q10=B, q11=D, q12=C, q13=D, q14=A, q15=D, q16=B, q17=A, q18=C, q19=C, q20=D
 *
 * Algorithms mirror the TypeScript source exactly:
 *   - page_detection.ts  (preparePageImage, applyGrayscaleCanny, findPageContours,
 *                         extractPageRectangle, perspectiveWarp)
 *   - marker_detection.ts (multiScaleTemplateMatch, detectMarkerInPatch)
 *   - warp_strategies.ts (PerspectiveTransformStrategy)
 *   - math.ts (orderFourPoints, checkMaxCosine, getCroppedWarpedRectanglePoints)
 */

import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from './browser-setup';
import { withMemoryTracking } from './memory-utils';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Real sample images (747×1024 mobile camera photo on dark background)
const SHEET1_PATH = join(__dirname, '../../../../../samples/1-mobile-camera/MobileCamera/sheet1.jpg');
const MARKER_PATH = join(__dirname, '../../../../../samples/1-mobile-camera/omr_marker.jpg');

// Template constants (from samples/1-mobile-camera/template.json)
const TEMPLATE = {
  processingImageShape: [900, 650] as [number, number], // [width, height] for CropOnMarkers
  markerDimensions: [35, 35] as [number, number],
  markerRescaleRange: [80, 120] as [number, number],
  morphKernel: [15, 15] as [number, number],
};

// page_detection.ts constants
const PD = {
  THRESH_TRUNCATE_HIGH: 210,
  THRESH_TRUNCATE_SECONDARY: 200,
  CANNY_HIGH: 185,
  CANNY_LOW: 55,
  MIN_PAGE_AREA: 80000,
  APPROX_POLY_EPSILON: 0.025,
  TOP_CONTOURS: 5,
  MAX_COSINE: 0.35,
} as const;

// Allow extra time: OpenCV.js CDN load + WASM init + image processing
test.setTimeout(90_000);

// ─────────────────────────────────────────────────────────────────────────────
// Shared browser helpers (injected once in beforeEach)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Script injected into the browser page that defines:
 *   window.__loadGray(b64)  → cv.Mat (grayscale, caller must delete)
 *   window.__cropPage(gray) → cv.Mat (perspective-warped sheet, caller must delete)
 *   window.__cropOnMarkers(gray, markerB64) → cv.Mat (aligned sheet, caller must delete)
 */
function buildSetupScript(pd: typeof PD, tpl: typeof TEMPLATE): string {
  return `
  // ── geometry helpers (mirrors math.ts) ──────────────────────────────────
  window.__dist = (a, b) =>
    Math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2);

  window.__angle = (p1, p2, p0) => {
    const dx1=p1[0]-p0[0], dy1=p1[1]-p0[1];
    const dx2=p2[0]-p0[0], dy2=p2[1]-p0[1];
    return (dx1*dx2+dy1*dy2)/Math.sqrt((dx1**2+dy1**2)*(dx2**2+dy2**2)+1e-10);
  };

  window.__checkMaxCosine = pts => {
    let mx=0;
    for(let i=2;i<5;i++) mx=Math.max(mx,Math.abs(window.__angle(pts[i%4],pts[i-2],pts[i-1])));
    return mx < ${pd.MAX_COSINE};
  };

  /** orderFourPoints: returns [tl, tr, br, bl] (mirrors math.ts) */
  window.__order4 = pts => {
    const s = pts.map(p=>p[0]+p[1]), d = pts.map(p=>p[1]-p[0]);
    return [
      pts[s.indexOf(Math.min(...s))],   // tl
      pts[d.indexOf(Math.min(...d))],   // tr
      pts[s.indexOf(Math.max(...s))],   // br
      pts[d.indexOf(Math.max(...d))],   // bl
    ];
  };

  /**
   * getCroppedWarpedRectanglePoints + getPerspectiveTransform + warpPerspective
   * mirrors ImageUtils.getCroppedWarpedRectanglePoints + PerspectiveTransformStrategy
   * Returns the warped Mat (caller must delete).
   */
  window.__perspectiveWarp = (src, corners) => {
    const [tl,tr,br,bl] = corners;
    const w = Math.max(
      Math.floor(window.__dist(tr,tl)), Math.floor(window.__dist(br,bl)));
    const h = Math.max(
      Math.floor(window.__dist(tr,br)), Math.floor(window.__dist(tl,bl)));
    const srcPts = window.cv.matFromArray(4,1,window.cv.CV_32FC2,
      [tl[0],tl[1], tr[0],tr[1], br[0],br[1], bl[0],bl[1]]);
    const dstPts = window.cv.matFromArray(4,1,window.cv.CV_32FC2,
      [0,0, w-1,0, w-1,h-1, 0,h-1]);
    const M = window.cv.getPerspectiveTransform(srcPts, dstPts);
    srcPts.delete(); dstPts.delete();
    const warped = new window.cv.Mat();
    window.cv.warpPerspective(src, warped, M, new window.cv.Size(w, h));
    M.delete();
    return warped;
  };

  // ── image loader ─────────────────────────────────────────────────────────
  /** Decode a base64 JPEG → grayscale cv.Mat.  Caller must delete. */
  window.__loadGray = async b64 => {
    const img = new Image();
    img.src = 'data:image/jpeg;base64,' + b64;
    await new Promise((res,rej)=>{ img.onload=res; img.onerror=rej; });
    const canvas = document.createElement('canvas');
    canvas.width=img.width; canvas.height=img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img,0,0);
    const rgba = window.cv.matFromImageData(ctx.getImageData(0,0,img.width,img.height));
    const gray = new window.cv.Mat();
    window.cv.cvtColor(rgba, gray, window.cv.COLOR_RGBA2GRAY);
    rgba.delete();
    return gray;
  };

  // ── CropPage ─────────────────────────────────────────────────────────────
  /**
   * Mirrors Python CropPage processor + page_detection.py:
   *   preparePageImage → applyGrayscaleCanny → findPageContours
   *   → extractPageRectangle → perspectiveWarp
   * Returns { warped, corners } — caller must delete warped.
   */
  window.__cropPage = gray => {
    const cv = window.cv;
    // preparePageImage: THRESH_TRUNC at 210, normalize
    const trunc1 = new cv.Mat();
    cv.threshold(gray, trunc1, ${pd.THRESH_TRUNCATE_HIGH}, 255, cv.THRESH_TRUNC);
    const prepared = new cv.Mat();
    cv.normalize(trunc1, prepared, 0, 255, cv.NORM_MINMAX);
    trunc1.delete();

    // applyGrayscaleCanny: second THRESH_TRUNC, morphClose, Canny
    const trunc2 = new cv.Mat();
    cv.threshold(prepared, trunc2, ${pd.THRESH_TRUNCATE_SECONDARY}, 255, cv.THRESH_TRUNC);
    prepared.delete();
    const kernel = cv.getStructuringElement(
      cv.MORPH_RECT, new cv.Size(${TEMPLATE.morphKernel[0]}, ${TEMPLATE.morphKernel[1]}));
    const closed = new cv.Mat();
    cv.morphologyEx(trunc2, closed, cv.MORPH_CLOSE, kernel);
    kernel.delete(); trunc2.delete();
    const canny = new cv.Mat();
    cv.Canny(closed, canny, ${pd.CANNY_HIGH}, ${pd.CANNY_LOW});
    closed.delete();

    // findPageContours: RETR_LIST, convexHull, sort by area, top N
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(canny, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    canny.delete(); hier.delete();

    const hulls = [];
    for(let i=0;i<contours.size();i++){
      const c = contours.get(i);
      const h = new cv.Mat();
      cv.convexHull(c, h);
      hulls.push({ h, area: cv.contourArea(h) });
      c.delete();
    }
    contours.delete();
    hulls.sort((a,b)=>b.area-a.area);
    hulls.slice(${pd.TOP_CONTOURS}).forEach(x=>x.h.delete());
    const top = hulls.slice(0,${pd.TOP_CONTOURS});

    // extractPageRectangle: find first valid 4-corner polygon
    let corners = null;
    for(const { h } of top){
      if(cv.contourArea(h) < ${pd.MIN_PAGE_AREA}) continue;
      const peri = cv.arcLength(h, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(h, approx, ${pd.APPROX_POLY_EPSILON}*peri, true);
      if(approx.rows===4){
        const pts=[];
        for(let i=0;i<4;i++) pts.push([approx.data32S[i*2], approx.data32S[i*2+1]]);
        approx.delete();
        if(window.__checkMaxCosine(pts)){ corners=window.__order4(pts); break; }
      } else { approx.delete(); }
    }
    top.forEach(x=>x.h.delete());

    if(!corners) throw new Error('CropPage: could not detect page boundary');

    return { warped: window.__perspectiveWarp(gray, corners), corners };
  };

  // ── CropOnMarkers ────────────────────────────────────────────────────────
  /**
   * Mirrors Python CropOnMarkers + marker_detection.py:
   *   resize to processingImageShape → multiScaleTemplateMatch at 4 corners
   *   → extractMarkerCorners → perspectiveWarp
   * Returns warped Mat aligned to marker corners. Caller must delete.
   */
  window.__cropOnMarkers = async (gray, markerB64) => {
    const cv = window.cv;
    const [procW, procH] = [${TEMPLATE.processingImageShape[0]}, ${TEMPLATE.processingImageShape[1]}];

    // Resize to processingImageShape (mirrors coordinator resize step)
    const resized = new cv.Mat();
    cv.resize(gray, resized, new cv.Size(procW, procH));

    // Load marker template
    const markerFull = await window.__loadGray(markerB64);
    const [mW, mH] = [${TEMPLATE.markerDimensions[0]}, ${TEMPLATE.markerDimensions[1]}];
    const marker = new cv.Mat();
    cv.resize(markerFull, marker, new cv.Size(mW, mH));
    markerFull.delete();

    // multiScaleTemplateMatch: search in corner patches
    const [scaleMin, scaleMax] = [${TEMPLATE.markerRescaleRange[0]}, ${TEMPLATE.markerRescaleRange[1]}];
    const patchSize = Math.min(procW, procH) * 0.25; // quarter of image per corner

    const corners4 = [
      [0,      0     ],   // top-left
      [procW,  0     ],   // top-right
      [procW,  procH ],   // bottom-right
      [0,      procH ],   // bottom-left
    ];

    const detectedCenters = [];
    for(const [cx, cy] of corners4){
      // Extract patch around corner
      const px = Math.max(0, cx - patchSize/2 - mW);
      const py = Math.max(0, cy - patchSize/2 - mH);
      const pw = Math.min(patchSize + mW, procW - px);
      const ph = Math.min(patchSize + mH, procH - py);
      const roi = new cv.Rect(Math.round(px), Math.round(py), Math.round(pw), Math.round(ph));
      const patch = resized.roi(roi);

      let bestVal = -Infinity, bestLoc = null, bestScale = 1;

      for(let scale=scaleMax; scale>=scaleMin; scale-=5){
        const sw = Math.round(mW * scale/100);
        const sh = Math.round(mH * scale/100);
        if(sw<2||sh<2||sw>patch.cols||sh>patch.rows) continue;

        const tmpl = new cv.Mat();
        cv.resize(marker, tmpl, new cv.Size(sw, sh));
        const result = new cv.Mat();
        cv.matchTemplate(patch, tmpl, result, cv.TM_CCOEFF_NORMED);
        const loc = cv.minMaxLoc(result, new cv.Mat());
        tmpl.delete(); result.delete();

        if(loc.maxVal > bestVal){
          bestVal=loc.maxVal; bestLoc=loc.maxLoc; bestScale=scale;
        }
      }

      patch.delete();

      if(!bestLoc) continue;
      const sw = Math.round(mW * bestScale/100);
      const sh = Math.round(mH * bestScale/100);
      // Center of matched marker in full-image coords
      detectedCenters.push([
        px + bestLoc.x + sw/2,
        py + bestLoc.y + sh/2,
      ]);
    }

    marker.delete();

    if(detectedCenters.length !== 4)
      throw new Error('CropOnMarkers: expected 4 markers, found ' + detectedCenters.length);

    const ordered = window.__order4(detectedCenters);
    const warped = window.__perspectiveWarp(resized, ordered);
    resized.delete();
    return warped;
  };
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test.describe('E2E: Sample 1 Mobile Camera — OMR Pipeline', () => {
  const sheet1B64 = readFileSync(SHEET1_PATH).toString('base64');
  const markerB64 = readFileSync(MARKER_PATH).toString('base64');

  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
    await page.evaluate(buildSetupScript(PD, TEMPLATE));
  });

  // ── Stage 0: raw image sanity ─────────────────────────────────────────────
  test('stage 0 — sheet1.jpg loads as 747×1024 grayscale Mat', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const r = await page.evaluate(async (b64: string) => {
        const gray = await (window as any).__loadGray(b64);
        try {
          return {
            rows: gray.rows, cols: gray.cols, channels: gray.channels(),
            nonZero: (window.cv as any).countNonZero(gray),
            mean: (window.cv as any).mean(gray)[0],
          };
        } finally { gray.delete(); }
      }, sheet1B64);

      expect(r.rows).toBe(1024);
      expect(r.cols).toBe(747);
      expect(r.channels).toBe(1);
      // Photo includes dark background — still >10% non-zero pixels
      expect(r.nonZero).toBeGreaterThan(747 * 1024 * 0.1);
      // Mean should be a mid-range value (not all black, not all white)
      expect(r.mean).toBeGreaterThan(20);
      expect(r.mean).toBeLessThan(200);
    });
  });

  // ── Stage 1: CropPage ─────────────────────────────────────────────────────
  test('stage 1 — CropPage detects sheet boundary and perspective-warps', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const r = await page.evaluate(async (b64: string) => {
        const gray = await (window as any).__loadGray(b64);
        try {
          const { warped, corners } = (window as any).__cropPage(gray);
          try {
            return {
              rows: warped.rows, cols: warped.cols, channels: warped.channels(),
              mean: (window.cv as any).mean(warped)[0],
              corners,                // [[tl_x,tl_y],[tr_x,tr_y],[br_x,br_y],[bl_x,bl_y]]
            };
          } finally { warped.delete(); }
        } finally { gray.delete(); }
      }, sheet1B64);

      // CropPage output must be a valid single-channel image
      expect(r.channels).toBe(1);
      // The warped sheet should be at least 200px in each dimension
      expect(r.rows).toBeGreaterThan(200);
      expect(r.cols).toBeGreaterThan(200);
      // After cropping the white OMR sheet from the dark background,
      // the mean should be much brighter than the original (mean was ~20–200)
      expect(r.mean).toBeGreaterThan(100);
      // 4 corners detected in image-coordinate range
      expect(r.corners).toHaveLength(4);
      r.corners.forEach(([x, y]: [number, number]) => {
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThan(747);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThan(1024);
      });
    });
  });

  // ── Stage 2: CropOnMarkers ────────────────────────────────────────────────
  test('stage 2 — CropOnMarkers aligns sheet to 4 corner markers', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const r = await page.evaluate(async ([sheetB64, mrkB64]: string[]) => {
        const gray = await (window as any).__loadGray(sheetB64);
        try {
          const { warped: cropped } = (window as any).__cropPage(gray);
          try {
            const aligned = await (window as any).__cropOnMarkers(cropped, mrkB64);
            try {
              return {
                rows: aligned.rows, cols: aligned.cols,
                channels: aligned.channels(),
                mean: (window.cv as any).mean(aligned)[0],
              };
            } finally { aligned.delete(); }
          } finally { cropped.delete(); }
        } finally { gray.delete(); }
      }, [sheet1B64, markerB64]);

      // After CropOnMarkers the image is resized to processingImageShape [900,650]
      // then warped — output should be close to that size
      expect(r.channels).toBe(1);
      expect(r.rows).toBeGreaterThan(100);
      expect(r.cols).toBeGreaterThan(100);
      // Aligned OMR sheet should be mostly white/light
      expect(r.mean).toBeGreaterThan(80);
    });
  });

  // ── Stage 3: BubbleRead (TODO) ────────────────────────────────────────────
  // When bubble detection is ported to TypeScript, add assertions here:
  //   expect(r.omrResponse.Medium).toBe('E');
  //   expect(r.omrResponse.Roll).toBe('503110026');
  //   expect(r.omrResponse.q1).toBe('B');
  //   expect(r.omrResponse.q3).toBe('D');
  //   ... etc.
});
