/**
 * Browser tests for CropOnMarkers.ts using OpenCV.js (Playwright).
 *
 * Port of Python CropOnMarkers (FOUR_MARKERS type only) verified against:
 *   samples/1-mobile-camera/MobileCamera/sheet1.jpg + omr_marker.jpg
 *
 * Tests:
 *   1. detectsMarkersInSampleImage — full pipeline on real image
 *   2. throwsWhenMarkerNotFound    — blank white image → ImageProcessingError
 *
 * Run with:
 *   cd omrchecker-js/packages/core && npx playwright test tests/browser/processors/CropOnMarkers.test.ts --reporter=list
 */

import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Real sample images
const SHEET1_PATH = join(
  __dirname,
  '../../../../../../samples/1-mobile-camera/MobileCamera/sheet1.jpg'
);
const MARKER_PATH = join(
  __dirname,
  '../../../../../../samples/1-mobile-camera/omr_marker.jpg'
);

// ─────────────────────────────────────────────────────────────────────────────
// SETUP_SCRIPT
//
// Inlines the full CropOnMarkers logic as plain JS so it can run inside
// page.evaluate() where TypeScript modules are unavailable.
//
// Implements:
//   window.__loadGray(b64)         → Promise<cv.Mat>  (grayscale, caller deletes)
//   window.__cropPage(gray)        → { warped, corners } (caller deletes warped)
//   window.CropOnMarkers           → class (accepts pre-decoded marker cv.Mat)
//   window.CropOnMarkers.fromGray  → static async factory (decodes base64 via canvas)
// ─────────────────────────────────────────────────────────────────────────────

const SETUP_SCRIPT = `
(function() {

// ── Constants ────────────────────────────────────────────────────────────────

const WarpMethod = {
  PERSPECTIVE_TRANSFORM: 'PERSPECTIVE_TRANSFORM',
  HOMOGRAPHY: 'HOMOGRAPHY',
};
window.WarpMethod = WarpMethod;

const WARP_METHOD_FLAG_VALUES = { INTER_LINEAR: 1, INTER_CUBIC: 2, INTER_NEAREST: 0 };
window.WARP_METHOD_FLAG_VALUES = WARP_METHOD_FLAG_VALUES;

// ── MathUtils ────────────────────────────────────────────────────────────────

const MathUtils = {
  distance(p1, p2) {
    return Math.hypot(p1[0]-p2[0], p1[1]-p2[1]);
  },
  getBoundingBoxOfPoints(points) {
    const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
    const minX=Math.min(...xs), minY=Math.min(...ys);
    const maxX=Math.max(...xs), maxY=Math.max(...ys);
    return {
      boundingBox: [[minX,minY],[maxX,minY],[maxX,maxY],[minX,maxY]],
      boxDimensions: [Math.floor(maxX-minX), Math.floor(maxY-minY)],
    };
  },
  shiftPointsFromOrigin(origin, pts) {
    return pts.map(p=>[origin[0]+p[0], origin[1]+p[1]]);
  },
  orderFourPoints(pts) {
    const s=pts.map(p=>p[0]+p[1]), d=pts.map(p=>p[1]-p[0]);
    const tl=pts[s.indexOf(Math.min(...s))];
    const tr=pts[d.indexOf(Math.min(...d))];
    const br=pts[s.indexOf(Math.max(...s))];
    const bl=pts[d.indexOf(Math.max(...d))];
    return { rect:[tl,tr,br,bl] };
  },
  getRectanglePoints(x, y, w, h) {
    return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]];
  },
};
window.MathUtils = MathUtils;

// ── ImageUtils ───────────────────────────────────────────────────────────────

const ImageUtils = {
  resizeSingle(img, w, h) {
    const dst = new cv.Mat();
    cv.resize(img, dst, new cv.Size(Math.floor(w), Math.floor(h)));
    return dst;
  },
  normalizeSingle(img) {
    const mm = cv.minMaxLoc(img, new cv.Mat());
    if (mm.maxVal === mm.minVal) return img.clone();
    const out = new cv.Mat();
    cv.normalize(img, out, 0, 255, cv.NORM_MINMAX);
    return out;
  },
  getCroppedWarpedRectanglePoints(corners) {
    const [tl,tr,br,bl] = corners;
    const maxW = Math.max(
      Math.floor(MathUtils.distance(tr,tl)),
      Math.floor(MathUtils.distance(br,bl))
    );
    const maxH = Math.max(
      Math.floor(MathUtils.distance(tr,br)),
      Math.floor(MathUtils.distance(tl,bl))
    );
    return [[[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], [maxW,maxH]];
  },
};
window.ImageUtils = ImageUtils;

// ── WarpStrategyFactory ──────────────────────────────────────────────────────

const WarpStrategyFactory = {
  create(method) {
    return {
      warpImage(image, coloredImage, ctrl, dest, dims) {
        const [w,h] = dims;
        const dsize = new cv.Size(w, h);
        const cMat = cv.matFromArray(4,1,cv.CV_32FC2, ctrl.flat());
        const dMat = cv.matFromArray(4,1,cv.CV_32FC2, dest.flat());
        let M;
        if (method === 'HOMOGRAPHY') {
          M = cv.findHomography(cMat, dMat, 0, 3.0);
        } else {
          M = cv.getPerspectiveTransform(cMat, dMat);
        }
        cMat.delete(); dMat.delete();
        const warpedGray = new cv.Mat();
        cv.warpPerspective(image, warpedGray, M, dsize);
        M.delete();
        return { warpedGray, warpedColored: null };
      },
    };
  },
};
window.WarpStrategyFactory = WarpStrategyFactory;

// ── WarpOnPointsCommon (abstract base) ───────────────────────────────────────

class WarpOnPointsCommon {
  constructor(options={}) {
    const parsed = this.validateAndRemapOptionsSchema(options);
    const tuning = options.tuning_options != null ? options.tuning_options : (parsed.tuning_options || {});
    this.enableCropping = parsed.enable_cropping != null ? parsed.enable_cropping : (options.enable_cropping || false);
    this.warpMethod = tuning.warp_method != null ? tuning.warp_method
      : (this.enableCropping ? WarpMethod.PERSPECTIVE_TRANSFORM : WarpMethod.HOMOGRAPHY);
    const flagKey = tuning.warp_method_flag || 'INTER_LINEAR';
    this.warpMethodFlag = WARP_METHOD_FLAG_VALUES[flagKey] != null ? WARP_METHOD_FLAG_VALUES[flagKey] : 1;
    this.coloredOutputsEnabled = options.colored_outputs_enabled || false;
    this.warpStrategy = WarpStrategyFactory.create(this.warpMethod);
  }
  validateAndRemapOptionsSchema(o) { throw new Error('Not implemented'); }
  prepareImageBeforeExtraction(img) { throw new Error('Not implemented'); }
  extractControlDestinationPoints(img, col, fp) { throw new Error('Not implemented'); }
  appendSaveImage(..._args) {}

  applyFilter(image, coloredImage, template, filePath) {
    const prepared = this.prepareImageBeforeExtraction(image);
    const [ctrl, dest, edgeMap] = this.extractControlDestinationPoints(prepared, coloredImage, filePath);
    const [parsedCtrl, parsedDest, dims] = this._parseAndPreparePoints(prepared, ctrl, dest);
    const [warped, warpedCol] = this._applyWarpStrategy(image, coloredImage, parsedCtrl, parsedDest, dims, edgeMap);
    this.appendSaveImage('Warped Image', [4,5,6], warped, warpedCol);
    return [warped, warpedCol, template];
  }

  _parseAndPreparePoints(image, ctrl, dest) {
    const seen = new Map();
    const uCtrl=[], uDest=[];
    for (let i=0; i<ctrl.length; i++) {
      const k = JSON.stringify(ctrl[i]);
      if (!seen.has(k)) { seen.set(k,1); uCtrl.push(ctrl[i]); uDest.push(dest[i]); }
    }
    const dims = this._calculateWarpedDimensions([image.cols, image.rows], uDest);
    return [uCtrl, uDest, dims];
  }

  _calculateWarpedDimensions(defaultDims, dest) {
    if (!this.enableCropping) return defaultDims;
    const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(dest);
    const shifted = MathUtils.shiftPointsFromOrigin([-boundingBox[0][0], -boundingBox[0][1]], dest);
    for (let i=0; i<dest.length; i++) dest[i] = shifted[i];
    return boxDimensions;
  }

  _applyWarpStrategy(image, coloredImage, ctrl, dest, dims, _edgeMap) {
    const [c,d,di] = this._preparePointsForStrategy(ctrl, dest, dims);
    const colIn = this.coloredOutputsEnabled ? coloredImage : null;
    const r = this.warpStrategy.warpImage(image, colIn, c, d, di);
    return [r.warpedGray, r.warpedColored];
  }

  _preparePointsForStrategy(ctrl, dest, dims) {
    if (this.warpMethod !== WarpMethod.PERSPECTIVE_TRANSFORM) return [ctrl, dest, dims];
    if (ctrl.length !== 4) throw new Error('Expected 4 control points for perspective transform, found ' + ctrl.length + '.');
    const { rect } = MathUtils.orderFourPoints(ctrl);
    const [newDest, newDims] = ImageUtils.getCroppedWarpedRectanglePoints(rect);
    return [rect, newDest, newDims];
  }
}
window.WarpOnPointsCommon = WarpOnPointsCommon;

// ── Marker detection helpers ─────────────────────────────────────────────────

/**
 * Port of prepareMarkerTemplate from marker_detection.ts
 * referenceImage: grayscale cv.Mat (caller retains ownership)
 * referenceZone: { origin:[x,y], dimensions:[w,h] }
 * Returns new cv.Mat (caller must delete)
 */
function prepareMarkerTemplate(referenceImage, referenceZone, markerDimensions, blurKernel, applyErodeSubtract) {
  const [x,y] = referenceZone.origin;
  const [w,h] = referenceZone.dimensions;
  const mats = [];
  try {
    let marker = referenceImage.roi(new cv.Rect(x,y,w,h)).clone();
    mats.push(marker);

    if (markerDimensions != null) {
      const r = ImageUtils.resizeSingle(marker, markerDimensions[0], markerDimensions[1]);
      mats.splice(mats.indexOf(marker),1); marker.delete();
      marker = r; mats.push(marker);
    }

    const blurred = new cv.Mat(); mats.push(blurred);
    cv.GaussianBlur(marker, blurred, new cv.Size(blurKernel[0], blurKernel[1]), 0);
    mats.splice(mats.indexOf(marker),1); marker.delete(); marker = blurred;

    const norm1 = new cv.Mat(); mats.push(norm1);
    cv.normalize(marker, norm1, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    mats.splice(mats.indexOf(marker),1); marker.delete(); marker = norm1;

    if (applyErodeSubtract) {
      const kernel = cv.Mat.ones(5,5,cv.CV_8U); mats.push(kernel);
      const eroded = new cv.Mat(); mats.push(eroded);
      cv.erode(marker, eroded, kernel, new cv.Point(-1,-1), 5);
      const sub = new cv.Mat(); mats.push(sub);
      cv.subtract(marker, eroded, sub);
      const norm2 = new cv.Mat(); mats.push(norm2);
      cv.normalize(sub, norm2, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
      mats.splice(mats.indexOf(marker),1); marker.delete(); marker = norm2;
    }

    mats.splice(mats.indexOf(marker),1);
    return marker;
  } finally {
    mats.forEach(m => { try { m.delete(); } catch(_) {} });
  }
}
window.prepareMarkerTemplate = prepareMarkerTemplate;

/**
 * Port of detectMarkerInPatch from marker_detection.ts
 */
function detectMarkerInPatch(patch, marker, zoneOffset, scaleRange, scaleSteps, minConfidence) {
  const descentPerStep = Math.floor((scaleRange[1] - scaleRange[0]) / scaleSteps);
  const mH = marker.rows, mW = marker.cols;
  const pH = patch.rows, pW = patch.cols;

  let bestPos = null, bestMarker = null, bestConf = 0.0;

  for (let sp=scaleRange[1]; sp>scaleRange[0]; sp-=descentPerStep) {
    const scale = sp/100;
    if (scale <= 0) continue;
    const sw = Math.floor(mW*scale), sh = Math.floor(mH*scale);
    if (sh>pH||sw>pW||sw<1||sh<1) continue;

    const sm = ImageUtils.resizeSingle(marker, sw, sh);
    const res = new cv.Mat();
    cv.matchTemplate(patch, sm, res, cv.TM_CCOEFF_NORMED);
    const loc = cv.minMaxLoc(res, new cv.Mat());
    res.delete();

    if (loc.maxVal > bestConf) {
      if (bestMarker) bestMarker.delete();
      bestConf = loc.maxVal;
      bestPos = [loc.maxLoc.x, loc.maxLoc.y];
      bestMarker = sm;
    } else {
      sm.delete();
    }
  }

  if (bestPos === null || bestMarker === null) return null;
  if (bestConf < minConfidence) { bestMarker.delete(); return null; }

  const h2=bestMarker.rows, w2=bestMarker.cols;
  const [x,y] = bestPos;
  const corners = MathUtils.getRectanglePoints(x,y,w2,h2);
  const abs = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);
  bestMarker.delete();
  return abs;
}
window.detectMarkerInPatch = detectMarkerInPatch;

// ── CropOnMarkers ────────────────────────────────────────────────────────────

const FOUR_MARKERS_ZONE_ORDER = [
  'topLeftMarker', 'topRightMarker', 'bottomRightMarker', 'bottomLeftMarker'
];

/**
 * CropOnMarkers browser implementation.
 *
 * Constructor accepts a pre-decoded grayscale cv.Mat for the reference image,
 * passed via assetMats: { [referenceImageKey]: cv.Mat }.
 *
 * For convenience, use the static async factory CropOnMarkers.fromBase64(options, assets)
 * which decodes the base64 string via canvas/Image, then calls the constructor.
 */
class CropOnMarkers extends WarpOnPointsCommon {
  /**
   * @param options     - CropOnMarkers options (type, reference_image, tuning_options, ...)
   * @param assetMats   - Map of filename key → pre-decoded grayscale cv.Mat
   */
  constructor(options, assetMats={}) {
    super(options);
    if (options.type !== 'FOUR_MARKERS') throw new Error('Only FOUR_MARKERS is supported');
    this.referenceImageKey = options.reference_image;
    this.markerDimensions = options.marker_dimensions || null;
    const t = options.tuning_options || {};
    this.minMatchingThreshold = t.min_matching_threshold != null ? t.min_matching_threshold : 0.3;
    this.markerRescaleRange = t.marker_rescale_range || [85, 115];
    this.markerRescaleSteps = t.marker_rescale_steps != null ? t.marker_rescale_steps : 5;
    this.applyErodeSubtract = t.apply_erode_subtract != null ? t.apply_erode_subtract : true;
    this.markerTemplates = new Map();
    const refMat = assetMats[this.referenceImageKey];
    if (!refMat) throw new Error('Asset Mat not found for key: ' + this.referenceImageKey);
    this._initResizedMarkers(refMat);
  }

  /**
   * Async factory: decodes base64 via canvas/Image, then constructs CropOnMarkers.
   * @param options - same as constructor options
   * @param assets  - { [referenceImageKey]: base64String }
   * @returns Promise<CropOnMarkers> (caller must call .dispose() when done)
   */
  static async fromBase64(options, assets) {
    const key = options.reference_image;
    const b64 = assets[key];
    if (!b64) throw new Error('Asset not found: ' + key);
    const refMat = await window.__loadGray(b64);
    const processor = new CropOnMarkers(options, { [key]: refMat });
    refMat.delete(); // processor copied what it needs in _initResizedMarkers
    return processor;
  }

  validateAndRemapOptionsSchema(options) {
    const t = options.tuning_options || {};
    return {
      enable_cropping: true,
      tuning_options: { warp_method: t.warp_method || WarpMethod.PERSPECTIVE_TRANSFORM },
    };
  }

  prepareImageBeforeExtraction(image) {
    return ImageUtils.normalizeSingle(image);
  }

  extractControlDestinationPoints(image, _col, filePath) {
    const allCorners = [];
    for (const zoneType of FOUR_MARKERS_ZONE_ORDER) {
      const marker = this.markerTemplates.get(zoneType);
      if (!marker) throw new Error('Marker template not initialized for zone: ' + zoneType);
      const zoneDesc = this._getQuadrantZoneDescription(zoneType, image, marker);
      const corners = this._findMarkerCornersInPatch(image, zoneDesc, marker, zoneType, filePath);
      const cx = (corners[0][0]+corners[1][0]+corners[2][0]+corners[3][0])/4;
      const cy = (corners[0][1]+corners[1][1]+corners[2][1]+corners[3][1])/4;
      allCorners.push([cx, cy]);
    }
    const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(allCorners);
    return [allCorners, warpedPoints, null];
  }

  /**
   * @param refImage - pre-decoded grayscale cv.Mat (not deleted by this function)
   */
  _initResizedMarkers(refImage) {
    const zone = { origin:[1,1], dimensions:[refImage.cols-1, refImage.rows-1] };
    for (const zoneType of FOUR_MARKERS_ZONE_ORDER) {
      const m = prepareMarkerTemplate(refImage, zone, this.markerDimensions, [5,5], this.applyErodeSubtract);
      this.markerTemplates.set(zoneType, m);
    }
  }

  _getQuadrantZoneDescription(zoneType, image, marker) {
    const h=image.rows, w=image.cols;
    const halfH=Math.floor(h/2), halfW=Math.floor(w/2);
    const mH=marker.rows, mW=marker.cols;
    let zs, ze;
    if (zoneType==='topLeftMarker')          { zs=[1,1];       ze=[halfW,halfH]; }
    else if (zoneType==='topRightMarker')    { zs=[halfW,1];    ze=[w,halfH]; }
    else if (zoneType==='bottomRightMarker') { zs=[halfW,halfH]; ze=[w,h]; }
    else                                     { zs=[1,halfH];    ze=[halfW,h]; }
    const ox = Math.floor((zs[0]+ze[0]-mW)/2);
    const oy = Math.floor((zs[1]+ze[1]-mH)/2);
    const marg_h = (ze[0]-zs[0]-mW)/2 - 1;
    const marg_v = (ze[1]-zs[1]-mH)/2 - 1;
    return {
      origin: [ox,oy],
      dimensions: [mW,mH],
      margins: { top:marg_v, right:marg_h, bottom:marg_v, left:marg_h },
    };
  }

  _findMarkerCornersInPatch(image, zoneDesc, marker, zoneType, filePath) {
    const { origin:[ox,oy], dimensions:[dw,dh], margins } = zoneDesc;
    const mt = Math.max(0, Math.floor(margins.top));
    const mr = Math.max(0, Math.floor(margins.right));
    const mb = Math.max(0, Math.floor(margins.bottom));
    const ml = Math.max(0, Math.floor(margins.left));
    const px = Math.max(0, ox-ml);
    const py = Math.max(0, oy-mt);
    const pw = Math.min(image.cols-px, dw+ml+mr);
    const ph = Math.min(image.rows-py, dh+mt+mb);
    if (pw<=0||ph<=0) throw new Error('Degenerate patch for zone: ' + zoneType);
    const patch = image.roi(new cv.Rect(px, py, pw, ph)).clone();
    let corners;
    try {
      corners = detectMarkerInPatch(
        patch, marker, [px,py],
        this.markerRescaleRange, this.markerRescaleSteps, this.minMatchingThreshold
      );
    } finally {
      patch.delete();
    }
    if (!corners) {
      const err = new Error('No marker found in patch for zone: ' + zoneType);
      err.name = 'ImageProcessingError';
      throw err;
    }
    return corners;
  }

  dispose() {
    for (const m of this.markerTemplates.values()) {
      try { if (!m.isDeleted()) m.delete(); } catch(_) {}
    }
    this.markerTemplates.clear();
  }
}
window.CropOnMarkers = CropOnMarkers;

// ── Image loader (canvas-based, async) ───────────────────────────────────────

/**
 * Decode a base64 JPEG → grayscale cv.Mat via canvas.
 * Caller must delete the returned Mat.
 */
window.__loadGray = async function(b64) {
  const img = new Image();
  img.src = 'data:image/jpeg;base64,' + b64;
  await new Promise((res,rej) => { img.onload=res; img.onerror=rej; });
  const canvas = document.createElement('canvas');
  canvas.width=img.width; canvas.height=img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img,0,0);
  const rgba = cv.matFromImageData(ctx.getImageData(0,0,img.width,img.height));
  const gray = new cv.Mat();
  cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
  rgba.delete();
  return gray;
};

// ── CropPage helper ───────────────────────────────────────────────────────────

window.__cropPage = function(gray) {
  const trunc1 = new cv.Mat();
  cv.threshold(gray, trunc1, 210, 255, cv.THRESH_TRUNC);
  const prepared = new cv.Mat();
  cv.normalize(trunc1, prepared, 0, 255, cv.NORM_MINMAX);
  trunc1.delete();

  const trunc2 = new cv.Mat();
  cv.threshold(prepared, trunc2, 200, 255, cv.THRESH_TRUNC);
  prepared.delete();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(15,15));
  const closed = new cv.Mat();
  cv.morphologyEx(trunc2, closed, cv.MORPH_CLOSE, kernel);
  kernel.delete(); trunc2.delete();
  const canny = new cv.Mat();
  cv.Canny(closed, canny, 185, 55);
  closed.delete();

  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(canny, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  canny.delete(); hier.delete();

  const hulls = [];
  for (let i=0; i<contours.size(); i++) {
    const c=contours.get(i);
    const h=new cv.Mat();
    cv.convexHull(c,h);
    hulls.push({h, area: cv.contourArea(h)});
    c.delete();
  }
  contours.delete();
  hulls.sort((a,b)=>b.area-a.area);
  hulls.slice(5).forEach(x=>x.h.delete());
  const top = hulls.slice(0,5);

  function angle(p1,p2,p0) {
    const dx1=p1[0]-p0[0],dy1=p1[1]-p0[1],dx2=p2[0]-p0[0],dy2=p2[1]-p0[1];
    return (dx1*dx2+dy1*dy2)/Math.sqrt((dx1**2+dy1**2)*(dx2**2+dy2**2)+1e-10);
  }
  function checkMaxCosine(pts) {
    let mx=0;
    for(let i=2;i<5;i++) mx=Math.max(mx,Math.abs(angle(pts[i%4],pts[i-2],pts[i-1])));
    return mx<0.35;
  }

  let corners=null;
  for (const {h} of top) {
    if (cv.contourArea(h)<80000) continue;
    const peri=cv.arcLength(h,true);
    const approx=new cv.Mat();
    cv.approxPolyDP(h, approx, 0.025*peri, true);
    if (approx.rows===4) {
      const pts=[];
      for(let i=0;i<4;i++) pts.push([approx.data32S[i*2], approx.data32S[i*2+1]]);
      approx.delete();
      if(checkMaxCosine(pts)) {
        const { rect } = MathUtils.orderFourPoints(pts);
        corners = rect; break;
      }
    } else { approx.delete(); }
  }
  top.forEach(x=>x.h.delete());
  if (!corners) throw new Error('CropPage: could not detect page boundary');

  // Perspective warp
  const [tl,tr,br,bl] = corners;
  const w=Math.max(
    Math.floor(MathUtils.distance(tr,tl)), Math.floor(MathUtils.distance(br,bl)));
  const h2=Math.max(
    Math.floor(MathUtils.distance(tr,br)), Math.floor(MathUtils.distance(tl,bl)));
  const srcPts = cv.matFromArray(4,1,cv.CV_32FC2,[tl[0],tl[1],tr[0],tr[1],br[0],br[1],bl[0],bl[1]]);
  const dstPts = cv.matFromArray(4,1,cv.CV_32FC2,[0,0,w-1,0,w-1,h2-1,0,h2-1]);
  const M = cv.getPerspectiveTransform(srcPts, dstPts);
  srcPts.delete(); dstPts.delete();
  const warped = new cv.Mat();
  cv.warpPerspective(gray, warped, M, new cv.Size(w,h2));
  M.delete();
  return { warped, corners };
};

})();
`;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test.describe('CropOnMarkers - Browser Tests', () => {
  // Allow extra time for OpenCV.js CDN load + WASM init + image processing
  test.use({ timeout: 90_000 });

  const sheet1B64 = readFileSync(SHEET1_PATH).toString('base64');
  const markerB64 = readFileSync(MARKER_PATH).toString('base64');

  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
    await page.evaluate(SETUP_SCRIPT);
  });

  // ─── Test 1: detectsMarkersInSampleImage ─────────────────────────────────────
  //
  // Load samples/1-mobile-camera/MobileCamera/sheet1.jpg, pass through CropPage,
  // then run CropOnMarkers with the omr_marker.jpg asset (decoded via canvas).
  // Verify: output is a valid grayscale image with reasonable dimensions.
  //
  test('detectsMarkersInSampleImage', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(
        async ([sheetB64, mrkB64]: string[]) => {
          const mats: any[] = [];
          try {
            // Stage 1: load and CropPage
            const gray = await (window as any).__loadGray(sheetB64);
            mats.push(gray);
            const { warped: croppedPage } = (window as any).__cropPage(gray);
            mats.push(croppedPage);

            // Stage 2: CropOnMarkers (use static factory to decode marker via canvas)
            const processor = await (window as any).CropOnMarkers.fromBase64(
              {
                type: 'FOUR_MARKERS',
                reference_image: 'omr_marker.jpg',
                marker_dimensions: [35, 35],
                tuning_options: {
                  marker_rescale_range: [80, 120],
                  marker_rescale_steps: 5,
                  min_matching_threshold: 0.2,
                  apply_erode_subtract: true,
                },
              },
              { 'omr_marker.jpg': mrkB64 }
            );

            const [warped] = processor.applyFilter(croppedPage, null, null, 'sheet1.jpg');
            mats.push(warped);
            processor.dispose();

            return {
              rows: warped.rows,
              cols: warped.cols,
              channels: warped.channels(),
              mean: (window as any).cv.mean(warped)[0],
            };
          } finally {
            mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
          }
        },
        [sheet1B64, markerB64]
      );

      // Output must be a valid grayscale image
      expect(result.channels).toBe(1);
      // The warped sheet should be at least 100px in each dimension
      expect(result.rows).toBeGreaterThan(100);
      expect(result.cols).toBeGreaterThan(100);
      // Aligned OMR sheet should be mostly white/light (OMR forms have white background)
      expect(result.mean).toBeGreaterThan(50);
    });
  });

  // ─── Test 2: throwsWhenMarkerNotFound ────────────────────────────────────────
  //
  // Pass a blank white 300x400 image — no markers present.
  // CropOnMarkers should throw an error (ImageProcessingError / Error).
  //
  test('throwsWhenMarkerNotFound', async ({ page }) => {
    const result = await page.evaluate(
      async (mrkB64: string) => {
        const mats: any[] = [];
        try {
          // Create a blank white image (no markers)
          const blankImage = new (window as any).cv.Mat(
            400, 300, (window as any).cv.CV_8UC1,
            new (window as any).cv.Scalar(255)
          );
          mats.push(blankImage);

          // Use a very high confidence threshold → marker won't be found in blank image
          const processor = await (window as any).CropOnMarkers.fromBase64(
            {
              type: 'FOUR_MARKERS',
              reference_image: 'omr_marker.jpg',
              marker_dimensions: [35, 35],
              tuning_options: {
                marker_rescale_range: [80, 120],
                marker_rescale_steps: 5,
                min_matching_threshold: 0.99, // very high → will not match in blank image
                apply_erode_subtract: true,
              },
            },
            { 'omr_marker.jpg': mrkB64 }
          );

          try {
            processor.applyFilter(blankImage, null, null, 'blank.jpg');
            processor.dispose();
            return { threw: false, message: '' };
          } catch (e: any) {
            processor.dispose();
            return { threw: true, message: e.message || String(e) };
          }
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      },
      markerB64
    );

    expect(result.threw).toBe(true);
    // Should mention "marker" or "zone" or similar in the error message
    expect(result.message.toLowerCase()).toMatch(/marker|zone|patch/i);
  });
});
