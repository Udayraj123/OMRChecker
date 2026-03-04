/**
 * Shared CropOnMarkers browser-side script for Playwright tests.
 *
 * CROP_ON_MARKERS_SCRIPT: inject after SHARED_UTILS_SCRIPT to add:
 *   window.CropOnMarkers  — full CropOnMarkers processor class (FOUR_MARKERS)
 *   window.__cropPage     — lightweight CropPage helper for pre-processing
 *
 * Depends on SHARED_UTILS_SCRIPT being injected first (uses window.WarpMethod,
 * window.MathUtils, window.ImageUtils, window.WarpOnPointsCommon, window.__loadGray).
 *
 * Used by:
 *   tests/browser/processors/CropOnMarkers.test.ts
 *   tests/browser/e2e-omrchecker.test.ts
 */

export const CROP_ON_MARKERS_SCRIPT = `
(function() {

const cv = window.cv;
const WarpMethod        = window.WarpMethod;
const MathUtils         = window.MathUtils;
const ImageUtils        = window.ImageUtils;
const WarpOnPointsCommon = window.WarpOnPointsCommon;

// ── Marker detection helpers ──────────────────────────────────────────────────

/**
 * Port of prepareMarkerTemplate from marker_detection.ts.
 * referenceImage: grayscale cv.Mat (caller retains ownership)
 * Returns new cv.Mat (caller must delete)
 */
function prepareMarkerTemplate(referenceImage, referenceZone, markerDimensions, blurKernel, applyErodeSubtract) {
  const [x, y] = referenceZone.origin;
  const [w, h] = referenceZone.dimensions;
  const mats = [];
  try {
    let marker = referenceImage.roi(new cv.Rect(x, y, w, h)).clone();
    mats.push(marker);

    if (markerDimensions != null) {
      const r = ImageUtils.resizeSingle(marker, markerDimensions[0], markerDimensions[1]);
      mats.splice(mats.indexOf(marker), 1); marker.delete(); marker = r; mats.push(marker);
    }

    const blurred = new cv.Mat(); mats.push(blurred);
    cv.GaussianBlur(marker, blurred, new cv.Size(blurKernel[0], blurKernel[1]), 0);
    mats.splice(mats.indexOf(marker), 1); marker.delete(); marker = blurred;

    const norm1 = new cv.Mat(); mats.push(norm1);
    cv.normalize(marker, norm1, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    mats.splice(mats.indexOf(marker), 1); marker.delete(); marker = norm1;

    if (applyErodeSubtract) {
      const kernel = cv.Mat.ones(5, 5, cv.CV_8U); mats.push(kernel);
      const eroded = new cv.Mat(); mats.push(eroded);
      cv.erode(marker, eroded, kernel, new cv.Point(-1, -1), 5);
      const sub = new cv.Mat(); mats.push(sub);
      cv.subtract(marker, eroded, sub);
      const norm2 = new cv.Mat(); mats.push(norm2);
      cv.normalize(sub, norm2, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
      mats.splice(mats.indexOf(marker), 1); marker.delete(); marker = norm2;
    }

    mats.splice(mats.indexOf(marker), 1);
    return marker;
  } finally {
    mats.forEach(m => { try { m.delete(); } catch (_) {} });
  }
}
window.prepareMarkerTemplate = prepareMarkerTemplate;

/** Port of detectMarkerInPatch from marker_detection.ts */
function detectMarkerInPatch(patch, marker, zoneOffset, scaleRange, scaleSteps, minConfidence) {
  const descentPerStep = Math.floor((scaleRange[1] - scaleRange[0]) / scaleSteps);
  const mH = marker.rows, mW = marker.cols;
  const pH = patch.rows,  pW = patch.cols;
  let bestPos = null, bestMarker = null, bestConf = 0.0;

  for (let sp = scaleRange[1]; sp > scaleRange[0]; sp -= descentPerStep) {
    const scale = sp / 100;
    if (scale <= 0) continue;
    const sw = Math.floor(mW * scale), sh = Math.floor(mH * scale);
    if (sh > pH || sw > pW || sw < 1 || sh < 1) continue;

    const sm = ImageUtils.resizeSingle(marker, sw, sh);
    const res = new cv.Mat();
    cv.matchTemplate(patch, sm, res, cv.TM_CCOEFF_NORMED);
    const loc = cv.minMaxLoc(res, new cv.Mat());
    res.delete();

    if (loc.maxVal > bestConf) {
      if (bestMarker) bestMarker.delete();
      bestConf = loc.maxVal;
      bestPos  = [loc.maxLoc.x, loc.maxLoc.y];
      bestMarker = sm;
    } else {
      sm.delete();
    }
  }

  if (bestPos === null || bestMarker === null) return null;
  if (bestConf < minConfidence) { bestMarker.delete(); return null; }

  const h2 = bestMarker.rows, w2 = bestMarker.cols;
  const [x, y] = bestPos;
  const corners = MathUtils.getRectanglePoints(x, y, w2, h2);
  const abs = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);
  bestMarker.delete();
  return abs;
}
window.detectMarkerInPatch = detectMarkerInPatch;

// ── CropOnMarkers ─────────────────────────────────────────────────────────────

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
  constructor(options, assetMats = {}) {
    super(options);
    if (options.type !== 'FOUR_MARKERS') throw new Error('Only FOUR_MARKERS is supported');
    this.referenceImageKey = options.reference_image;
    this.markerDimensions  = options.marker_dimensions || null;
    const t = options.tuning_options || {};
    this.minMatchingThreshold = t.min_matching_threshold != null ? t.min_matching_threshold : 0.3;
    this.markerRescaleRange   = t.marker_rescale_range  || [85, 115];
    this.markerRescaleSteps   = t.marker_rescale_steps  != null ? t.marker_rescale_steps : 5;
    this.applyErodeSubtract   = t.apply_erode_subtract  != null ? t.apply_erode_subtract : true;
    this.markerTemplates      = new Map();
    const refMat = assetMats[this.referenceImageKey];
    if (!refMat) throw new Error('Asset Mat not found for key: ' + this.referenceImageKey);
    this._initResizedMarkers(refMat);
  }

  /** Async factory: decodes base64 via canvas/Image, then constructs CropOnMarkers. */
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
      const corners  = this._findMarkerCornersInPatch(image, zoneDesc, marker, zoneType, filePath);
      const cx = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
      const cy = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
      allCorners.push([cx, cy]);
    }
    const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(allCorners);
    return [allCorners, warpedPoints, null];
  }

  _initResizedMarkers(refImage) {
    const zone = { origin: [1, 1], dimensions: [refImage.cols - 1, refImage.rows - 1] };
    for (const zoneType of FOUR_MARKERS_ZONE_ORDER) {
      const m = prepareMarkerTemplate(refImage, zone, this.markerDimensions, [5, 5], this.applyErodeSubtract);
      this.markerTemplates.set(zoneType, m);
    }
  }

  _getQuadrantZoneDescription(zoneType, image, marker) {
    const h = image.rows, w = image.cols;
    const halfH = Math.floor(h / 2), halfW = Math.floor(w / 2);
    const mH = marker.rows, mW = marker.cols;
    let zs, ze;
    if      (zoneType === 'topLeftMarker')      { zs = [1,     1];     ze = [halfW, halfH]; }
    else if (zoneType === 'topRightMarker')     { zs = [halfW, 1];     ze = [w,     halfH]; }
    else if (zoneType === 'bottomRightMarker')  { zs = [halfW, halfH]; ze = [w,     h];     }
    else                                        { zs = [1,     halfH]; ze = [halfW, h];     }
    const ox     = Math.floor((zs[0] + ze[0] - mW) / 2);
    const oy     = Math.floor((zs[1] + ze[1] - mH) / 2);
    const marg_h = (ze[0] - zs[0] - mW) / 2 - 1;
    const marg_v = (ze[1] - zs[1] - mH) / 2 - 1;
    return {
      origin:     [ox, oy],
      dimensions: [mW, mH],
      margins:    { top: marg_v, right: marg_h, bottom: marg_v, left: marg_h },
    };
  }

  _findMarkerCornersInPatch(image, zoneDesc, marker, zoneType, filePath) {
    const { origin: [ox, oy], dimensions: [dw, dh], margins } = zoneDesc;
    const mt = Math.max(0, Math.floor(margins.top));
    const mr = Math.max(0, Math.floor(margins.right));
    const mb = Math.max(0, Math.floor(margins.bottom));
    const ml = Math.max(0, Math.floor(margins.left));
    const px = Math.max(0, ox - ml);
    const py = Math.max(0, oy - mt);
    const pw = Math.min(image.cols - px, dw + ml + mr);
    const ph = Math.min(image.rows - py, dh + mt + mb);
    if (pw <= 0 || ph <= 0) throw new Error('Degenerate patch for zone: ' + zoneType);
    const patch = image.roi(new cv.Rect(px, py, pw, ph)).clone();
    let corners;
    try {
      corners = detectMarkerInPatch(
        patch, marker, [px, py],
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
      try { if (!m.isDeleted()) m.delete(); } catch (_) {}
    }
    this.markerTemplates.clear();
  }
}
window.CropOnMarkers = CropOnMarkers;

// ── __cropPage: lightweight CropPage helper ───────────────────────────────────
//
// Minimal CropPage implementation for use as a pre-processing step before
// CropOnMarkers. Avoids importing the full CropPage class.

window.__cropPage = function(gray) {
  const trunc1 = new cv.Mat();
  cv.threshold(gray, trunc1, 210, 255, cv.THRESH_TRUNC);
  const prepared = new cv.Mat();
  cv.normalize(trunc1, prepared, 0, 255, cv.NORM_MINMAX);
  trunc1.delete();

  const trunc2 = new cv.Mat();
  cv.threshold(prepared, trunc2, 200, 255, cv.THRESH_TRUNC);
  prepared.delete();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(15, 15));
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
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const h = new cv.Mat();
    cv.convexHull(c, h);
    hulls.push({ h, area: cv.contourArea(h) });
    c.delete();
  }
  contours.delete();
  hulls.sort((a, b) => b.area - a.area);
  hulls.slice(5).forEach(x => x.h.delete());
  const top = hulls.slice(0, 5);

  function angle(p1, p2, p0) {
    const dx1 = p1[0]-p0[0], dy1 = p1[1]-p0[1];
    const dx2 = p2[0]-p0[0], dy2 = p2[1]-p0[1];
    return (dx1*dx2 + dy1*dy2) / Math.sqrt((dx1**2+dy1**2) * (dx2**2+dy2**2) + 1e-10);
  }
  function checkMaxCosine(pts) {
    let mx = 0;
    for (let i = 2; i < 5; i++) mx = Math.max(mx, Math.abs(angle(pts[i%4], pts[i-2], pts[i-1])));
    return mx < 0.35;
  }

  let corners = null;
  for (const { h } of top) {
    if (cv.contourArea(h) < 80000) continue;
    const peri  = cv.arcLength(h, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(h, approx, 0.025 * peri, true);
    if (approx.rows === 4) {
      const pts = [];
      for (let i = 0; i < 4; i++) pts.push([approx.data32S[i*2], approx.data32S[i*2+1]]);
      approx.delete();
      if (checkMaxCosine(pts)) { const { rect } = MathUtils.orderFourPoints(pts); corners = rect; break; }
    } else { approx.delete(); }
  }
  top.forEach(x => x.h.delete());
  if (!corners) throw new Error('CropPage: could not detect page boundary');

  const [tl, tr, br, bl] = corners;
  const w = Math.max(Math.floor(MathUtils.distance(tr, tl)), Math.floor(MathUtils.distance(br, bl)));
  const h2 = Math.max(Math.floor(MathUtils.distance(tr, br)), Math.floor(MathUtils.distance(tl, bl)));
  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [tl[0],tl[1], tr[0],tr[1], br[0],br[1], bl[0],bl[1]]);
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [0,0, w-1,0, w-1,h2-1, 0,h2-1]);
  const M = cv.getPerspectiveTransform(srcPts, dstPts);
  srcPts.delete(); dstPts.delete();
  const warped = new cv.Mat();
  cv.warpPerspective(gray, warped, M, new cv.Size(w, h2));
  M.delete();
  return { warped, corners };
};

})();
`;
