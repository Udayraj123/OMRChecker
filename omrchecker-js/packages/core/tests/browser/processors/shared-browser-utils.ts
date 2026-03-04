/**
 * Shared browser-side JavaScript utilities for Playwright tests.
 *
 * SHARED_UTILS_SCRIPT: inject-once foundation used by CropPage.test.ts,
 * CropOnMarkers.test.ts, and e2e-omrchecker.test.ts via page.evaluate().
 *
 * Provides:
 *   window.WarpMethod               — warp strategy enum
 *   window.WARP_METHOD_FLAG_VALUES  — interpolation flag enum
 *   window.MathUtils                — geometry helpers
 *   window.ImageUtils               — cv.Mat helpers
 *   window.WarpStrategyFactory      — strategy factory
 *   window.WarpOnPointsCommon       — abstract base class
 *   window.__loadGray(b64)          — decode base64 JPEG → grayscale cv.Mat
 */

export const SHARED_UTILS_SCRIPT = `
(function() {

// ── Constants ─────────────────────────────────────────────────────────────────

const WarpMethod = {
  PERSPECTIVE_TRANSFORM: 'PERSPECTIVE_TRANSFORM',
  HOMOGRAPHY: 'HOMOGRAPHY',
  REMAP_GRIDDATA: 'REMAP_GRIDDATA',
  DOC_REFINE: 'DOC_REFINE',
};
window.WarpMethod = WarpMethod;

const WARP_METHOD_FLAG_VALUES = { INTER_LINEAR: 1, INTER_CUBIC: 2, INTER_NEAREST: 0 };
window.WARP_METHOD_FLAG_VALUES = WARP_METHOD_FLAG_VALUES;

// ── MathUtils ─────────────────────────────────────────────────────────────────

const MathUtils = {
  distance(p1, p2) {
    return Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
  },
  getBoundingBoxOfPoints(points) {
    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const minX = Math.min(...xs), minY = Math.min(...ys);
    const maxX = Math.max(...xs), maxY = Math.max(...ys);
    return {
      boundingBox: [[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY]],
      boxDimensions: [Math.floor(maxX - minX), Math.floor(maxY - minY)],
    };
  },
  shiftPointsFromOrigin(newOrigin, listOfPoints) {
    return listOfPoints.map(p => [newOrigin[0] + p[0], newOrigin[1] + p[1]]);
  },
  orderFourPoints(points) {
    const sums  = points.map(p => p[0] + p[1]);
    const diffs = points.map(p => p[1] - p[0]);
    const rect = [
      points[sums.indexOf(Math.min(...sums))],
      points[diffs.indexOf(Math.min(...diffs))],
      points[sums.indexOf(Math.max(...sums))],
      points[diffs.indexOf(Math.max(...diffs))],
    ];
    return { rect, orderedIndices: [] };
  },
  validateRect(points) {
    if (points.length !== 4) return false;
    let maxCos = 0;
    for (let i = 2; i < 5; i++) {
      const p1 = points[i % 4], p2 = points[(i - 2) % 4], p0 = points[(i - 1) % 4];
      const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
      const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
      const cos = Math.abs(
        (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2) + 1e-10)
      );
      if (cos > maxCos) maxCos = cos;
    }
    return maxCos < 0.35;
  },
  getRectanglePoints(x, y, w, h) {
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
  },
};
window.MathUtils = MathUtils;

// ── ImageUtils ────────────────────────────────────────────────────────────────

const ImageUtils = {
  resizeSingle(img, w, h) {
    const dst = new window.cv.Mat();
    window.cv.resize(img, dst, new window.cv.Size(Math.floor(w), Math.floor(h)));
    return dst;
  },
  normalizeSingle(image, alpha, beta, normType) {
    const cv = window.cv;
    alpha    = alpha    != null ? alpha    : 0;
    beta     = beta     != null ? beta     : 255;
    normType = normType != null ? normType : cv.NORM_MINMAX;
    if (!image || image.empty()) return image;
    const minMax = cv.minMaxLoc(image, new cv.Mat());
    if (minMax.maxVal === minMax.minVal) return image.clone();
    const normalized = new cv.Mat();
    cv.normalize(image, normalized, alpha, beta, normType);
    return normalized;
  },
  getCroppedWarpedRectanglePoints(orderedCorners) {
    const [tl, tr, br, bl] = orderedCorners;
    const maxWidth  = Math.max(
      Math.floor(MathUtils.distance(tr, tl)),
      Math.floor(MathUtils.distance(br, bl))
    );
    const maxHeight = Math.max(
      Math.floor(MathUtils.distance(tr, br)),
      Math.floor(MathUtils.distance(tl, bl))
    );
    return [
      [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
      [maxWidth, maxHeight],
    ];
  },
};
window.ImageUtils = ImageUtils;

// ── WarpStrategyFactory ───────────────────────────────────────────────────────

const WarpStrategyFactory = {
  create(methodName) {
    return {
      methodName,
      warpImage(image, coloredImage, ctrl, dest, dims) {
        const cv = window.cv;
        const [w, h] = dims;
        const n = ctrl.length;
        const dsize = new cv.Size(w, h);
        let M;
        if (methodName === 'PERSPECTIVE_TRANSFORM') {
          const cMat = cv.matFromArray(4, 1, cv.CV_32FC2, ctrl.flat());
          const dMat = cv.matFromArray(4, 1, cv.CV_32FC2, dest.flat());
          M = cv.getPerspectiveTransform(cMat, dMat);
          cMat.delete(); dMat.delete();
        } else {
          const cMat = cv.matFromArray(n, 1, cv.CV_32FC2, ctrl.flat());
          const dMat = cv.matFromArray(n, 1, cv.CV_32FC2, dest.flat());
          M = cv.findHomography(cMat, dMat, 0, 3.0);
          cMat.delete(); dMat.delete();
        }
        const warpedGray = new cv.Mat();
        cv.warpPerspective(image, warpedGray, M, dsize, cv.INTER_LINEAR);
        let warpedColored = null;
        if (coloredImage) {
          warpedColored = new cv.Mat();
          cv.warpPerspective(coloredImage, warpedColored, M, dsize, cv.INTER_LINEAR);
        }
        M.delete();
        return { warpedGray, warpedColored, warpedDebug: null };
      },
    };
  },
};
window.WarpStrategyFactory = WarpStrategyFactory;

// ── WarpOnPointsCommon (abstract base) ────────────────────────────────────────

class WarpOnPointsCommon {
  constructor(options = {}) {
    const parsed = this.validateAndRemapOptionsSchema(options);
    const tuningOptions = options.tuning_options != null ? options.tuning_options
      : (parsed.tuning_options != null ? parsed.tuning_options : {});
    this.enableCropping = parsed.enable_cropping != null ? parsed.enable_cropping
      : (options.enable_cropping != null ? options.enable_cropping : false);
    this.warpMethod = tuningOptions.warp_method != null ? tuningOptions.warp_method
      : (this.enableCropping ? WarpMethod.PERSPECTIVE_TRANSFORM : WarpMethod.HOMOGRAPHY);
    const flagKey = tuningOptions.warp_method_flag != null ? tuningOptions.warp_method_flag : 'INTER_LINEAR';
    this.warpMethodFlag = WARP_METHOD_FLAG_VALUES[flagKey] != null ? WARP_METHOD_FLAG_VALUES[flagKey] : 1;
    this.coloredOutputsEnabled = options.colored_outputs_enabled != null ? options.colored_outputs_enabled : false;
    this.warpStrategy = WarpStrategyFactory.create(this.warpMethod);
  }

  validateAndRemapOptionsSchema(_options) { throw new Error('Not implemented'); }
  prepareImageBeforeExtraction(_image) { throw new Error('Not implemented'); }
  extractControlDestinationPoints(_image, _colored, _filePath) { throw new Error('Not implemented'); }
  appendSaveImage(..._args) {}

  applyFilter(image, coloredImage, template, filePath) {
    const prepared = this.prepareImageBeforeExtraction(image);
    const [controlPts, destPts, edgeMap] = this.extractControlDestinationPoints(prepared, coloredImage, filePath);
    const [parsedCtrl, parsedDest, dims] = this._parseAndPreparePoints(prepared, controlPts, destPts);
    const [warpedImage, warpedColored] = this._applyWarpStrategy(image, coloredImage, parsedCtrl, parsedDest, dims, edgeMap);
    this.appendSaveImage('Warped Image', [4, 5, 6], warpedImage, warpedColored);
    return [warpedImage, warpedColored, template];
  }

  _parseAndPreparePoints(image, controlPoints, destinationPoints) {
    const seen = new Map();
    const uniqueCtrl = [], uniqueDest = [];
    for (let i = 0; i < controlPoints.length; i++) {
      const key = JSON.stringify(controlPoints[i]);
      if (!seen.has(key)) {
        seen.set(key, destinationPoints[i]);
        uniqueCtrl.push(controlPoints[i]);
        uniqueDest.push(destinationPoints[i]);
      }
    }
    const dims = this._calculateWarpedDimensions([image.cols, image.rows], uniqueDest);
    return [uniqueCtrl, uniqueDest, dims];
  }

  _calculateWarpedDimensions(defaultDims, destinationPoints) {
    if (!this.enableCropping) return defaultDims;
    const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(destinationPoints);
    const fromOrigin = [-boundingBox[0][0], -boundingBox[0][1]];
    const shifted = MathUtils.shiftPointsFromOrigin(fromOrigin, destinationPoints);
    for (let i = 0; i < destinationPoints.length; i++) {
      destinationPoints[i] = shifted[i];
    }
    return boxDimensions;
  }

  _applyWarpStrategy(image, coloredImage, controlPoints, destinationPoints, warpedDimensions, _edgeContoursMap) {
    const [ctrl, dest, dims] = this._preparePointsForStrategy(controlPoints, destinationPoints, warpedDimensions);
    const coloredInput = this.coloredOutputsEnabled ? coloredImage : null;
    const result = this.warpStrategy.warpImage(image, coloredInput, ctrl, dest, dims);
    return [result.warpedGray, result.warpedColored];
  }

  _preparePointsForStrategy(controlPoints, destinationPoints, warpedDimensions) {
    if (this.warpMethod !== WarpMethod.PERSPECTIVE_TRANSFORM) {
      return [controlPoints, destinationPoints, warpedDimensions];
    }
    if (controlPoints.length !== 4) {
      throw new Error('Expected 4 control points for perspective transform, found ' + controlPoints.length + '.');
    }
    const { rect: orderedCtrl } = MathUtils.orderFourPoints(controlPoints);
    const [newDest, newDims] = ImageUtils.getCroppedWarpedRectanglePoints(orderedCtrl);
    return [orderedCtrl, newDest, newDims];
  }
}
window.WarpOnPointsCommon = WarpOnPointsCommon;

// ── Image loader (canvas-based, async) ────────────────────────────────────────

window.__loadGray = async function(b64) {
  const img = new Image();
  img.src = 'data:image/jpeg;base64,' + b64;
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
  const canvas = document.createElement('canvas');
  canvas.width = img.width; canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const rgba = window.cv.matFromImageData(ctx.getImageData(0, 0, img.width, img.height));
  const gray = new window.cv.Mat();
  window.cv.cvtColor(rgba, gray, window.cv.COLOR_RGBA2GRAY);
  rgba.delete();
  return gray;
};

})();
`;
