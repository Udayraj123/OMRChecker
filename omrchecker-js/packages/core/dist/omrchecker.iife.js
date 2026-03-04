"use strict";
var OMRChecker = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/index.ts
  var index_exports = {};
  __export(index_exports, {
    APPROX_POLY_EPSILON_FACTOR: () => APPROX_POLY_EPSILON_FACTOR,
    AlignmentError: () => AlignmentError,
    AnswerKeyError: () => AnswerKeyError,
    BubbleDetectionError: () => BubbleDetectionError,
    BubbleField: () => BubbleField,
    BubbleReader: () => BubbleReader,
    BubblesScanBox: () => BubblesScanBox,
    CANNY_THRESHOLD_HIGH: () => CANNY_THRESHOLD_HIGH,
    CANNY_THRESHOLD_LOW: () => CANNY_THRESHOLD_LOW,
    CLR_BLACK: () => CLR_BLACK,
    CLR_DARK_BLUE: () => CLR_DARK_BLUE,
    CLR_DARK_GRAY: () => CLR_DARK_GRAY,
    CLR_DARK_GREEN: () => CLR_DARK_GREEN,
    CLR_DARK_RED: () => CLR_DARK_RED,
    CLR_GRAY: () => CLR_GRAY,
    CLR_GREEN: () => CLR_GREEN,
    CLR_LIGHT_GRAY: () => CLR_LIGHT_GRAY,
    CLR_NEAR_BLACK: () => CLR_NEAR_BLACK,
    CLR_WHITE: () => CLR_WHITE,
    CONFIG_FILENAME: () => CONFIG_FILENAME,
    CONTOUR_THICKNESS_STANDARD: () => CONTOUR_THICKNESS_STANDARD,
    CUSTOM_BUBBLE_FIELD_TYPE_PATTERN: () => CUSTOM_BUBBLE_FIELD_TYPE_PATTERN,
    ConfigError: () => ConfigError,
    ConfigLoadError: () => ConfigLoadError,
    CropOnMarkers: () => CropOnMarkers,
    CropPage: () => CropPage,
    DocRefineRectifyStrategy: () => DocRefineRectifyStrategy,
    DrawingUtils: () => DrawingUtils,
    EDGE_TYPES_IN_ORDER: () => EDGE_TYPES_IN_ORDER,
    ERROR_CODES: () => ERROR_CODES,
    EVALUATION_FILENAME: () => EVALUATION_FILENAME,
    EdgeType: () => EdgeType,
    EvaluationError: () => EvaluationError,
    FIELD_LABEL_NUMBER_REGEX: () => FIELD_LABEL_NUMBER_REGEX,
    FieldBlock: () => FieldBlock,
    FieldDefinitionError: () => FieldDefinitionError,
    FilePatternResolver: () => FilePatternResolver,
    GridDataRemapStrategy: () => GridDataRemapStrategy,
    HomographyStrategy: () => HomographyStrategy,
    ImageProcessingError: () => ImageProcessingError,
    ImageReadError: () => ImageReadError,
    ImageUtils: () => ImageUtils,
    InputError: () => InputError,
    InputFileNotFoundError: () => InputFileNotFoundError,
    InvalidConfigValueError: () => InvalidConfigValueError,
    Logger: () => Logger,
    MARKED_TEMPLATE_TRANSPARENCY: () => MARKED_TEMPLATE_TRANSPARENCY,
    MIN_PAGE_AREA: () => MIN_PAGE_AREA,
    MathUtils: () => MathUtils,
    NumberAggregate: () => NumberAggregate,
    OMRChecker: () => OMRChecker,
    OMRCheckerError: () => OMRCheckerError,
    OUTPUT_MODES: () => OUTPUT_MODES,
    PAPER_SATURATION_THRESHOLD: () => PAPER_SATURATION_THRESHOLD,
    PAPER_VALUE_THRESHOLD: () => PAPER_VALUE_THRESHOLD,
    PIXEL_VALUE_MAX: () => PIXEL_VALUE_MAX,
    PathUtils: () => PathUtils,
    PerspectiveTransformStrategy: () => PerspectiveTransformStrategy,
    PointParser: () => PointParser,
    PreprocessorError: () => PreprocessorError,
    ProcessingError: () => ProcessingError,
    SUPPORTED_PROCESSOR_NAMES: () => SUPPORTED_PROCESSOR_NAMES,
    ScanBox: () => ScanBox,
    ScannerType: () => ScannerType,
    SchemaValidationError: () => SchemaValidationError,
    ScoringError: () => ScoringError,
    StatsByLabel: () => StatsByLabel,
    TEMPLATE_FILENAME: () => TEMPLATE_FILENAME,
    TEXT_SIZE: () => TEXT_SIZE,
    THRESH_PAGE_TRUNCATE_HIGH: () => THRESH_PAGE_TRUNCATE_HIGH,
    THRESH_PAGE_TRUNCATE_SECONDARY: () => THRESH_PAGE_TRUNCATE_SECONDARY,
    TOP_CONTOURS_COUNT: () => TOP_CONTOURS_COUNT,
    Template: () => Template,
    TemplateError: () => TemplateError,
    ValidationError: () => ValidationError,
    WARP_METHOD_FLAG_VALUES: () => WARP_METHOD_FLAG_VALUES,
    WarpMethod: () => WarpMethod,
    WarpMethodFlags: () => WarpMethodFlags,
    WarpOnPointsCommon: () => WarpOnPointsCommon,
    WarpStrategy: () => WarpStrategy,
    WarpStrategyFactory: () => WarpStrategyFactory,
    WarpedDimensionsCalculator: () => WarpedDimensionsCalculator,
    ZonePreset: () => ZonePreset,
    appendCsvRow: () => appendCsvRow,
    applyColoredCanny: () => applyColoredCanny,
    applyGrayscaleCanny: () => applyGrayscaleCanny,
    bboxCenter: () => bboxCenter,
    calculateFileChecksum: () => calculateFileChecksum,
    computeBoundingBox: () => computeBoundingBox,
    computePointDistances: () => computePointDistances,
    createStructuringElement: () => createStructuringElement,
    deepSerialize: () => deepSerialize,
    detectContoursUsingCanny: () => detectContoursUsingCanny,
    detectDotCorners: () => detectDotCorners,
    detectLineCornersAndEdges: () => detectLineCornersAndEdges,
    detectMarkerInPatch: () => detectMarkerInPatch,
    euclideanDistance: () => euclideanDistance,
    extractMarkerCorners: () => extractMarkerCorners,
    extractPageRectangle: () => extractPageRectangle,
    extractPatchCornersAndEdges: () => extractPatchCornersAndEdges,
    findPageContourAndCorners: () => findPageContourAndCorners,
    findPageContours: () => findPageContours,
    formatCsvRow: () => formatCsvRow,
    loadJson: () => loadJson,
    logger: () => logger,
    multiScaleTemplateMatch: () => multiScaleTemplateMatch,
    orderFourPoints: () => orderFourPoints,
    parseFieldString: () => parseFieldString,
    parseFields: () => parseFields,
    parseJsonString: () => parseJsonString,
    prepareMarkerTemplate: () => prepareMarkerTemplate,
    preparePageImage: () => preparePageImage,
    preprocessDotZone: () => preprocessDotZone,
    preprocessLineZone: () => preprocessLineZone,
    printFileChecksum: () => printFileChecksum,
    validateBlurKernel: () => validateBlurKernel,
    validateMarkerDetection: () => validateMarkerDetection,
    vectorMagnitude: () => vectorMagnitude
  });

  // src/utils/constants.ts
  var TEMPLATE_FILENAME = "template.json";
  var EVALUATION_FILENAME = "evaluation.json";
  var CONFIG_FILENAME = "config.json";
  var SUPPORTED_PROCESSOR_NAMES = [
    "AutoRotate",
    "Contrast",
    "CropOnMarkers",
    "CropPage",
    "FeatureBasedAlignment",
    "GaussianBlur",
    "Levels",
    "MedianBlur"
    // "WarpOnPoints",
  ];
  var FIELD_LABEL_NUMBER_REGEX = /([^\d]+)(\d*)/;
  var ERROR_CODES = {
    MULTI_BUBBLE_WARN: 1,
    NO_MARKER_ERR: 2
  };
  var CUSTOM_BUBBLE_FIELD_TYPE_PATTERN = /^CUSTOM_.*$/;
  var TEXT_SIZE = 0.95;
  var CLR_BLACK = [0, 0, 0];
  var CLR_DARK_GRAY = [100, 100, 100];
  var CLR_DARK_BLUE = [255, 20, 20];
  var CLR_DARK_GREEN = [20, 255, 20];
  var CLR_DARK_RED = [20, 20, 255];
  var CLR_NEAR_BLACK = [20, 20, 10];
  var CLR_GRAY = [130, 130, 130];
  var CLR_LIGHT_GRAY = [200, 200, 200];
  var CLR_GREEN = [100, 200, 100];
  var CLR_WHITE = [255, 255, 255];
  var MARKED_TEMPLATE_TRANSPARENCY = 0.65;
  var PAPER_VALUE_THRESHOLD = 180;
  var PAPER_SATURATION_THRESHOLD = 40;
  var OUTPUT_MODES = {
    SET_LAYOUT: "setLayout",
    DEFAULT: "default",
    MODERATION: "moderation"
  };

  // src/utils/exceptions.ts
  var OMRCheckerError = class _OMRCheckerError extends Error {
    constructor(message, context = {}) {
      super(message);
      this.name = "OMRCheckerError";
      this.context = context;
      Object.setPrototypeOf(this, _OMRCheckerError.prototype);
    }
    toString() {
      if (Object.keys(this.context).length > 0) {
        const contextStr = Object.entries(this.context).map(([k, v]) => `${k}=${v}`).join(", ");
        return `${this.message} (${contextStr})`;
      }
      return this.message;
    }
  };
  var InputError = class _InputError extends OMRCheckerError {
    constructor(message, context = {}) {
      super(message, context);
      this.name = "InputError";
      Object.setPrototypeOf(this, _InputError.prototype);
    }
  };
  var ImageReadError = class _ImageReadError extends InputError {
    constructor(path, reason) {
      const msg = reason ? `Unable to read image: '${path}' - ${reason}` : `Unable to read image: '${path}'`;
      super(msg, { path, reason });
      this.name = "ImageReadError";
      this.path = path;
      this.reason = reason;
      Object.setPrototypeOf(this, _ImageReadError.prototype);
    }
  };
  var ValidationError = class _ValidationError extends OMRCheckerError {
    constructor(message, context = {}) {
      super(message, context);
      this.name = "ValidationError";
      Object.setPrototypeOf(this, _ValidationError.prototype);
    }
  };
  var SchemaValidationError = class _SchemaValidationError extends ValidationError {
    constructor(schemaName, errors, dataPath) {
      const msg = dataPath ? `Schema validation failed for '${schemaName}' at '${dataPath}'` : `Schema validation failed for '${schemaName}'`;
      super(msg, { schema: schemaName, errors, dataPath });
      this.name = "SchemaValidationError";
      this.schemaName = schemaName;
      this.errors = errors;
      this.dataPath = dataPath;
      Object.setPrototypeOf(this, _SchemaValidationError.prototype);
    }
  };
  var ProcessingError = class _ProcessingError extends OMRCheckerError {
    constructor(message, context = {}) {
      super(message, context);
      this.name = "ProcessingError";
      Object.setPrototypeOf(this, _ProcessingError.prototype);
    }
  };
  var ImageProcessingError = class _ImageProcessingError extends ProcessingError {
    constructor(message, context = {}) {
      super(message, context);
      this.name = "ImageProcessingError";
      this.operation = context.operation;
      this.filePath = context.filePath;
      this.reason = context.reason;
      Object.setPrototypeOf(this, _ImageProcessingError.prototype);
    }
  };
  var AlignmentError = class _AlignmentError extends ProcessingError {
    constructor(filePath, reason) {
      const msg = reason ? `Image alignment failed for '${filePath}': ${reason}` : `Image alignment failed for '${filePath}'`;
      super(msg, { filePath, reason });
      this.name = "AlignmentError";
      this.filePath = filePath;
      this.reason = reason;
      Object.setPrototypeOf(this, _AlignmentError.prototype);
    }
  };
  var BubbleDetectionError = class _BubbleDetectionError extends ProcessingError {
    constructor(filePath, fieldId, reason) {
      let msg = `Bubble detection failed for '${filePath}'`;
      if (fieldId) msg += ` at field '${fieldId}'`;
      if (reason) msg += `: ${reason}`;
      super(msg, { filePath, fieldId, reason });
      this.name = "BubbleDetectionError";
      this.filePath = filePath;
      this.fieldId = fieldId;
      this.reason = reason;
      Object.setPrototypeOf(this, _BubbleDetectionError.prototype);
    }
  };
  var TemplateError = class _TemplateError extends OMRCheckerError {
    constructor(message, context = {}) {
      super(message, context);
      this.name = "TemplateError";
      Object.setPrototypeOf(this, _TemplateError.prototype);
    }
  };
  var PreprocessorError = class _PreprocessorError extends TemplateError {
    constructor(preprocessorName, filePath, reason) {
      let msg = `Preprocessor '${preprocessorName}' failed`;
      if (filePath) msg += ` for '${filePath}'`;
      if (reason) msg += `: ${reason}`;
      super(msg, { preprocessor: preprocessorName, filePath, reason });
      this.name = "PreprocessorError";
      this.preprocessorName = preprocessorName;
      this.filePath = filePath;
      this.reason = reason;
      Object.setPrototypeOf(this, _PreprocessorError.prototype);
    }
  };
  var FieldDefinitionError = class _FieldDefinitionError extends TemplateError {
    constructor(fieldId, reason, templatePath) {
      const msg = templatePath ? `Invalid field definition '${fieldId}': ${reason} in '${templatePath}'` : `Invalid field definition '${fieldId}': ${reason}`;
      super(msg, { fieldId, reason, templatePath });
      this.name = "FieldDefinitionError";
      this.fieldId = fieldId;
      this.reason = reason;
      this.templatePath = templatePath;
      Object.setPrototypeOf(this, _FieldDefinitionError.prototype);
    }
  };
  var EvaluationError = class _EvaluationError extends OMRCheckerError {
    constructor(message, context = {}) {
      super(message, context);
      this.name = "EvaluationError";
      Object.setPrototypeOf(this, _EvaluationError.prototype);
    }
  };
  var AnswerKeyError = class _AnswerKeyError extends EvaluationError {
    constructor(reason, questionId) {
      const msg = questionId ? `Answer key error: ${reason} (question: ${questionId})` : `Answer key error: ${reason}`;
      super(msg, { reason, questionId });
      this.name = "AnswerKeyError";
      this.reason = reason;
      this.questionId = questionId;
      Object.setPrototypeOf(this, _AnswerKeyError.prototype);
    }
  };
  var ScoringError = class _ScoringError extends EvaluationError {
    constructor(reason, filePath, questionId) {
      let msg = `Scoring failed: ${reason}`;
      if (filePath) msg += ` for '${filePath}'`;
      if (questionId) msg += ` at question '${questionId}'`;
      super(msg, { reason, filePath, questionId });
      this.name = "ScoringError";
      this.reason = reason;
      this.filePath = filePath;
      this.questionId = questionId;
      Object.setPrototypeOf(this, _ScoringError.prototype);
    }
  };
  var ConfigError = class _ConfigError extends OMRCheckerError {
    constructor(message, context = {}) {
      super(message, context);
      this.name = "ConfigError";
      Object.setPrototypeOf(this, _ConfigError.prototype);
    }
  };
  var InvalidConfigValueError = class _InvalidConfigValueError extends ConfigError {
    constructor(key, value, reason) {
      super(`Invalid configuration value for '${key}': ${value} - ${reason}`, {
        key,
        value: String(value),
        reason
      });
      this.name = "InvalidConfigValueError";
      this.key = key;
      this.value = value;
      this.reason = reason;
      Object.setPrototypeOf(this, _InvalidConfigValueError.prototype);
    }
  };
  var ConfigLoadError = class _ConfigLoadError extends ConfigError {
    constructor(path, reason) {
      super(`Failed to load configuration '${path}': ${reason}`, {
        path,
        reason
      });
      this.name = "ConfigLoadError";
      this.path = path;
      this.reason = reason;
      Object.setPrototypeOf(this, _ConfigLoadError.prototype);
    }
  };
  var InputFileNotFoundError = class _InputFileNotFoundError extends InputError {
    constructor(path, fileType) {
      const fileDesc = fileType ? `${fileType} ` : "";
      super(`Input ${fileDesc}file not found: '${path}'`, {
        path,
        file_type: fileType
      });
      this.name = "InputFileNotFoundError";
      this.path = path;
      this.fileType = fileType;
      Object.setPrototypeOf(this, _InputFileNotFoundError.prototype);
    }
  };

  // src/utils/math.ts
  var EdgeType = /* @__PURE__ */ ((EdgeType3) => {
    EdgeType3["TOP"] = "TOP";
    EdgeType3["RIGHT"] = "RIGHT";
    EdgeType3["BOTTOM"] = "BOTTOM";
    EdgeType3["LEFT"] = "LEFT";
    return EdgeType3;
  })(EdgeType || {});
  var _MathUtils = class _MathUtils {
    /**
     * Calculate Euclidean distance between two points
     */
    static distance(point1, point2) {
      const dx = point1[0] - point2[0];
      const dy = point1[1] - point2[1];
      return Math.hypot(dx, dy);
    }
    /**
     * Shift points from origin by adding offset
     */
    static shiftPointsFromOrigin(newOrigin, listOfPoints) {
      return listOfPoints.map((point) => _MathUtils.addPoints(newOrigin, point));
    }
    /**
     * Add two points (vector addition)
     */
    static addPoints(newOrigin, point) {
      return [
        newOrigin[0] + point[0],
        newOrigin[1] + point[1]
      ];
    }
    /**
     * Subtract two points (vector subtraction)
     */
    static subtractPoints(point, newOrigin) {
      return [
        point[0] - newOrigin[0],
        point[1] - newOrigin[1]
      ];
    }
    /**
     * Shift points to origin by subtracting offset
     */
    static shiftPointsToOrigin(newOrigin, listOfPoints) {
      return listOfPoints.map((point) => _MathUtils.subtractPoints(point, newOrigin));
    }
    /**
     * Get point on line by length ratio (linear interpolation)
     */
    static getPointOnLineByRatio(edgeLine, lengthRatio) {
      const [start, end] = edgeLine;
      return [
        start[0] + (end[0] - start[0]) * lengthRatio,
        start[1] + (end[1] - start[1]) * lengthRatio
      ];
    }
    /**
     * Order four points in rectangle order: [tl, tr, br, bl]
     * Based on sum and diff heuristics
     */
    static orderFourPoints(points) {
      if (points.length !== 4) {
        throw new Error(`Expected 4 points, got ${points.length}`);
      }
      const sums = points.map((p) => p[0] + p[1]);
      const diffs = points.map((p) => p[1] - p[0]);
      const minSumIdx = sums.indexOf(Math.min(...sums));
      const minDiffIdx = diffs.indexOf(Math.min(...diffs));
      const maxSumIdx = sums.indexOf(Math.max(...sums));
      const maxDiffIdx = diffs.indexOf(Math.max(...diffs));
      const orderedIndices = [minSumIdx, minDiffIdx, maxSumIdx, maxDiffIdx];
      const rect = [
        points[minSumIdx],
        points[minDiffIdx],
        points[maxSumIdx],
        points[maxDiffIdx]
      ];
      return { rect, orderedIndices };
    }
    /**
     * Convert points to integer tuples
     */
    static getTuplePoints(points) {
      return points.map((point) => [
        Math.floor(point[0]),
        Math.floor(point[1])
      ]);
    }
    /**
     * Get bounding box of points as rectangle
     * Returns ordered rectangle [tl, tr, br, bl] and dimensions [width, height]
     */
    static getBoundingBoxOfPoints(points) {
      if (points.length === 0) {
        throw new Error("Cannot get bounding box of empty points array");
      }
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      const minX = Math.min(...xs);
      const minY = Math.min(...ys);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);
      const boundingBox = [
        [minX, minY],
        // top-left
        [maxX, minY],
        // top-right
        [maxX, maxY],
        // bottom-right
        [minX, maxY]
        // bottom-left
      ];
      const boxDimensions = [
        Math.floor(maxX - minX),
        Math.floor(maxY - minY)
      ];
      return { boundingBox, boxDimensions };
    }
    /**
     * Validate if 4 points form a valid rectangle
     */
    static validateRect(approx) {
      return approx.length === 4 && _MathUtils.checkMaxCosine(approx);
    }
    /**
     * Get rectangle points from origin and dimensions
     */
    static getRectanglePointsFromBox(origin, dimensions) {
      const [x, y] = origin;
      const [w, h] = dimensions;
      return _MathUtils.getRectanglePoints(x, y, w, h);
    }
    /**
     * Get rectangle points from x, y, width, height
     * Returns in order: [tl, tr, br, bl]
     */
    static getRectanglePoints(x, y, w, h) {
      return [
        [x, y],
        // top-left
        [x + w, y],
        // top-right
        [x + w, y + h],
        // bottom-right
        [x, y + h]
        // bottom-left
      ];
    }
    /**
     * Select edge from rectangle by type
     */
    static selectEdgeFromRectangle(rectangle, edgeType) {
      const [tl, tr, br, bl] = rectangle;
      switch (edgeType) {
        case "TOP" /* TOP */:
          return [tl, tr];
        case "RIGHT" /* RIGHT */:
          return [tr, br];
        case "BOTTOM" /* BOTTOM */:
          return [br, bl];
        case "LEFT" /* LEFT */:
          return [bl, tl];
        default:
          return [tl, tr];
      }
    }
    /**
     * Check if point is inside rectangle
     * Rectangle format: [[x1, y1], [x2, y2]] or [x1, y1, x2, y2]
     */
    static rectangleContains(point, rect) {
      let rectStart;
      let rectEnd;
      if (rect.length === 2 && Array.isArray(rect[0])) {
        [rectStart, rectEnd] = rect;
      } else if (rect.length === 4 && typeof rect[0] === "number") {
        const nums = rect;
        rectStart = [nums[0], nums[1]];
        rectEnd = [nums[2], nums[3]];
      } else {
        throw new Error("Invalid rectangle format");
      }
      return !(point[0] < rectStart[0] || point[1] < rectStart[1] || point[0] > rectEnd[0] || point[1] > rectEnd[1]);
    }
    /**
     * Check if quadrilateral is close to rectangle based on cosine of angles
     * Assumes 4 points
     */
    static checkMaxCosine(approx) {
      if (approx.length !== 4) {
        return false;
      }
      let maxCosine = 0;
      let minCosine = 1.5;
      for (let i = 2; i < 5; i++) {
        const cosine = Math.abs(
          _MathUtils.angle(approx[i % 4], approx[i - 2], approx[i - 1])
        );
        maxCosine = Math.max(cosine, maxCosine);
        minCosine = Math.min(cosine, minCosine);
      }
      if (maxCosine >= _MathUtils.MAX_COSINE) {
        console.warn("Quadrilateral is not a rectangle.");
        return false;
      }
      return true;
    }
    /**
     * Calculate cosine of angle between three points
     * Used for rectangle validation
     */
    static angle(p1, p2, p0) {
      const dx1 = p1[0] - p0[0];
      const dy1 = p1[1] - p0[1];
      const dx2 = p2[0] - p0[0];
      const dy2 = p2[1] - p0[1];
      const dotProduct = dx1 * dx2 + dy1 * dy2;
      const magnitude = Math.sqrt(
        (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
      );
      return dotProduct / magnitude;
    }
    /**
     * Check if three points are collinear
     */
    static checkCollinearPoints(point1, point2, point3) {
      const [x1, y1] = point1;
      const [x2, y2] = point2;
      const [x3, y3] = point3;
      return (y1 - y2) * (x1 - x3) === (y1 - y3) * (x1 - x2);
    }
    /**
     * Convert named color to BGR tuple (for OpenCV compatibility)
     * Browser version: returns RGB, OpenCV.js handles BGR internally
     */
    static toBgr(anyColor) {
      const ctx = document.createElement("canvas").getContext("2d");
      if (!ctx) {
        throw new Error("Cannot create canvas context for color conversion");
      }
      ctx.fillStyle = anyColor;
      const computed = ctx.fillStyle;
      const hex = computed.replace("#", "");
      const r = parseInt(hex.substr(0, 2), 16);
      const g = parseInt(hex.substr(2, 2), 16);
      const b = parseInt(hex.substr(4, 2), 16);
      return [b, g, r];
    }
    /**
     * Split array into chunks of specified size
     */
    static chunks(inputList, chunkSize) {
      const size = Math.max(1, chunkSize);
      const result = [];
      for (let i = 0; i < inputList.length; i += size) {
        result.push(inputList.slice(i, i + size));
      }
      return result;
    }
  };
  /** Maximum cosine value for rectangle validation */
  _MathUtils.MAX_COSINE = 0.35;
  var MathUtils = _MathUtils;

  // src/utils/stats.ts
  var StatsByLabel = class {
    constructor(...labels) {
      this.labelCounts = {};
      for (const label of labels) {
        this.labelCounts[label] = 0;
      }
    }
    /**
     * Increment count for a label
     * 
     * @param label - Label to increment
     * @param number - Amount to increment by (default: 1)
     * @throws Error if label is not in allowed labels
     */
    push(label, number = 1) {
      if (!(label in this.labelCounts)) {
        const allowedLabels = Object.keys(this.labelCounts);
        throw new Error(
          `Unknown label passed to stats by label: ${label}, allowed labels: ${allowedLabels.join(", ")}`
        );
      }
      this.labelCounts[label] += number;
    }
    /**
     * Get label counts
     */
    getLabelCounts() {
      return { ...this.labelCounts };
    }
    /**
     * Convert to JSON-serializable object
     */
    toJSON() {
      return {
        label_counts: this.labelCounts
      };
    }
    /**
     * Convert to string representation
     */
    toString() {
      return JSON.stringify(this.toJSON());
    }
  };
  var NumberAggregate = class {
    constructor() {
      this.collection = [];
      this.runningSum = 0;
      this.runningAverage = 0;
    }
    /**
     * Add a number to the aggregate
     * 
     * @param numberLike - Number to add
     * @param label - Label associated with this number
     */
    push(numberLike, label) {
      this.collection.push([numberLike, label]);
      this.runningSum += numberLike;
      this.runningAverage = this.runningSum / this.collection.length;
    }
    /**
     * Merge another aggregate into this one
     * 
     * @param otherAggregate - Another NumberAggregate to merge
     */
    merge(otherAggregate) {
      this.collection.push(...otherAggregate.collection);
      this.runningSum += otherAggregate.runningSum;
      this.runningAverage = this.runningSum / this.collection.length;
    }
    /**
     * Get collection of all values
     */
    getCollection() {
      return [...this.collection];
    }
    /**
     * Get running sum
     */
    getRunningSum() {
      return this.runningSum;
    }
    /**
     * Get running average
     */
    getRunningAverage() {
      return this.runningAverage;
    }
    /**
     * Convert to JSON-serializable object
     */
    toJSON() {
      return {
        collection: this.collection,
        running_sum: this.runningSum,
        running_average: this.runningAverage
      };
    }
    /**
     * Convert to string representation
     */
    toString() {
      return JSON.stringify(this.toJSON());
    }
  };

  // src/utils/geometry.ts
  function euclideanDistance(point1, point2) {
    const dx = point1[0] - point2[0];
    const dy = point1[1] - point2[1];
    return Math.sqrt(dx * dx + dy * dy);
  }
  function vectorMagnitude(vector) {
    return Math.sqrt(vector.reduce((sum, x) => sum + x * x, 0));
  }
  function bboxCenter(origin, dimensions) {
    return [
      origin[0] + dimensions[0] / 2,
      origin[1] + dimensions[1] / 2
    ];
  }

  // src/utils/checksum.ts
  async function calculateFileChecksum(fileData, algorithm = "SHA-256") {
    const buffer = fileData instanceof Blob ? await fileData.arrayBuffer() : fileData;
    try {
      const hashBuffer = await crypto.subtle.digest(algorithm, buffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
      return hashHex;
    } catch (error) {
      const message = `Unsupported hash algorithm: ${algorithm}`;
      throw new Error(message, { cause: error });
    }
  }
  async function printFileChecksum(fileData, algorithm = "SHA-256") {
    try {
      const checksum = await calculateFileChecksum(fileData, algorithm);
      console.log(`${algorithm}: ${checksum}`);
    } catch (error) {
    }
  }

  // src/cv-shim.ts
  var cv_shim_default = globalThis.cv;

  // src/utils/image.ts
  var ImageUtils = class _ImageUtils {
    /**
     * Compute warped rectangle points for a perspective-cropped region.
     *
     * Given four ordered corner points [tl, tr, br, bl], computes the destination
     * rectangle dimensions that preserve the maximum width and height, and returns
     * the axis-aligned destination points starting from the origin.
     *
     * Port of Python ImageUtils.get_cropped_warped_rectangle_points.
     *
     * @param orderedCorners - Four points [tl, tr, br, bl] as [x, y] arrays
     * @returns Tuple of [warpedPoints, [maxWidth, maxHeight]]
     */
    static getCroppedWarpedRectanglePoints(orderedCorners) {
      const [tl, tr, br, bl] = orderedCorners;
      const maxWidth = Math.max(
        Math.floor(MathUtils.distance(tr, tl)),
        Math.floor(MathUtils.distance(br, bl))
      );
      const maxHeight = Math.max(
        Math.floor(MathUtils.distance(tr, br)),
        Math.floor(MathUtils.distance(tl, bl))
      );
      const warpedPoints = [
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
      ];
      return [warpedPoints, [maxWidth, maxHeight]];
    }
    /**
     * Resize a single image with optional aspect ratio preservation.
     * 
     * @param image - Input image Mat
     * @param width - Target width (if null, calculated from height)
     * @param height - Target height (if null, calculated from width)
     * @returns Resized image Mat (caller must delete)
     */
    static resizeSingle(image, width, height) {
      if (!image || image.empty()) {
        throw new Error("Cannot resize empty or null image");
      }
      const h = image.rows;
      const w = image.cols;
      let targetWidth = width != null ? width : null;
      let targetHeight = height != null ? height : null;
      if (targetHeight === null && targetWidth !== null) {
        targetHeight = Math.floor(h * targetWidth / w);
      }
      if (targetWidth === null && targetHeight !== null) {
        targetWidth = Math.floor(w * targetHeight / h);
      }
      if (targetWidth === null || targetHeight === null) {
        throw new Error("Must provide at least one dimension for resize");
      }
      if (targetHeight === h && targetWidth === w) {
        return image.clone();
      }
      const resized = new cv_shim_default.Mat();
      cv_shim_default.resize(
        image,
        resized,
        new cv_shim_default.Size(Math.floor(targetWidth), Math.floor(targetHeight)),
        0,
        0,
        cv_shim_default.INTER_LINEAR
      );
      return resized;
    }
    /**
     * Resize multiple images to the same dimensions.
     * 
     * @param images - Array of image Mats
     * @param width - Target width
     * @param height - Target height
     * @returns Array of resized Mats (caller must delete each)
     */
    static resizeMultiple(images, width, height) {
      if (images.length === 0) {
        return [];
      }
      return images.map((image) => {
        if (!image || image.empty()) {
          return image;
        }
        return _ImageUtils.resizeSingle(image, width, height);
      });
    }
    /**
     * Resize image(s) to match a target shape (height, width).
     * 
     * @param imageShape - Target shape as [height, width]
     * @param images - One or more images to resize
     * @returns Resized image(s)
     */
    static resizeToShape(imageShape, ...images) {
      const [h, w] = imageShape;
      const resized = _ImageUtils.resizeMultiple(images, w, h);
      return images.length === 1 ? resized[0] : resized;
    }
    /**
     * Resize image(s) to match target dimensions (width, height).
     * 
     * @param imageDimensions - Target dimensions as [width, height]
     * @param images - One or more images to resize
     * @returns Resized image(s)
     */
    static resizeToDimensions(imageDimensions, ...images) {
      const [w, h] = imageDimensions;
      const resized = _ImageUtils.resizeMultiple(images, w, h);
      return images.length === 1 ? resized[0] : resized;
    }
    /**
     * Normalize a single image to a specific range.
     * 
     * @param image - Input image
     * @param alpha - Lower bound of normalization range (default: 0)
     * @param beta - Upper bound of normalization range (default: 255)
     * @param normType - Normalization type (default: NORM_MINMAX)
     * @returns Normalized image (caller must delete)
     */
    static normalizeSingle(image, alpha = 0, beta = 255, normType = cv_shim_default.NORM_MINMAX) {
      if (!image || image.empty()) {
        return image;
      }
      const minMax = cv_shim_default.minMaxLoc(image, new cv_shim_default.Mat());
      if (minMax.maxVal === minMax.minVal) {
        return image.clone();
      }
      const normalized = new cv_shim_default.Mat();
      cv_shim_default.normalize(image, normalized, alpha, beta, normType);
      return normalized;
    }
    /**
     * Normalize multiple images.
     * 
     * @param images - Images to normalize
     * @param alpha - Lower bound
     * @param beta - Upper bound  
     * @param normType - Normalization type
     * @returns Normalized images
     */
    static normalize(images, alpha = 0, beta = 255, normType = cv_shim_default.NORM_MINMAX) {
      if (images.length === 0) {
        throw new Error("Must provide at least one image to normalize");
      }
      const normalized = images.map(
        (image) => _ImageUtils.normalizeSingle(image, alpha, beta, normType)
      );
      return images.length === 1 ? normalized[0] : normalized;
    }
    /**
     * Extract contours array from OpenCV findContours result.
     * 
     * OpenCV.js returns contours differently than Python OpenCV.
     * This helper provides compatibility.
     * 
     * @param contoursResult - Result from cv.findContours
     * @returns cv.MatVector of contours
     */
    static grabContours(contoursResult) {
      if (contoursResult instanceof cv_shim_default.MatVector) {
        return contoursResult;
      }
      if (contoursResult && contoursResult.contours) {
        return contoursResult.contours;
      }
      throw new Error("Invalid contours format from OpenCV.js");
    }
    /**
     * Automatic Canny edge detection using computed thresholds.
     * 
     * Calculates optimal thresholds based on image median intensity.
     * 
     * @param image - Input grayscale image
     * @param sigma - Threshold multiplier (default: 0.93)
     * @returns Edge detected image (caller must delete)
     */
    static autoCanny(image, sigma = 0.93) {
      if (!image || image.empty()) {
        throw new Error("Cannot apply Canny to empty image");
      }
      const mean = cv_shim_default.mean(image);
      const v = mean[0];
      const lower = Math.max(0, Math.floor((1 - sigma) * v));
      const upper = Math.min(255, Math.floor((1 + sigma) * v));
      const edges = new cv_shim_default.Mat();
      cv_shim_default.Canny(image, edges, lower, upper);
      return edges;
    }
    /**
     * Rotate image while optionally keeping original shape.
     * 
     * @param image - Input image
     * @param rotationCode - OpenCV rotation code (ROTATE_90_CLOCKWISE, etc.)
     * @param keepOriginalShape - If true, resize back to original dimensions
     * @returns Rotated image (caller must delete)
     */
    static rotate(image, rotationCode, keepOriginalShape = false) {
      if (!image || image.empty()) {
        throw new Error("Cannot rotate empty image");
      }
      const originalHeight = image.rows;
      const originalWidth = image.cols;
      const rotated = new cv_shim_default.Mat();
      cv_shim_default.rotate(image, rotated, rotationCode);
      if (keepOriginalShape) {
        if (rotated.rows !== originalHeight || rotated.cols !== originalWidth) {
          const resized = new cv_shim_default.Mat();
          cv_shim_default.resize(
            rotated,
            resized,
            new cv_shim_default.Size(originalWidth, originalHeight),
            0,
            0,
            cv_shim_default.INTER_LINEAR
          );
          rotated.delete();
          return resized;
        }
      }
      return rotated;
    }
    /**
     * Pad an image symmetrically from the center with a constant border value.
     *
     * Port of Python ImageUtils.pad_image_from_center.
     *
     * @param image - Input image Mat
     * @param paddingWidth - Number of pixels to pad on each side horizontally
     * @param paddingHeight - Number of pixels to pad on each side vertically (default: 0)
     * @param value - Border fill value (default: 255 = white)
     * @returns Object containing paddedImage Mat and padRange [top, bottom, left, right]
     *          padRange indices into the padded image where the original image was placed.
     *          The caller is responsible for deleting the returned paddedImage.
     */
    static padImageFromCenter(image, paddingWidth, paddingHeight = 0, value = 255) {
      const padRange = [
        paddingHeight,
        paddingHeight + image.rows,
        paddingWidth,
        paddingWidth + image.cols
      ];
      const paddedImage = new cv_shim_default.Mat();
      cv_shim_default.copyMakeBorder(
        image,
        paddedImage,
        paddingHeight,
        paddingHeight,
        paddingWidth,
        paddingWidth,
        cv_shim_default.BORDER_CONSTANT,
        new cv_shim_default.Scalar(value, value, value, value)
      );
      return { paddedImage, padRange };
    }
    /**
     * Apply gamma correction to an image.
     *
     * Builds a lookup table mapping each pixel intensity i → ((i/255)^(1/gamma))*255,
     * then applies it via cv.LUT.
     *
     * Port of Python ImageUtils.adjust_gamma.
     *
     * @param image - Input grayscale image Mat (CV_8UC1)
     * @param gamma - Gamma value (default: 1.0, >1 brightens, <1 darkens)
     * @returns Gamma-corrected image Mat (caller must delete)
     */
    static adjustGamma(image, gamma = 1) {
      const invGamma = 1 / gamma;
      const tableData = new Uint8Array(256);
      for (let i = 0; i < 256; i++) {
        tableData[i] = Math.round(Math.pow(i / 255, invGamma) * 255);
      }
      const lutMat = cv_shim_default.matFromArray(1, 256, cv_shim_default.CV_8UC1, Array.from(tableData));
      const result = new cv_shim_default.Mat();
      cv_shim_default.LUT(image, lutMat, result);
      lutMat.delete();
      return result;
    }
    /**
     * Split a contour's boundary points into four edge groups based on proximity
     * to each side of a quadrilateral defined by four corner points.
     *
     * The patch corners are first ordered [tl, tr, br, bl] via MathUtils.orderFourPoints.
     * Each source contour point is assigned to the nearest edge (TOP, RIGHT, BOTTOM, LEFT)
     * using point-to-segment distance. Corner points are inserted at the start and end of
     * each edge's list, and the list is reversed if needed to maintain clockwise order.
     *
     * Port of Python ImageUtils.split_patch_contour_on_corners.
     *
     * @param patchCorners - Four [x, y] corner points of the bounding patch
     * @param sourceContour - Array of [x, y] points from the detected contour boundary
     * @returns Object with orderedCorners [tl, tr, br, bl] and edgeContoursMap
     *          keyed by 'TOP' | 'RIGHT' | 'BOTTOM' | 'LEFT'
     */
    static splitPatchContourOnCorners(patchCorners, sourceContour) {
      const { rect } = MathUtils.orderFourPoints(patchCorners);
      const orderedCorners = Array.from(rect);
      const edgeTypes = ["TOP", "RIGHT", "BOTTOM", "LEFT"];
      const edgeContoursMap = {
        TOP: [],
        RIGHT: [],
        BOTTOM: [],
        LEFT: []
      };
      function distToSegment(px, py, ax, ay, bx, by) {
        const dx = bx - ax;
        const dy = by - ay;
        const lenSq = dx * dx + dy * dy;
        if (lenSq === 0) return Math.hypot(px - ax, py - ay);
        let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
      }
      for (const pt of sourceContour) {
        let minDist = Infinity;
        let nearestEdge = "TOP";
        for (let i = 0; i < 4; i++) {
          const [ax, ay] = orderedCorners[i];
          const [bx, by] = orderedCorners[(i + 1) % 4];
          const d = distToSegment(pt[0], pt[1], ax, ay, bx, by);
          if (d < minDist) {
            minDist = d;
            nearestEdge = edgeTypes[i];
          }
        }
        edgeContoursMap[nearestEdge].push(pt);
      }
      for (let i = 0; i < 4; i++) {
        const edgeType = edgeTypes[i];
        const startPt = orderedCorners[i];
        const endPt = orderedCorners[(i + 1) % 4];
        const edgeContour = edgeContoursMap[edgeType];
        if (edgeContour.length > 0) {
          const distToFirst = MathUtils.distance(startPt, edgeContour[0]);
          const distToLast = MathUtils.distance(startPt, edgeContour[edgeContour.length - 1]);
          if (distToLast < distToFirst) {
            edgeContour.reverse();
          }
        }
        edgeContoursMap[edgeType].unshift(startPt);
        edgeContoursMap[edgeType].push(endPt);
      }
      return { orderedCorners, edgeContoursMap };
    }
    /**
     * Overlay two images with transparency.
     *
     * @param image1 - First image
     * @param image2 - Second image (must be same size as image1)
     * @param transparency - Alpha value for image1 (0-1, default: 0.5)
     * @returns Blended image (caller must delete)
     */
    static overlayImage(image1, image2, transparency = 0.5) {
      if (!image1 || !image2 || image1.empty() || image2.empty()) {
        throw new Error("Cannot overlay empty images");
      }
      if (image1.rows !== image2.rows || image1.cols !== image2.cols || image1.type() !== image2.type()) {
        throw new Error("Images must have same dimensions and type for overlay");
      }
      const overlay = new cv_shim_default.Mat();
      cv_shim_default.addWeighted(
        image1,
        transparency,
        image2,
        1 - transparency,
        0,
        overlay
      );
      return overlay;
    }
  };

  // src/utils/drawing.ts
  var DrawingUtils = class {
    static drawMatches(image, fromPoints, warpedImage, toPoints) {
      const result = new cv_shim_default.Mat();
      const imageMats = new cv_shim_default.MatVector();
      imageMats.push_back(image);
      imageMats.push_back(warpedImage);
      cv_shim_default.hconcat(imageMats, result);
      imageMats.delete();
      const w = image.cols;
      const fromTuples = MathUtils.getTuplePoints(fromPoints);
      const toTuples = MathUtils.getTuplePoints(toPoints);
      for (let i = 0; i < Math.min(fromTuples.length, toTuples.length); i++) {
        const fromPoint = new cv_shim_default.Point(fromTuples[i][0], fromTuples[i][1]);
        const toPoint = new cv_shim_default.Point(w + toTuples[i][0], toTuples[i][1]);
        cv_shim_default.line(result, fromPoint, toPoint, CLR_GREEN, 3);
      }
      return result;
    }
    static drawBoxDiagonal(image, position, positionDiagonal, color = CLR_DARK_GRAY, border = 3) {
      const pt1 = new cv_shim_default.Point(Math.floor(position[0]), Math.floor(position[1]));
      const pt2 = new cv_shim_default.Point(Math.floor(positionDiagonal[0]), Math.floor(positionDiagonal[1]));
      cv_shim_default.rectangle(image, pt1, pt2, color, border);
    }
    static drawContour(image, contour, color = CLR_GREEN, thickness = 2) {
      if (contour.some((pt) => pt === null || pt === void 0)) {
        throw new ImageProcessingError("Invalid contour provided", {
          contour: JSON.stringify(contour)
        });
      }
      const contourMat = cv_shim_default.matFromArray(contour.length, 1, cv_shim_default.CV_32SC2, contour.flat());
      const contours = new cv_shim_default.MatVector();
      contours.push_back(contourMat);
      cv_shim_default.drawContours(image, contours, -1, color, thickness);
      contours.delete();
      contourMat.delete();
    }
    static drawBox(image, position, boxDimensions, color, style = "BOX_HOLLOW", thicknessFactor = 1 / 12, border = 3, centered = false) {
      const [x, y] = position;
      const [boxW, boxH] = boxDimensions;
      let pos = [
        Math.floor(x + boxW * thicknessFactor),
        Math.floor(y + boxH * thicknessFactor)
      ];
      let posDiag = [
        Math.floor(x + boxW - boxW * thicknessFactor),
        Math.floor(y + boxH - boxH * thicknessFactor)
      ];
      if (centered) {
        const centeredPosition = [
          Math.floor((3 * pos[0] - posDiag[0]) / 2),
          Math.floor((3 * pos[1] - posDiag[1]) / 2)
        ];
        const centeredDiagonal = [
          Math.floor((pos[0] + posDiag[0]) / 2),
          Math.floor((pos[1] + posDiag[1]) / 2)
        ];
        pos = centeredPosition;
        posDiag = centeredDiagonal;
      }
      let finalColor = color;
      let finalBorder = border;
      if (style === "BOX_HOLLOW") {
        finalColor = color != null ? color : CLR_GRAY;
      } else if (style === "BOX_FILLED") {
        finalColor = color != null ? color : CLR_DARK_GRAY;
        finalBorder = -1;
      }
      this.drawBoxDiagonal(image, pos, posDiag, finalColor, finalBorder);
      return { position: pos, positionDiagonal: posDiag };
    }
    static drawArrows(image, startPoints, endPoints, color = CLR_GREEN, thickness = 2, lineType = cv_shim_default.LINE_AA, tipLength = 0.1) {
      const startTuples = MathUtils.getTuplePoints(startPoints);
      const endTuples = MathUtils.getTuplePoints(endPoints);
      for (let i = 0; i < Math.min(startTuples.length, endTuples.length); i++) {
        const start = new cv_shim_default.Point(startTuples[i][0], startTuples[i][1]);
        const end = new cv_shim_default.Point(endTuples[i][0], endTuples[i][1]);
        cv_shim_default.arrowedLine(image, start, end, color, thickness, lineType, 0, tipLength);
      }
      return image;
    }
    static drawTextResponsive(image, text, position, textSize = TEXT_SIZE, thickness = 2, centered = false, color = CLR_BLACK, lineType = cv_shim_default.LINE_AA, fontFace = cv_shim_default.FONT_HERSHEY_SIMPLEX) {
      const h = image.rows;
      const w = image.cols;
      const textPosition = (sizeX, sizeY) => [
        position[0] - Math.max(0, position[0] + sizeX - w),
        position[1] - Math.max(0, position[1] + sizeY - h)
      ];
      this.drawText(image, text, textPosition, textSize, thickness, centered, color, lineType, fontFace);
    }
    static drawText(image, textValue, position, textSize = TEXT_SIZE, thickness = 2, centered = false, color = CLR_BLACK, lineType = cv_shim_default.LINE_AA, fontFace = cv_shim_default.FONT_HERSHEY_SIMPLEX) {
      let finalPosition = position;
      if (centered) {
        if (typeof position === "function") {
          throw new ImageProcessingError(`centered=${centered} but position is a callable`, {
            centered,
            position: position.toString()
          });
        }
        const textPosition = position;
        finalPosition = (sizeX, sizeY) => [
          textPosition[0] - Math.floor(sizeX / 2),
          textPosition[1] + Math.floor(sizeY / 2)
        ];
      }
      if (typeof finalPosition === "function") {
        const textSizeResult = cv_shim_default.getTextSize(textValue, fontFace, textSize, thickness);
        finalPosition = finalPosition(textSizeResult.size.width, textSizeResult.size.height);
      }
      const pt = new cv_shim_default.Point(Math.floor(finalPosition[0]), Math.floor(finalPosition[1]));
      cv_shim_default.putText(image, textValue, pt, fontFace, textSize, color, thickness, lineType);
    }
    static drawSymbol(image, symbol, position, positionDiagonal, color = CLR_BLACK) {
      const centerPosition = (sizeX, sizeY) => [
        Math.floor((position[0] + positionDiagonal[0] - sizeX) / 2),
        Math.floor((position[1] + positionDiagonal[1] + sizeY) / 2)
      ];
      this.drawText(image, symbol, centerPosition, TEXT_SIZE, 2, false, color);
    }
    static drawLine(image, start, end, color = CLR_BLACK, thickness = 3) {
      const pt1 = new cv_shim_default.Point(start[0], start[1]);
      const pt2 = new cv_shim_default.Point(end[0], end[1]);
      cv_shim_default.line(image, pt1, pt2, color, thickness);
    }
    static drawPolygon(image, points, color = CLR_BLACK, thickness = 1, closed = true) {
      const n = points.length;
      for (let i = 0; i < n; i++) {
        if (!closed && i === n - 1) {
          continue;
        }
        this.drawLine(image, points[i % n], points[(i + 1) % n], color, thickness);
      }
    }
    static drawGroup(image, start, bubbleDimensions, boxEdge, color, thickness = 3, thicknessFactor = 7 / 10) {
      const [startX, startY] = start;
      const [boxW, boxH] = bubbleDimensions;
      let startPos;
      let endPos;
      if (boxEdge === "TOP") {
        endPos = [startX + Math.floor(boxW * thicknessFactor), startY];
        startPos = [startX + Math.floor(boxW * (1 - thicknessFactor)), startY];
        this.drawLine(image, startPos, endPos, color, thickness);
      } else if (boxEdge === "RIGHT") {
        startPos = [startX + boxW, startY];
        endPos = [startX, Math.floor(startY + boxH * thicknessFactor)];
        startPos = [startX, Math.floor(startY + boxH * (1 - thicknessFactor))];
        this.drawLine(image, startPos, endPos, color, thickness);
      } else if (boxEdge === "BOTTOM") {
        startPos = [startX, startY + boxH];
        endPos = [Math.floor(startX + boxW * thicknessFactor), startY];
        startPos = [Math.floor(startX + boxW * (1 - thicknessFactor)), startY];
        this.drawLine(image, startPos, endPos, color, thickness);
      } else if (boxEdge === "LEFT") {
        endPos = [startX, Math.floor(startY + boxH * thicknessFactor)];
        startPos = [startX, Math.floor(startY + boxH * (1 - thicknessFactor))];
        this.drawLine(image, startPos, endPos, color, thickness);
      }
    }
  };

  // src/utils/serialization.ts
  function deepSerialize(obj) {
    if (obj === null || obj === void 0) return obj;
    if (typeof obj === "string" || typeof obj === "number" || typeof obj === "boolean") return obj;
    if (Array.isArray(obj)) return obj.map(deepSerialize);
    if (obj instanceof Map) {
      return Object.fromEntries([...obj.entries()].map(([k, v]) => [k, deepSerialize(v)]));
    }
    if (typeof obj === "object") {
      return Object.fromEntries(
        Object.entries(obj).map(([k, v]) => [k, deepSerialize(v)])
      );
    }
    try {
      return String(obj);
    } catch {
      return obj;
    }
  }

  // src/utils/logger.ts
  var DEFAULT_LOG_LEVEL_MAP = {
    critical: true,
    error: true,
    warning: true,
    info: true,
    debug: true
  };
  var Logger = class {
    constructor(name) {
      this.name = name;
      this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP };
    }
    setLogLevels(levels) {
      this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP, ...levels };
    }
    resetLogLevels() {
      this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP };
    }
    logutil(methodType, ...msg) {
      if (this.showLogsByType[methodType] === false) return;
      const str = msg.map((v) => typeof v === "string" ? v : String(v)).join(" ");
      if (methodType === "critical" || methodType === "error") {
        console.error(str);
      } else if (methodType === "warning") {
        console.warn(str);
      } else if (methodType === "debug") {
        console.debug(str);
      } else {
        console.log(str);
      }
    }
    debug(...msg) {
      this.logutil("debug", ...msg);
    }
    info(...msg) {
      this.logutil("info", ...msg);
    }
    warning(...msg) {
      this.logutil("warning", ...msg);
    }
    error(...msg) {
      this.logutil("error", ...msg);
    }
    critical(...msg) {
      this.logutil("critical", ...msg);
    }
  };
  var logger = new Logger("omrchecker");

  // src/utils/csv.ts
  function formatCsvRow(dataLine) {
    if (dataLine.length === 0) return "";
    return dataLine.map((v) => {
      if (/^-?\d+(\.\d+)?$/.test(v)) return v;
      return `"${v.replace(/"/g, '""')}"`;
    }).join(",");
  }
  function appendCsvRow(rows, dataLine) {
    rows.push(formatCsvRow(dataLine));
  }

  // src/utils/file.ts
  async function loadJson(path) {
    let response;
    try {
      response = await fetch(path);
    } catch {
      throw new InputFileNotFoundError(path, "JSON");
    }
    if (!response.ok) {
      throw new InputFileNotFoundError(path, "JSON");
    }
    const text = await response.text();
    try {
      return JSON.parse(text);
    } catch (error) {
      throw new ConfigLoadError(path, `Invalid JSON format: ${error}`);
    }
  }
  function parseJsonString(jsonStr, sourcePath = "<inline>") {
    try {
      return JSON.parse(jsonStr);
    } catch (error) {
      throw new ConfigLoadError(sourcePath, `Invalid JSON format: ${error}`);
    }
  }
  var _PathUtils = class _PathUtils {
    static removeNonUtfCharacters(pathString) {
      return [...pathString].filter((c) => _PathUtils.PRINTABLE.has(c)).join("");
    }
    static sepBasedPosixPath(path) {
      const normalized = path.replace(/\\/g, "/");
      return _PathUtils.removeNonUtfCharacters(normalized);
    }
    constructor(outputDir) {
      this.outputDir = outputDir;
      this.saveMarkedDir = `${outputDir}/CheckedOMRs`;
      this.imageMetricsDir = `${outputDir}/ImageMetrics`;
      this.resultsDir = `${outputDir}/Results`;
      this.manualDir = `${outputDir}/Manual`;
      this.errorsDir = `${this.manualDir}/ErrorFiles`;
      this.multiMarkedDir = `${this.manualDir}/MultiMarkedFiles`;
      this.evaluationsDir = `${outputDir}/Evaluations`;
      this.debugDir = `${outputDir}/Debug`;
    }
  };
  _PathUtils.PRINTABLE = new Set(
    Array.from({ length: 128 }, (_, i) => String.fromCharCode(i)).filter(
      (c) => c.trim() !== "" || c === " "
    )
  );
  var PathUtils = _PathUtils;

  // src/utils/file_pattern_resolver.ts
  var FilePatternResolver = class {
    constructor(options) {
      var _a;
      this.baseDir = options == null ? void 0 : options.baseDir;
      this.existsCheck = (_a = options == null ? void 0 : options.existsCheck) != null ? _a : (() => false);
    }
    resolvePattern(pattern, fields, options) {
      const { originalPath, collisionStrategy = "skip" } = options != null ? options : {};
      try {
        const formatted = this._formatPattern(pattern, fields);
        const sanitized = this._sanitizePath(formatted);
        let resolvedPath = sanitized;
        if (originalPath && !this._getSuffix(resolvedPath)) {
          const ext = this._getSuffix(originalPath);
          if (ext) resolvedPath = resolvedPath + ext;
        }
        if (this.baseDir) {
          resolvedPath = this.baseDir + "/" + resolvedPath;
        }
        return this._handleCollision(resolvedPath, collisionStrategy);
      } catch (e) {
        if (e instanceof KeyError) {
          logger.warning(`Pattern references undefined field: ${e.key}`);
        } else {
          logger.error(`Error resolving pattern '${pattern}': ${e}`);
        }
        return null;
      }
    }
    _formatPattern(pattern, fields) {
      return pattern.replace(/\{(\w+)\}/g, (_match, key) => {
        if (!(key in fields)) {
          throw new KeyError(key);
        }
        return fields[key];
      });
    }
    _sanitizePath(pathStr) {
      const parts = pathStr.split("/");
      const sanitizedParts = [];
      for (const part of parts) {
        let sanitized = part.replace(/[<>:"|?*\\]/g, "_");
        sanitized = sanitized.replace(/_+/g, "_");
        sanitized = sanitized.replace(/^[_ ]+|[_ ]+$/g, "");
        if (sanitized) {
          sanitizedParts.push(sanitized);
        }
      }
      return sanitizedParts.join("/");
    }
    /**
     * Returns the file extension including the leading dot (e.g. ".jpg"),
     * or an empty string if there is no extension.
     */
    _getSuffix(pathStr) {
      var _a;
      const name = (_a = pathStr.split("/").pop()) != null ? _a : "";
      const dotIndex = name.lastIndexOf(".");
      if (dotIndex <= 0) return "";
      return name.slice(dotIndex);
    }
    /**
     * Returns the file name without extension (stem).
     */
    _getStem(pathStr) {
      var _a;
      const name = (_a = pathStr.split("/").pop()) != null ? _a : "";
      const dotIndex = name.lastIndexOf(".");
      if (dotIndex <= 0) return name;
      return name.slice(0, dotIndex);
    }
    /**
     * Returns the parent directory path (everything before the last '/').
     */
    _getParent(pathStr) {
      const idx = pathStr.lastIndexOf("/");
      if (idx < 0) return "";
      return pathStr.slice(0, idx);
    }
    _handleCollision(path, strategy) {
      if (!this.existsCheck(path)) {
        return path;
      }
      if (strategy === "skip") {
        return null;
      }
      if (strategy === "overwrite") {
        return path;
      }
      if (strategy === "increment") {
        const stem = this._getStem(path);
        const suffix = this._getSuffix(path);
        const parent = this._getParent(path);
        let counter = 1;
        while (counter < 9999) {
          const newName = `${stem}_${String(counter).padStart(3, "0")}${suffix}`;
          const newPath = parent ? parent + "/" + newName : newName;
          if (!this.existsCheck(newPath)) {
            return newPath;
          }
          counter++;
        }
        return null;
      }
      return null;
    }
    resolveBatch(patternsAndFields, collisionStrategy = "skip") {
      const results = [];
      for (const [pattern, fields, originalPath] of patternsAndFields) {
        const resolved = this.resolvePattern(pattern, fields, { originalPath, collisionStrategy });
        results.push([resolved, fields]);
      }
      return results;
    }
  };
  var KeyError = class extends Error {
    constructor(key) {
      super(`KeyError: '${key}'`);
      this.key = key;
      this.name = "KeyError";
    }
  };

  // src/processors/image/point_utils.ts
  function argsort(arr) {
    return arr.map((v, i) => [v, i]).sort(([a], [b]) => a - b).map(([, i]) => i);
  }
  function isNx2(arr) {
    return Array.isArray(arr) && arr.every(
      (p) => Array.isArray(p) && p.length === 2 && p.every((v) => typeof v === "number")
    );
  }
  function clonePoints(points) {
    return points.map((p) => [...p]);
  }
  function ensurePointArray(input) {
    if (!isNx2(input)) {
      throw new TypeError(`Cannot convert input to Nx2 point array: ${JSON.stringify(input)}`);
    }
    return input.map((p) => [p[0], p[1]]);
  }
  var PointParser = class _PointParser {
    /**
     * Parse a points specification.
     *
     * @param pointsSpec - One of:
     *   - `number[][]`            — an Nx2 array; used as both control and dest
     *   - `[number[][], number[][]]` — a 2-tuple of (control, dest) arrays
     *   - `string`                — a named reference resolved via options
     * @param options - Optional resolution context for string references.
     * @returns Tuple `[controlPoints, destinationPoints]`, each Nx2.
     */
    static parsePoints(pointsSpec, options = {}) {
      const { templateDimensions, pageDimensions, context } = options;
      if (typeof pointsSpec === "string") {
        return _PointParser._parseStringReference(pointsSpec, templateDimensions, pageDimensions, context);
      }
      if (Array.isArray(pointsSpec) && pointsSpec.length === 2 && isNx2(pointsSpec[0]) && isNx2(pointsSpec[1])) {
        const [control, dest] = pointsSpec;
        return [ensurePointArray(control), ensurePointArray(dest)];
      }
      if (Array.isArray(pointsSpec) && isNx2(pointsSpec)) {
        const points = ensurePointArray(pointsSpec);
        return [points, clonePoints(points)];
      }
      throw new TypeError(
        `Invalid points specification type: ${typeof pointsSpec}. Expected list, tuple, string, or numpy array.`
      );
    }
    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------
    static _parseStringReference(reference, templateDimensions, pageDimensions, context) {
      if (reference === "template.dimensions") {
        if (templateDimensions == null) {
          throw new Error("template.dimensions reference requires template_dimensions");
        }
        return _PointParser._createCornerPoints(templateDimensions);
      }
      if (reference === "page_dimensions") {
        if (pageDimensions == null) {
          throw new Error("page_dimensions reference requires page_dimensions");
        }
        return _PointParser._createCornerPoints(pageDimensions);
      }
      if (context != null && reference in context) {
        return _PointParser.parsePoints(context[reference], {
          templateDimensions,
          pageDimensions,
          context
        });
      }
      throw new Error(`Unknown point reference: ${reference}`);
    }
    /**
     * Create four corner points for a rectangle of the given dimensions.
     * Order: [[0,0], [w-1,0], [w-1,h-1], [0,h-1]]
     */
    static _createCornerPoints(dimensions) {
      const [w, h] = dimensions;
      const corners = [
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
      ];
      return [corners, clonePoints(corners)];
    }
    /**
     * Validate that control and destination point arrays are compatible.
     *
     * @param controlPoints  - Nx2 control points.
     * @param destPoints     - Nx2 destination points.
     * @param minPoints      - Minimum required number of points (default 4).
     * @throws Error if validation fails.
     */
    static validatePoints(controlPoints, destPoints, minPoints = 4) {
      if (!isNx2(controlPoints)) {
        throw new Error(
          `control_points must be Nx2 array, got shape [${controlPoints.length}]`
        );
      }
      if (!isNx2(destPoints)) {
        throw new Error(
          `destination_points must be Nx2 array, got shape [${destPoints.length}]`
        );
      }
      if (controlPoints.length !== destPoints.length) {
        throw new Error(
          `Mismatch: ${controlPoints.length} control points vs ${destPoints.length} destination points`
        );
      }
      if (controlPoints.length < minPoints) {
        throw new Error(
          `At least ${minPoints} points required, got ${controlPoints.length}`
        );
      }
    }
  };
  var WarpedDimensionsCalculator = class {
    /**
     * Calculate output dimensions from a set of destination points.
     *
     * @param points       - Nx2 destination points.
     * @param padding      - Extra pixels added to each side (default 0).
     * @param maxDimension - Optional cap; if either dimension exceeds this value
     *                       both are scaled down proportionally.
     * @returns `[width, height]` as integers.
     */
    static calculateFromPoints(points, padding = 0, maxDimension) {
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      let width = Math.ceil(maxX - minX) + 1 + 2 * padding;
      let height = Math.ceil(maxY - minY) + 1 + 2 * padding;
      if (maxDimension != null) {
        const largest = Math.max(width, height);
        if (width > maxDimension || height > maxDimension) {
          const scale = maxDimension / largest;
          width = Math.floor(width * scale);
          height = Math.floor(height * scale);
        }
      }
      return [width, height];
    }
    /**
     * Scale a pair of dimensions by `scale`.
     *
     * @param dimensions - `[width, height]` input.
     * @param scale      - Scale factor (default 1.0).
     * @returns `[width, height]` after scaling, truncated to integers.
     */
    static calculateFromDimensions(dimensions, scale = 1) {
      const [w, h] = dimensions;
      return [Math.floor(w * scale), Math.floor(h * scale)];
    }
  };
  function orderFourPoints(points) {
    if (points.length !== 4) {
      throw new Error(`order_four_points requires exactly 4 points, got ${points.length}`);
    }
    const ySortedIndices = argsort(points.map((p) => p[1]));
    const sortedByY = ySortedIndices.map((i) => points[i]);
    const topPoints = sortedByY.slice(0, 2);
    const bottomPoints = sortedByY.slice(2);
    const topXIndices = argsort(topPoints.map((p) => p[0]));
    const [topLeft, topRight] = topXIndices.map((i) => topPoints[i]);
    const bottomXIndices = argsort(bottomPoints.map((p) => p[0]));
    const [bottomLeft, bottomRight] = bottomXIndices.map((i) => bottomPoints[i]);
    return [topLeft, topRight, bottomRight, bottomLeft];
  }
  function computePointDistances(points1, points2) {
    if (points1.length !== points2.length) {
      throw new Error("Point arrays must have same length");
    }
    return points1.map((p1, i) => {
      const p2 = points2[i];
      const dx = p2[0] - p1[0];
      const dy = p2[1] - p1[1];
      return Math.sqrt(dx * dx + dy * dy);
    });
  }
  function computeBoundingBox(points) {
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    return [
      Math.floor(Math.min(...xs)),
      Math.floor(Math.min(...ys)),
      Math.ceil(Math.max(...xs)),
      Math.ceil(Math.max(...ys))
    ];
  }

  // src/processors/image/constants.ts
  var PIXEL_VALUE_MAX = 255;
  var THRESH_PAGE_TRUNCATE_HIGH = 210;
  var THRESH_PAGE_TRUNCATE_SECONDARY = 200;
  var CANNY_THRESHOLD_HIGH = 185;
  var CANNY_THRESHOLD_LOW = 55;
  var MIN_PAGE_AREA = 8e4;
  var APPROX_POLY_EPSILON_FACTOR = 0.025;
  var CONTOUR_THICKNESS_STANDARD = 10;
  var TOP_CONTOURS_COUNT = 5;

  // src/processors/image/page_detection.ts
  var CLR_WHITE2 = [255, 255, 255];
  var HSV_WHITE_LOW = [0, 0, 150];
  var HSV_WHITE_HIGH = [180, 60, 255];
  function preparePageImage(image) {
    const truncated = new cv_shim_default.Mat();
    cv_shim_default.threshold(image, truncated, THRESH_PAGE_TRUNCATE_HIGH, 255, cv_shim_default.THRESH_TRUNC);
    const normalized = ImageUtils.normalizeSingle(truncated);
    truncated.delete();
    return normalized;
  }
  function applyColoredCanny(image, coloredImage) {
    const hsv = new cv_shim_default.Mat();
    const hsvChannels = new cv_shim_default.MatVector();
    const maskV = new cv_shim_default.Mat();
    const maskS = new cv_shim_default.Mat();
    const mask = new cv_shim_default.Mat();
    const masked = new cv_shim_default.Mat();
    const edges = new cv_shim_default.Mat();
    try {
      cv_shim_default.cvtColor(coloredImage, hsv, cv_shim_default.COLOR_BGR2HSV);
      cv_shim_default.split(hsv, hsvChannels);
      const vChannel = hsvChannels.get(2);
      const sChannel = hsvChannels.get(1);
      cv_shim_default.threshold(vChannel, maskV, HSV_WHITE_LOW[2] - 1, 255, cv_shim_default.THRESH_BINARY);
      cv_shim_default.threshold(sChannel, maskS, HSV_WHITE_HIGH[1], 255, cv_shim_default.THRESH_BINARY_INV);
      cv_shim_default.bitwise_and(maskV, maskS, mask);
      vChannel.delete();
      sChannel.delete();
      cv_shim_default.bitwise_and(image, image, masked, mask);
      cv_shim_default.Canny(masked, edges, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW);
      return edges.clone();
    } finally {
      hsv.delete();
      hsvChannels.delete();
      maskV.delete();
      maskS.delete();
      mask.delete();
      masked.delete();
      edges.delete();
    }
  }
  function applyGrayscaleCanny(image, morphKernel) {
    const truncated = new cv_shim_default.Mat();
    cv_shim_default.threshold(image, truncated, THRESH_PAGE_TRUNCATE_SECONDARY, 255, cv_shim_default.THRESH_TRUNC);
    const normalized = ImageUtils.normalizeSingle(truncated);
    truncated.delete();
    let processed;
    let ownedClosed = null;
    if (morphKernel && morphKernel.rows > 1) {
      ownedClosed = new cv_shim_default.Mat();
      cv_shim_default.morphologyEx(normalized, ownedClosed, cv_shim_default.MORPH_CLOSE, morphKernel);
      processed = ownedClosed;
    } else {
      processed = normalized;
    }
    const edges = new cv_shim_default.Mat();
    try {
      cv_shim_default.Canny(processed, edges, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW);
      return edges.clone();
    } finally {
      edges.delete();
      normalized.delete();
      if (ownedClosed) {
        ownedClosed.delete();
      }
    }
  }
  function findPageContours(cannyEdge) {
    const contours = new cv_shim_default.MatVector();
    const hierarchy = new cv_shim_default.Mat();
    try {
      cv_shim_default.findContours(cannyEdge, contours, hierarchy, cv_shim_default.RETR_LIST, cv_shim_default.CHAIN_APPROX_SIMPLE);
    } finally {
      hierarchy.delete();
    }
    const hullList = [];
    const size = contours.size();
    for (let i = 0; i < size; i++) {
      const hull = new cv_shim_default.Mat();
      cv_shim_default.convexHull(contours.get(i), hull);
      hullList.push(hull);
    }
    contours.delete();
    hullList.sort((a, b) => cv_shim_default.contourArea(b) - cv_shim_default.contourArea(a));
    const topContours = hullList.slice(0, TOP_CONTOURS_COUNT);
    for (let i = TOP_CONTOURS_COUNT; i < hullList.length; i++) {
      hullList[i].delete();
    }
    return topContours;
  }
  function extractApproxPoints(approx) {
    const n = approx.rows;
    const points = [];
    for (let i = 0; i < n; i++) {
      const x = approx.data32S[i * 2];
      const y = approx.data32S[i * 2 + 1];
      points.push([x, y]);
    }
    return points;
  }
  function extractPageRectangle(contours) {
    for (const contour of contours) {
      const area = cv_shim_default.contourArea(contour);
      if (area < MIN_PAGE_AREA) {
        continue;
      }
      const perimeter = cv_shim_default.arcLength(contour, true);
      const epsilon = APPROX_POLY_EPSILON_FACTOR * perimeter;
      const approx = new cv_shim_default.Mat();
      try {
        cv_shim_default.approxPolyDP(contour, approx, epsilon, true);
        if (approx.rows !== 4) {
          continue;
        }
        const points = extractApproxPoints(approx);
        if (MathUtils.validateRect(points)) {
          const corners = points.map(([x, y]) => [x, y]);
          const fullContour = contour.clone();
          return [corners, fullContour];
        }
      } finally {
        approx.delete();
      }
    }
    return [null, null];
  }
  function findPageContourAndCorners(image, options = {}) {
    const { coloredImage, useColoredCanny = false, morphKernel, filePath, debugImage } = options;
    const prepared = preparePageImage(image);
    let cannyEdge;
    try {
      if (useColoredCanny && coloredImage) {
        cannyEdge = applyColoredCanny(prepared, coloredImage);
      } else {
        cannyEdge = applyGrayscaleCanny(prepared, morphKernel);
      }
    } finally {
      prepared.delete();
    }
    let contours;
    try {
      contours = findPageContours(cannyEdge);
    } finally {
      cannyEdge.delete();
    }
    let corners = null;
    let pageContour = null;
    try {
      [corners, pageContour] = extractPageRectangle(contours);
    } finally {
      for (const c of contours) {
        c.delete();
      }
    }
    if (corners !== null && pageContour !== null && debugImage) {
      const approxPoints = corners.map(([x, y]) => [x, y]);
      DrawingUtils.drawContour(debugImage, approxPoints, CLR_WHITE2, CONTOUR_THICKNESS_STANDARD);
    }
    if (pageContour === null) {
      throw new ImageProcessingError("Paper boundary not found", {
        filePath,
        reason: `No valid rectangle found in top contour candidates`
      });
    }
    return [corners, pageContour];
  }

  // src/processors/image/warp_strategies.ts
  var WarpStrategy = class {
  };
  var PerspectiveTransformStrategy = class extends WarpStrategy {
    constructor(interpolationFlag) {
      super();
      this.interpolationFlag = interpolationFlag != null ? interpolationFlag : cv_shim_default.INTER_LINEAR;
    }
    getName() {
      return "PerspectiveTransform";
    }
    warpImage(image, coloredImage, controlPoints, destinationPoints, warpedDimensions, debugImage) {
      if (controlPoints.length !== 4) {
        throw new Error(
          `PerspectiveTransform requires exactly 4 control points, got ${controlPoints.length}`
        );
      }
      const [w, h] = warpedDimensions;
      const controlMat = cv_shim_default.matFromArray(4, 1, cv_shim_default.CV_32FC2, controlPoints.flat());
      const destMat = cv_shim_default.matFromArray(4, 1, cv_shim_default.CV_32FC2, destinationPoints.flat());
      const M = cv_shim_default.getPerspectiveTransform(controlMat, destMat);
      const dsize = new cv_shim_default.Size(w, h);
      const warpedGray = new cv_shim_default.Mat();
      cv_shim_default.warpPerspective(image, warpedGray, M, dsize, this.interpolationFlag);
      let warpedColored = null;
      if (coloredImage) {
        warpedColored = new cv_shim_default.Mat();
        cv_shim_default.warpPerspective(coloredImage, warpedColored, M, dsize, this.interpolationFlag);
      }
      let warpedDebug = null;
      if (debugImage) {
        warpedDebug = new cv_shim_default.Mat();
        cv_shim_default.warpPerspective(debugImage, warpedDebug, M, dsize, this.interpolationFlag);
      }
      controlMat.delete();
      destMat.delete();
      M.delete();
      return { warpedGray, warpedColored, warpedDebug };
    }
  };
  var HomographyStrategy = class extends WarpStrategy {
    constructor(options) {
      var _a, _b, _c;
      super();
      this.interpolationFlag = (_a = options == null ? void 0 : options.interpolationFlag) != null ? _a : cv_shim_default.INTER_LINEAR;
      this.useRansac = (_b = options == null ? void 0 : options.useRansac) != null ? _b : false;
      this.ransacThreshold = (_c = options == null ? void 0 : options.ransacThreshold) != null ? _c : 3;
    }
    getName() {
      return "Homography";
    }
    warpImage(image, coloredImage, controlPoints, destinationPoints, warpedDimensions, debugImage) {
      if (controlPoints.length < 4) {
        throw new Error(
          `Homography requires at least 4 control points, got ${controlPoints.length}`
        );
      }
      const [w, h] = warpedDimensions;
      const n = controlPoints.length;
      const controlMat = cv_shim_default.matFromArray(n, 1, cv_shim_default.CV_32FC2, controlPoints.flat());
      const destMat = cv_shim_default.matFromArray(n, 1, cv_shim_default.CV_32FC2, destinationPoints.flat());
      const method = this.useRansac ? cv_shim_default.RANSAC : 0;
      const M = cv_shim_default.findHomography(controlMat, destMat, method, this.ransacThreshold);
      if (!M || M.rows === 0) {
        controlMat.delete();
        destMat.delete();
        if (M) M.delete();
        throw new Error("Failed to compute homography matrix");
      }
      const dsize = new cv_shim_default.Size(w, h);
      const warpedGray = new cv_shim_default.Mat();
      cv_shim_default.warpPerspective(image, warpedGray, M, dsize, this.interpolationFlag);
      let warpedColored = null;
      if (coloredImage) {
        warpedColored = new cv_shim_default.Mat();
        cv_shim_default.warpPerspective(coloredImage, warpedColored, M, dsize, this.interpolationFlag);
      }
      let warpedDebug = null;
      if (debugImage) {
        warpedDebug = new cv_shim_default.Mat();
        cv_shim_default.warpPerspective(debugImage, warpedDebug, M, dsize, this.interpolationFlag);
      }
      controlMat.delete();
      destMat.delete();
      M.delete();
      return { warpedGray, warpedColored, warpedDebug };
    }
  };
  var GridDataRemapStrategy = class extends WarpStrategy {
    constructor(interpolationMethod = "cubic") {
      super();
      this.interpolationMethod = interpolationMethod;
    }
    getName() {
      return "GridDataRemap";
    }
    warpImage(image, coloredImage, controlPoints, destinationPoints, warpedDimensions, debugImage) {
      const ctrl4 = controlPoints.slice(0, 4);
      const dest4 = destinationPoints.slice(0, 4);
      const perspStrategy = new PerspectiveTransformStrategy(cv_shim_default.INTER_LINEAR);
      return perspStrategy.warpImage(image, coloredImage, ctrl4, dest4, warpedDimensions, debugImage);
    }
  };
  var DocRefineRectifyStrategy = class extends WarpStrategy {
    getName() {
      return "DocRefineRectify";
    }
    warpImage(_image, _coloredImage, _controlPoints, _destinationPoints, _warpedDimensions, _debugImage) {
      throw new Error("DocRefineRectify is not available in browser environment");
    }
  };
  var WarpStrategyFactory = class {
    /**
     * Create a WarpStrategy by method name with optional configuration.
     *
     * @param methodName - One of: PERSPECTIVE_TRANSFORM, HOMOGRAPHY, REMAP_GRIDDATA, DOC_REFINE
     * @param config - Optional configuration object passed to the strategy constructor
     * @throws Error if the method name is unknown
     */
    static create(methodName, config) {
      const StrategyClass = this.strategies[methodName];
      if (!StrategyClass) {
        throw new Error(`Unknown warp method '${methodName}'.`);
      }
      if (config !== void 0) {
        return new StrategyClass(config);
      }
      return new StrategyClass();
    }
    /**
     * Returns the list of available warp method names.
     */
    static getAvailableMethods() {
      return Object.keys(this.strategies);
    }
  };
  WarpStrategyFactory.strategies = {
    PERSPECTIVE_TRANSFORM: PerspectiveTransformStrategy,
    HOMOGRAPHY: HomographyStrategy,
    REMAP_GRIDDATA: GridDataRemapStrategy,
    DOC_REFINE: DocRefineRectifyStrategy
  };

  // src/processors/constants.ts
  var EdgeType2 = {
    TOP: "TOP",
    RIGHT: "RIGHT",
    BOTTOM: "BOTTOM",
    LEFT: "LEFT"
  };
  var EDGE_TYPES_IN_ORDER = [
    EdgeType2.TOP,
    EdgeType2.RIGHT,
    EdgeType2.BOTTOM,
    EdgeType2.LEFT
  ];
  var ScannerType = {
    PATCH_DOT: "PATCH_DOT",
    PATCH_LINE: "PATCH_LINE",
    TEMPLATE_MATCH: "TEMPLATE_MATCH"
  };
  var WarpMethod = {
    PERSPECTIVE_TRANSFORM: "PERSPECTIVE_TRANSFORM",
    HOMOGRAPHY: "HOMOGRAPHY",
    REMAP_GRIDDATA: "REMAP_GRIDDATA",
    DOC_REFINE: "DOC_REFINE",
    WARP_AFFINE: "WARP_AFFINE"
  };
  var WarpMethodFlags = {
    INTER_LINEAR: "INTER_LINEAR",
    // cv.INTER_LINEAR = 1
    INTER_CUBIC: "INTER_CUBIC",
    // cv.INTER_CUBIC = 2
    INTER_NEAREST: "INTER_NEAREST"
    // cv.INTER_NEAREST = 0
  };
  var WARP_METHOD_FLAG_VALUES = {
    INTER_LINEAR: 1,
    INTER_CUBIC: 2,
    INTER_NEAREST: 0
  };
  var ZonePreset = {
    topLeftDot: "topLeftDot",
    topRightDot: "topRightDot",
    bottomRightDot: "bottomRightDot",
    bottomLeftDot: "bottomLeftDot",
    topLeftMarker: "topLeftMarker",
    topRightMarker: "topRightMarker",
    bottomRightMarker: "bottomRightMarker",
    bottomLeftMarker: "bottomLeftMarker",
    topLine: "topLine",
    leftLine: "leftLine",
    bottomLine: "bottomLine",
    rightLine: "rightLine"
  };

  // src/processors/image/WarpOnPointsCommon.ts
  var WarpOnPointsCommon = class {
    constructor(options = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h;
      const parsed = this.validateAndRemapOptionsSchema(options);
      const tuningOptions = (_b = (_a = options.tuning_options) != null ? _a : parsed.tuning_options) != null ? _b : {};
      this.enableCropping = (_d = (_c = parsed.enable_cropping) != null ? _c : options.enable_cropping) != null ? _d : false;
      this.warpMethod = (_e = tuningOptions.warp_method) != null ? _e : this.enableCropping ? WarpMethod.PERSPECTIVE_TRANSFORM : WarpMethod.HOMOGRAPHY;
      this.warpMethodFlag = (_g = WARP_METHOD_FLAG_VALUES[(_f = tuningOptions.warp_method_flag) != null ? _f : "INTER_LINEAR"]) != null ? _g : 1;
      this.coloredOutputsEnabled = (_h = options.colored_outputs_enabled) != null ? _h : false;
      this.warpStrategy = WarpStrategyFactory.create(this.warpMethod);
    }
    // ── No-op hook (overridden in tests / subclasses) ─────────────────────────────
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    appendSaveImage(..._args) {
    }
    // ── Public pipeline ───────────────────────────────────────────────────────────
    applyFilter(image, coloredImage, template, filePath) {
      const prepared = this.prepareImageBeforeExtraction(image);
      const [controlPts, destPts, edgeMap] = this.extractControlDestinationPoints(
        prepared,
        coloredImage,
        filePath
      );
      const [parsedCtrl, parsedDest, dims] = this._parseAndPreparePoints(
        prepared,
        controlPts,
        destPts
      );
      const [warpedImage, warpedColored] = this._applyWarpStrategy(
        image,
        coloredImage,
        parsedCtrl,
        parsedDest,
        dims,
        edgeMap
      );
      this.appendSaveImage("Warped Image", [4, 5, 6], warpedImage, warpedColored);
      return [warpedImage, warpedColored, template];
    }
    // ── Internal helpers ──────────────────────────────────────────────────────────
    _parseAndPreparePoints(image, controlPoints, destinationPoints) {
      const seen = /* @__PURE__ */ new Map();
      const uniqueCtrl = [];
      const uniqueDest = [];
      for (let i = 0; i < controlPoints.length; i++) {
        const key = JSON.stringify(controlPoints[i]);
        if (!seen.has(key)) {
          seen.set(key, destinationPoints[i]);
          uniqueCtrl.push(controlPoints[i]);
          uniqueDest.push(destinationPoints[i]);
        }
      }
      const dims = this._calculateWarpedDimensions(
        [image.cols, image.rows],
        uniqueDest
      );
      return [uniqueCtrl, uniqueDest, dims];
    }
    _calculateWarpedDimensions(defaultDims, destinationPoints) {
      if (!this.enableCropping) return defaultDims;
      const pts = destinationPoints;
      const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(pts);
      const fromOrigin = [-boundingBox[0][0], -boundingBox[0][1]];
      const shifted = MathUtils.shiftPointsFromOrigin(fromOrigin, pts);
      for (let i = 0; i < destinationPoints.length; i++) {
        destinationPoints[i] = shifted[i];
      }
      return boxDimensions;
    }
    _applyWarpStrategy(image, coloredImage, controlPoints, destinationPoints, warpedDimensions, _edgeContoursMap) {
      const [ctrl, dest, dims] = this._preparePointsForStrategy(
        controlPoints,
        destinationPoints,
        warpedDimensions
      );
      const coloredInput = this.coloredOutputsEnabled ? coloredImage : null;
      const result = this.warpStrategy.warpImage(image, coloredInput, ctrl, dest, dims);
      return [result.warpedGray, result.warpedColored];
    }
    _preparePointsForStrategy(controlPoints, destinationPoints, warpedDimensions) {
      if (this.warpMethod !== WarpMethod.PERSPECTIVE_TRANSFORM) {
        return [controlPoints, destinationPoints, warpedDimensions];
      }
      if (controlPoints.length !== 4) {
        throw new TemplateError(
          `Expected 4 control points for perspective transform, found ${controlPoints.length}.`
        );
      }
      const { rect: orderedCtrl } = MathUtils.orderFourPoints(
        controlPoints
      );
      const [newDest, newDims] = ImageUtils.getCroppedWarpedRectanglePoints(
        orderedCtrl
      );
      return [orderedCtrl, newDest, newDims];
    }
  };

  // src/processors/image/CropPage.ts
  var _CropPage = class _CropPage extends WarpOnPointsCommon {
    constructor(options = {}) {
      var _a, _b, _c, _d;
      super(options);
      const morphKernel = (_b = (_a = options["morph_kernel"]) != null ? _a : options["morphKernel"]) != null ? _b : _CropPage.defaults.morphKernel;
      this.useColoredCanny = (_d = (_c = options["use_colored_canny"]) != null ? _c : options["useColoredCanny"]) != null ? _d : _CropPage.defaults.useColoredCanny;
      this.morphKernel = cv_shim_default.getStructuringElement(
        cv_shim_default.MORPH_RECT,
        new cv_shim_default.Size(morphKernel[0], morphKernel[1])
      );
    }
    getClassName() {
      return "CropPage";
    }
    getName() {
      return "CropPage";
    }
    validateAndRemapOptionsSchema(options) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i;
      const tuningOptions = (_b = (_a = options["tuning_options"]) != null ? _a : options["tuningOptions"]) != null ? _b : {};
      const morphKernel = (_d = (_c = options["morph_kernel"]) != null ? _c : options["morphKernel"]) != null ? _d : _CropPage.defaults.morphKernel;
      const useColoredCanny = (_f = (_e = options["use_colored_canny"]) != null ? _e : options["useColoredCanny"]) != null ? _f : _CropPage.defaults.useColoredCanny;
      return {
        morph_kernel: morphKernel,
        use_colored_canny: useColoredCanny,
        max_points_per_edge: (_h = (_g = options["max_points_per_edge"]) != null ? _g : options["maxPointsPerEdge"]) != null ? _h : null,
        enable_cropping: true,
        tuning_options: {
          warp_method: (_i = tuningOptions["warp_method"]) != null ? _i : WarpMethod.PERSPECTIVE_TRANSFORM,
          normalize_config: [],
          canny_config: []
        }
      };
    }
    prepareImageBeforeExtraction(image) {
      return ImageUtils.normalizeSingle(image);
    }
    extractControlDestinationPoints(image, _coloredImage, filePath) {
      const [corners, pageContour] = findPageContourAndCorners(image, {
        morphKernel: this.morphKernel,
        useColoredCanny: this.useColoredCanny,
        filePath
      });
      pageContour.delete();
      const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(corners);
      return [corners, warpedPoints, null];
    }
    /**
     * Release the morphKernel Mat to avoid memory leaks.
     */
    dispose() {
      if (this.morphKernel && !this.morphKernel.isDeleted()) {
        this.morphKernel.delete();
      }
    }
  };
  _CropPage.defaults = {
    morphKernel: [10, 10],
    useColoredCanny: false
  };
  var CropPage = _CropPage;

  // src/processors/image/crop_on_patches/marker_detection.ts
  function prepareMarkerTemplate(referenceImage, referenceZone, markerDimensions, blurKernel = [5, 5], applyErodeSubtract = true) {
    const mats = [];
    try {
      const [x, y] = referenceZone.origin;
      const [w, h] = referenceZone.dimensions;
      const roi = referenceImage.roi(new cv_shim_default.Rect(x, y, w, h));
      let marker = roi.clone();
      mats.push(marker);
      if (markerDimensions !== void 0) {
        const resized = ImageUtils.resizeSingle(marker, markerDimensions[0], markerDimensions[1]);
        mats.push(resized);
        const oldMarker = marker;
        marker = resized;
        mats.splice(mats.indexOf(oldMarker), 1);
        oldMarker.delete();
      }
      const blurred = new cv_shim_default.Mat();
      mats.push(blurred);
      cv_shim_default.GaussianBlur(marker, blurred, new cv_shim_default.Size(blurKernel[0], blurKernel[1]), 0);
      {
        const oldMarker = marker;
        marker = blurred;
        mats.splice(mats.indexOf(oldMarker), 1);
        oldMarker.delete();
      }
      const normalized = new cv_shim_default.Mat();
      mats.push(normalized);
      cv_shim_default.normalize(marker, normalized, 0, 255, cv_shim_default.NORM_MINMAX, cv_shim_default.CV_8U);
      {
        const oldMarker = marker;
        marker = normalized;
        mats.splice(mats.indexOf(oldMarker), 1);
        oldMarker.delete();
      }
      if (applyErodeSubtract) {
        const kernel = cv_shim_default.Mat.ones(5, 5, cv_shim_default.CV_8U);
        mats.push(kernel);
        const eroded = new cv_shim_default.Mat();
        mats.push(eroded);
        cv_shim_default.erode(marker, eroded, kernel, new cv_shim_default.Point(-1, -1), 5);
        const subtracted = new cv_shim_default.Mat();
        mats.push(subtracted);
        cv_shim_default.subtract(marker, eroded, subtracted);
        const renormalized = new cv_shim_default.Mat();
        mats.push(renormalized);
        cv_shim_default.normalize(subtracted, renormalized, 0, 255, cv_shim_default.NORM_MINMAX, cv_shim_default.CV_8U);
        const oldMarker = marker;
        marker = renormalized;
        mats.splice(mats.indexOf(oldMarker), 1);
        oldMarker.delete();
      }
      mats.splice(mats.indexOf(marker), 1);
      return marker;
    } finally {
      mats.forEach((m) => {
        try {
          m.delete();
        } catch (_) {
        }
      });
    }
  }
  function multiScaleTemplateMatch(patch, marker, scaleRange = [85, 115], scaleSteps = 5) {
    const descentPerStep = Math.floor((scaleRange[1] - scaleRange[0]) / scaleSteps);
    const markerH = marker.rows;
    const markerW = marker.cols;
    const patchH = patch.rows;
    const patchW = patch.cols;
    let bestPosition = null;
    let bestMarker = null;
    let bestConfidence = 0;
    let bestScalePercent = null;
    for (let scalePercent = scaleRange[1]; scalePercent > scaleRange[0]; scalePercent -= descentPerStep) {
      const scale = scalePercent / 100;
      if (scale <= 0) continue;
      const scaledW = Math.floor(markerW * scale);
      const scaledH = Math.floor(markerH * scale);
      const scaledMarker = ImageUtils.resizeSingle(marker, scaledW, scaledH);
      if (scaledH > patchH || scaledW > patchW) {
        scaledMarker.delete();
        continue;
      }
      const matchResult = new cv_shim_default.Mat();
      cv_shim_default.matchTemplate(patch, scaledMarker, matchResult, cv_shim_default.TM_CCOEFF_NORMED);
      const minMax = cv_shim_default.minMaxLoc(matchResult, new cv_shim_default.Mat());
      const confidence = minMax.maxVal;
      matchResult.delete();
      if (confidence > bestConfidence) {
        if (bestMarker !== null) {
          bestMarker.delete();
        }
        bestScalePercent = scalePercent;
        bestMarker = scaledMarker;
        bestConfidence = confidence;
        bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
      } else {
        scaledMarker.delete();
      }
    }
    return {
      position: bestPosition,
      optimalMarker: bestMarker,
      confidence: bestConfidence,
      optimalScalePercent: bestScalePercent
    };
  }
  function extractMarkerCorners(position, marker, zoneOffset = [0, 0]) {
    const h = marker.rows;
    const w = marker.cols;
    const [x, y] = position;
    const corners = MathUtils.getRectanglePoints(x, y, w, h);
    const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);
    return absoluteCorners;
  }
  function detectMarkerInPatch(patch, marker, zoneOffset = [0, 0], scaleRange = [85, 115], scaleSteps = 5, minConfidence = 0.3) {
    const { position, optimalMarker, confidence } = multiScaleTemplateMatch(
      patch,
      marker,
      scaleRange,
      scaleSteps
    );
    if (position === null || optimalMarker === null) {
      if (optimalMarker !== null) {
        optimalMarker.delete();
      }
      return null;
    }
    if (confidence < minConfidence) {
      optimalMarker.delete();
      return null;
    }
    const corners = extractMarkerCorners(position, optimalMarker, zoneOffset);
    optimalMarker.delete();
    return corners;
  }
  function validateMarkerDetection(corners, expectedAreaRange) {
    if (corners === null || corners === void 0) return false;
    if (corners.length !== 4) return false;
    if (!corners.every((c) => Array.isArray(c) && c.length === 2)) return false;
    if (expectedAreaRange !== null && expectedAreaRange !== void 0) {
      const flatData = corners.flat().map(Math.round);
      const contourMat = cv_shim_default.matFromArray(4, 1, cv_shim_default.CV_32SC2, flatData);
      const area = cv_shim_default.contourArea(contourMat);
      contourMat.delete();
      const [minArea, maxArea] = expectedAreaRange;
      if (!(minArea <= area && area <= maxArea)) return false;
    }
    return true;
  }

  // src/processors/image/CropOnMarkers.ts
  var FOUR_MARKERS_ZONE_ORDER = [
    "topLeftMarker",
    "topRightMarker",
    "bottomRightMarker",
    "bottomLeftMarker"
  ];
  var CropOnMarkers = class _CropOnMarkers extends WarpOnPointsCommon {
    /**
     * @param options   - CropOnMarkers options
     * @param assetMats - Map of filename key → pre-decoded grayscale cv.Mat
     *                    The caller retains ownership of the input Mats;
     *                    this class clones/processes them internally.
     */
    constructor(options, assetMats) {
      var _a, _b, _c, _d, _e;
      super(options);
      /**
       * Prepared marker template Mats keyed by zone type.
       * Populated in initResizedMarkers(); each Mat must be deleted via dispose().
       */
      this.markerTemplates = /* @__PURE__ */ new Map();
      if (options.type !== "FOUR_MARKERS") {
        throw new ImageProcessingError(
          `CropOnMarkers: unsupported type '${options.type}'. Only 'FOUR_MARKERS' is supported.`
        );
      }
      this.referenceImageKey = options.reference_image;
      this.markerDimensions = options.marker_dimensions;
      const tuning = (_a = options.tuning_options) != null ? _a : {};
      this.minMatchingThreshold = (_b = tuning.min_matching_threshold) != null ? _b : 0.3;
      this.markerRescaleRange = (_c = tuning.marker_rescale_range) != null ? _c : [85, 115];
      this.markerRescaleSteps = (_d = tuning.marker_rescale_steps) != null ? _d : 5;
      this.applyErodeSubtract = (_e = tuning.apply_erode_subtract) != null ? _e : true;
      const refMat = assetMats[this.referenceImageKey];
      if (!refMat) {
        throw new ImageProcessingError(
          `CropOnMarkers: asset Mat not found for key '${this.referenceImageKey}'`,
          { key: this.referenceImageKey, availableKeys: Object.keys(assetMats) }
        );
      }
      this.initResizedMarkers(refMat);
    }
    /**
     * Async factory that decodes a base64-encoded image via the browser's
     * Image/canvas API and constructs a CropOnMarkers instance.
     *
     * @param options - CropOnMarkers options (reference_image key must exist in assets)
     * @param assets  - Map of filename key → base64 string (data URL or raw base64)
     * @returns Initialized CropOnMarkers instance (call .dispose() when done)
     */
    static async fromBase64(options, assets) {
      const key = options.reference_image;
      const b64 = assets[key];
      if (!b64) {
        throw new ImageProcessingError(
          `CropOnMarkers.fromBase64: asset '${key}' not found in assets map`,
          { key, availableKeys: Object.keys(assets) }
        );
      }
      const refMat = await _CropOnMarkers.decodeBase64ViaCanvas(b64);
      try {
        return new _CropOnMarkers(options, { [key]: refMat });
      } finally {
        refMat.delete();
      }
    }
    // ── Abstract method implementations ──────────────────────────────────────────
    validateAndRemapOptionsSchema(options) {
      var _a, _b, _c;
      const tuning = (_a = options["tuning_options"]) != null ? _a : {};
      return {
        enable_cropping: true,
        points_layout: (_b = options["type"]) != null ? _b : "FOUR_MARKERS",
        tuning_options: {
          warp_method: (_c = tuning["warp_method"]) != null ? _c : WarpMethod.PERSPECTIVE_TRANSFORM
        }
      };
    }
    /**
     * Normalize the image before marker extraction.
     * Port of Python CropOnCustomMarkers.prepare_image_before_extraction.
     */
    prepareImageBeforeExtraction(image) {
      return ImageUtils.normalizeSingle(image);
    }
    /**
     * Detect all 4 corner markers and return their centers as control points.
     *
     * Returns [controlPoints, warpedPoints, null].
     * controlPoints = 4 center points of the detected markers (one per zone).
     * warpedPoints = corresponding destination points.
     */
    extractControlDestinationPoints(image, _coloredImage, filePath) {
      const allCorners = [];
      for (const zoneType of FOUR_MARKERS_ZONE_ORDER) {
        const marker = this.markerTemplates.get(zoneType);
        if (!marker) {
          throw new ImageProcessingError(
            `CropOnMarkers: marker template not initialized for zone '${zoneType}'`,
            { filePath, zoneType }
          );
        }
        const zoneDesc = this.getQuadrantZoneDescription(zoneType, image, marker);
        const markerCorners = this.findMarkerCornersInPatch(image, zoneDesc, marker, zoneType, filePath);
        const centerX = (markerCorners[0][0] + markerCorners[1][0] + markerCorners[2][0] + markerCorners[3][0]) / 4;
        const centerY = (markerCorners[0][1] + markerCorners[1][1] + markerCorners[2][1] + markerCorners[3][1]) / 4;
        allCorners.push([centerX, centerY]);
      }
      const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(allCorners);
      return [allCorners, warpedPoints, null];
    }
    // ── Private helpers ───────────────────────────────────────────────────────────
    /**
     * Prepare a marker template Mat for each zone from the reference image.
     * All zones in FOUR_MARKERS share the same reference image.
     *
     * @param referenceImage - pre-decoded grayscale cv.Mat (not modified)
     */
    initResizedMarkers(referenceImage) {
      const referenceZone = _CropOnMarkers.getDefaultScanZoneForImage(referenceImage);
      for (const zoneType of FOUR_MARKERS_ZONE_ORDER) {
        const marker = prepareMarkerTemplate(
          referenceImage,
          referenceZone,
          this.markerDimensions,
          [5, 5],
          this.applyErodeSubtract
        );
        this.markerTemplates.set(zoneType, marker);
      }
    }
    /**
     * Decode a base64-encoded image to a grayscale cv.Mat using the
     * browser's Image/canvas API (works without cv.imdecode).
     *
     * @param b64 - base64 string (data URL or raw base64)
     * @returns Promise<cv.Mat> grayscale image (caller must delete)
     */
    static async decodeBase64ViaCanvas(b64) {
      const dataUrl = b64.startsWith("data:") ? b64 : `data:image/jpeg;base64,${b64}`;
      const img = new Image();
      img.src = dataUrl;
      await new Promise((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error(`Failed to load image from base64 data`));
      });
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new ImageProcessingError("CropOnMarkers: could not create canvas 2D context");
      }
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      const rgba = cv_shim_default.matFromImageData(imageData);
      const gray = new cv_shim_default.Mat();
      cv_shim_default.cvtColor(rgba, gray, cv_shim_default.COLOR_RGBA2GRAY);
      rgba.delete();
      return gray;
    }
    /**
     * Compute the default reference zone covering the full image.
     * Port of Python CropOnCustomMarkers.get_default_scan_zone_for_image.
     */
    static getDefaultScanZoneForImage(image) {
      return {
        origin: [1, 1],
        dimensions: [image.cols - 1, image.rows - 1]
      };
    }
    /**
     * Compute the zone bounds (origin, dimensions, margins) for a given quadrant.
     *
     * Port of Python CropOnCustomMarkers.get_quadrant_zone_description.
     *
     * @param zoneType - Which corner marker zone
     * @param image    - The full image (for dimensions)
     * @param marker   - The marker template (for its dimensions)
     */
    getQuadrantZoneDescription(zoneType, image, marker) {
      const h = image.rows;
      const w = image.cols;
      const halfHeight = Math.floor(h / 2);
      const halfWidth = Math.floor(w / 2);
      const markerH = marker.rows;
      const markerW = marker.cols;
      let zoneStart;
      let zoneEnd;
      if (zoneType === "topLeftMarker") {
        zoneStart = [1, 1];
        zoneEnd = [halfWidth, halfHeight];
      } else if (zoneType === "topRightMarker") {
        zoneStart = [halfWidth, 1];
        zoneEnd = [w, halfHeight];
      } else if (zoneType === "bottomRightMarker") {
        zoneStart = [halfWidth, halfHeight];
        zoneEnd = [w, h];
      } else {
        zoneStart = [1, halfHeight];
        zoneEnd = [halfWidth, h];
      }
      const originX = Math.floor((zoneStart[0] + zoneEnd[0] - markerW) / 2);
      const originY = Math.floor((zoneStart[1] + zoneEnd[1] - markerH) / 2);
      const marginH = (zoneEnd[0] - zoneStart[0] - markerW) / 2 - 1;
      const marginV = (zoneEnd[1] - zoneStart[1] - markerH) / 2 - 1;
      return {
        origin: [originX, originY],
        dimensions: [markerW, markerH],
        margins: {
          top: marginV,
          right: marginH,
          bottom: marginV,
          left: marginH
        }
      };
    }
    /**
     * Extract the patch for a zone and run detectMarkerInPatch on it.
     *
     * Port of Python CropOnCustomMarkers.find_marker_corners_in_patch + compute_scan_zone.
     *
     * @param image      - Full prepared image
     * @param zoneDesc   - Zone description (origin, dimensions, margins)
     * @param marker     - Marker template
     * @param zoneType   - Zone type label (for error messages)
     * @param filePath   - File path (for error messages)
     * @returns 4 corner points [[x,y], ...] in absolute image coordinates
     */
    findMarkerCornersInPatch(image, zoneDesc, marker, zoneType, filePath) {
      const { origin, dimensions, margins } = zoneDesc;
      const [ox, oy] = origin;
      const [dw, dh] = dimensions;
      const marginTop = Math.max(0, Math.floor(margins.top));
      const marginRight = Math.max(0, Math.floor(margins.right));
      const marginBottom = Math.max(0, Math.floor(margins.bottom));
      const marginLeft = Math.max(0, Math.floor(margins.left));
      const patchX = Math.max(0, ox - marginLeft);
      const patchY = Math.max(0, oy - marginTop);
      const patchW = Math.min(image.cols - patchX, dw + marginLeft + marginRight);
      const patchH = Math.min(image.rows - patchY, dh + marginTop + marginBottom);
      if (patchW <= 0 || patchH <= 0) {
        throw new ImageProcessingError(
          `CropOnMarkers: degenerate patch for zone '${zoneType}'`,
          { filePath, zoneType, patchX, patchY, patchW, patchH }
        );
      }
      const rect = new cv_shim_default.Rect(patchX, patchY, patchW, patchH);
      const patch = image.roi(rect).clone();
      const zoneOffset = [patchX, patchY];
      let corners;
      try {
        corners = detectMarkerInPatch(
          patch,
          marker,
          zoneOffset,
          this.markerRescaleRange,
          this.markerRescaleSteps,
          this.minMatchingThreshold
        );
      } finally {
        patch.delete();
      }
      if (corners === null) {
        throw new ImageProcessingError(
          `CropOnMarkers: no marker found in patch for zone '${zoneType}'`,
          { filePath, zoneType }
        );
      }
      return corners;
    }
    /**
     * Release all marker template Mats to free Emscripten WASM memory.
     * Call this when the processor is no longer needed.
     */
    dispose() {
      for (const [, mat] of this.markerTemplates) {
        try {
          if (!mat.isDeleted()) {
            mat.delete();
          }
        } catch (_) {
        }
      }
      this.markerTemplates.clear();
    }
  };

  // src/processors/image/crop_on_patches/dot_line_detection.ts
  function preprocessDotZone(zone, dotKernel, dotThreshold = 150, blurKernel) {
    const mats = [];
    try {
      let working;
      if (blurKernel !== void 0) {
        const blurred = new cv_shim_default.Mat();
        mats.push(blurred);
        cv_shim_default.GaussianBlur(zone, blurred, new cv_shim_default.Size(blurKernel[0], blurKernel[1]), 0);
        working = blurred;
      } else {
        working = zone;
      }
      const kernelHeight = dotKernel.rows;
      const kernelWidth = dotKernel.cols;
      const { paddedImage, padRange } = ImageUtils.padImageFromCenter(
        working,
        kernelWidth * 2,
        kernelHeight * 2,
        255
      );
      mats.push(paddedImage);
      const morphed = new cv_shim_default.Mat();
      mats.push(morphed);
      cv_shim_default.morphologyEx(
        paddedImage,
        morphed,
        cv_shim_default.MORPH_OPEN,
        dotKernel,
        new cv_shim_default.Point(-1, -1),
        3
      );
      const thresholded = new cv_shim_default.Mat();
      mats.push(thresholded);
      cv_shim_default.threshold(morphed, thresholded, dotThreshold, 255, cv_shim_default.THRESH_TRUNC);
      const normalised = ImageUtils.normalizeSingle(thresholded);
      mats.push(normalised);
      const [rowStart, rowEnd, colStart, colEnd] = padRange;
      const cropped = normalised.roi(new cv_shim_default.Rect(colStart, rowStart, colEnd - colStart, rowEnd - rowStart)).clone();
      return cropped;
    } finally {
      mats.forEach((m) => m.delete());
    }
  }
  function preprocessLineZone(zone, lineKernel, gammaLow, lineThreshold = 180, blurKernel) {
    const mats = [];
    try {
      let working;
      if (blurKernel !== void 0) {
        const blurred = new cv_shim_default.Mat();
        mats.push(blurred);
        cv_shim_default.GaussianBlur(zone, blurred, new cv_shim_default.Size(blurKernel[0], blurKernel[1]), 0);
        working = blurred;
      } else {
        working = zone;
      }
      const darkerImage = ImageUtils.adjustGamma(working, gammaLow);
      mats.push(darkerImage);
      const thresholded = new cv_shim_default.Mat();
      mats.push(thresholded);
      cv_shim_default.threshold(darkerImage, thresholded, lineThreshold, 255, cv_shim_default.THRESH_TRUNC);
      const normalised = ImageUtils.normalizeSingle(thresholded);
      mats.push(normalised);
      const kernelHeight = lineKernel.rows;
      const kernelWidth = lineKernel.cols;
      const { paddedImage, padRange } = ImageUtils.padImageFromCenter(
        normalised,
        kernelWidth * 2,
        kernelHeight * 2,
        255
      );
      mats.push(paddedImage);
      const whiteThresholded = new cv_shim_default.Mat();
      mats.push(whiteThresholded);
      cv_shim_default.threshold(paddedImage, whiteThresholded, lineThreshold, 255, cv_shim_default.THRESH_TRUNC);
      const whiteNormalised = ImageUtils.normalizeSingle(whiteThresholded);
      mats.push(whiteNormalised);
      const lineMorphed = new cv_shim_default.Mat();
      mats.push(lineMorphed);
      cv_shim_default.morphologyEx(
        whiteNormalised,
        lineMorphed,
        cv_shim_default.MORPH_OPEN,
        lineKernel,
        new cv_shim_default.Point(-1, -1),
        3
      );
      const [rowStart, rowEnd, colStart, colEnd] = padRange;
      const cropped = lineMorphed.roi(new cv_shim_default.Rect(colStart, rowStart, colEnd - colStart, rowEnd - rowStart)).clone();
      return cropped;
    } finally {
      mats.forEach((m) => m.delete());
    }
  }
  function detectContoursUsingCanny(zone, cannyLow = 55, cannyHigh = 185) {
    const edges = new cv_shim_default.Mat();
    const contourVec = new cv_shim_default.MatVector();
    const hierarchy = new cv_shim_default.Mat();
    try {
      cv_shim_default.Canny(zone, edges, cannyHigh, cannyLow);
      cv_shim_default.findContours(edges, contourVec, hierarchy, cv_shim_default.RETR_LIST, cv_shim_default.CHAIN_APPROX_SIMPLE);
      const result = [];
      const size = contourVec.size();
      for (let i = 0; i < size; i++) {
        result.push(contourVec.get(i).clone());
      }
      result.sort((a, b) => cv_shim_default.contourArea(b) - cv_shim_default.contourArea(a));
      return result;
    } finally {
      edges.delete();
      contourVec.delete();
      hierarchy.delete();
    }
  }
  function extractPatchCornersAndEdges(contour, scannerType) {
    const data = contour.data32S;
    const boundaryPoints = [];
    for (let i = 0; i < data.length; i += 2) {
      boundaryPoints.push([data[i], data[i + 1]]);
    }
    if (boundaryPoints.length === 0) {
      throw new Error("Contour has no points");
    }
    const hull = new cv_shim_default.Mat();
    try {
      cv_shim_default.convexHull(contour, hull, false, true);
      let patchCorners;
      if (scannerType === ScannerType.PATCH_DOT) {
        const rect = cv_shim_default.boundingRect(hull);
        patchCorners = MathUtils.getRectanglePoints(rect.x, rect.y, rect.width, rect.height);
      } else if (scannerType === ScannerType.PATCH_LINE) {
        const rotRect = cv_shim_default.minAreaRect(hull);
        const boxPtsArray = cv_shim_default.boxPoints(rotRect);
        patchCorners = boxPtsArray.map((pt) => [Math.round(pt.x), Math.round(pt.y)]);
      } else {
        throw new Error(`Unsupported scanner type: ${scannerType}`);
      }
      const { orderedCorners, edgeContoursMap } = ImageUtils.splitPatchContourOnCorners(
        patchCorners,
        boundaryPoints
      );
      return {
        corners: orderedCorners,
        edgeContoursMap
      };
    } finally {
      hull.delete();
    }
  }
  function detectDotCorners(zone, zoneOffset, dotKernel, dotThreshold = 150, blurKernel) {
    const preprocessed = preprocessDotZone(zone, dotKernel, dotThreshold, blurKernel);
    const contours = detectContoursUsingCanny(preprocessed);
    preprocessed.delete();
    try {
      if (contours.length === 0) {
        return null;
      }
      const { corners } = extractPatchCornersAndEdges(contours[0], ScannerType.PATCH_DOT);
      if (corners === null || corners.length === 0) {
        return null;
      }
      const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);
      return absoluteCorners;
    } finally {
      contours.forEach((c) => c.delete());
    }
  }
  function detectLineCornersAndEdges(zone, zoneOffset, lineKernel, gammaLow, lineThreshold = 180, blurKernel) {
    const preprocessed = preprocessLineZone(zone, lineKernel, gammaLow, lineThreshold, blurKernel);
    const contours = detectContoursUsingCanny(preprocessed);
    preprocessed.delete();
    try {
      if (contours.length === 0) {
        return { corners: null, edgeContoursMap: null };
      }
      const { corners, edgeContoursMap } = extractPatchCornersAndEdges(
        contours[0],
        ScannerType.PATCH_LINE
      );
      if (corners === null || edgeContoursMap === null) {
        return { corners: null, edgeContoursMap: null };
      }
      const absoluteCorners = MathUtils.shiftPointsFromOrigin(zoneOffset, corners);
      const shiftedEdgeContoursMap = {};
      for (const edgeType of EDGE_TYPES_IN_ORDER) {
        shiftedEdgeContoursMap[edgeType] = MathUtils.shiftPointsFromOrigin(
          zoneOffset,
          edgeContoursMap[edgeType]
        );
      }
      return { corners: absoluteCorners, edgeContoursMap: shiftedEdgeContoursMap };
    } finally {
      contours.forEach((c) => c.delete());
    }
  }
  function validateBlurKernel(zoneShape, blurKernel, zoneLabel = "") {
    const [zoneH, zoneW] = zoneShape;
    const [blurH, blurW] = blurKernel;
    if (!(zoneH > blurH && zoneW > blurW)) {
      const labelStr = zoneLabel ? ` '${zoneLabel}'` : "";
      throw new Error(
        `The zone${labelStr} is smaller than provided blur kernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
      );
    }
    return true;
  }
  function createStructuringElement(shape, size) {
    const shapeMap = {
      rect: cv_shim_default.MORPH_RECT,
      ellipse: cv_shim_default.MORPH_ELLIPSE,
      cross: cv_shim_default.MORPH_CROSS
    };
    if (!(shape in shapeMap)) {
      throw new Error(`Unknown shape: ${shape}. Use ${JSON.stringify(Object.keys(shapeMap))}`);
    }
    return cv_shim_default.getStructuringElement(shapeMap[shape], new cv_shim_default.Size(size[0], size[1]));
  }

  // src/template/constants.ts
  var ZERO_MARGINS = { top: 0, bottom: 0, left: 0, right: 0 };
  var BUILTIN_BUBBLE_FIELD_TYPES = {
    QTYPE_INT: {
      bubble_values: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      direction: "vertical"
    },
    QTYPE_INT_FROM_1: {
      bubble_values: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
      direction: "vertical"
    },
    QTYPE_MCQ4: {
      bubble_values: ["A", "B", "C", "D"],
      direction: "horizontal"
    },
    QTYPE_MCQ5: {
      bubble_values: ["A", "B", "C", "D", "E"],
      direction: "horizontal"
    }
  };

  // src/template/parseFields.ts
  var FIELD_STRING_REGEX = /^([^.\d]+)(\d+)\.{2,3}(\d+)$/;
  function parseFieldString(fieldString) {
    if (!fieldString.includes(".")) {
      return [fieldString];
    }
    const match = FIELD_STRING_REGEX.exec(fieldString);
    if (!match) {
      throw new OMRCheckerError(
        `Invalid field string format: '${fieldString}'. Expected format like 'q1..5'`,
        { field_string: fieldString }
      );
    }
    const [, fieldPrefix, startStr, endStr] = match;
    const start = parseInt(startStr, 10);
    const end = parseInt(endStr, 10);
    if (start >= end) {
      throw new OMRCheckerError(
        `Invalid range in fields string: '${fieldString}', start: ${start} is not less than end: ${end}`,
        { field_string: fieldString, start, end }
      );
    }
    const result = [];
    for (let i = start; i <= end; i++) {
      result.push(`${fieldPrefix}${i}`);
    }
    return result;
  }
  function parseFields(key, fields) {
    const parsedFields = [];
    const fieldsSet = /* @__PURE__ */ new Set();
    for (const fieldString of fields) {
      const fieldsArray = parseFieldString(fieldString);
      const currentSet = new Set(fieldsArray);
      const overlap = [...currentSet].filter((f) => fieldsSet.has(f));
      if (overlap.length > 0) {
        throw new OMRCheckerError(
          `Given field string '${fieldString}' has overlapping field(s) with other fields in '${key}': ${fields}`,
          {
            field_string: fieldString,
            key,
            overlapping_fields: overlap
          }
        );
      }
      for (const f of currentSet) {
        fieldsSet.add(f);
      }
      parsedFields.push(...fieldsArray);
    }
    return parsedFields;
  }

  // src/template/BubblesScanBox.ts
  var ScanBox = class {
    constructor(fieldIndex, fieldLabel, origin, dimensions, margins) {
      this.fieldIndex = fieldIndex;
      this.dimensions = dimensions;
      this.margins = margins;
      this.origin = origin;
      this.x = Math.round(origin[0]);
      this.y = Math.round(origin[1]);
      this.fieldLabel = fieldLabel;
      this.name = `${fieldLabel}_${fieldIndex}`;
      this.shifts = [0, 0];
    }
    resetShifts() {
      this.shifts = [0, 0];
    }
    getShiftedPosition(extraShifts = [0, 0]) {
      return [
        this.x + this.shifts[0] + extraShifts[0],
        this.y + this.shifts[1] + extraShifts[1]
      ];
    }
  };
  var BubblesScanBox = class extends ScanBox {
    constructor(fieldIndex, fieldLabel, origin, dimensions, margins, bubbleValue, bubbleFieldType) {
      super(fieldIndex, fieldLabel, origin, dimensions, margins);
      this.bubbleValue = bubbleValue;
      this.bubbleDimensions = dimensions;
      this.bubbleFieldType = bubbleFieldType;
      this.name = `${fieldLabel}_${bubbleValue}`;
    }
  };

  // src/template/BubbleField.ts
  var BubbleField = class {
    constructor(direction, emptyValue, fieldBlockName, bubbleDimensions, bubbleValues, bubblesGap, bubbleFieldType, fieldLabel, origin) {
      this.direction = direction;
      this.emptyValue = emptyValue;
      this.fieldLabel = fieldLabel;
      this.id = `${fieldBlockName}::${fieldLabel}`;
      this.name = fieldLabel;
      this.origin = [origin[0], origin[1]];
      this.bubbleDimensions = bubbleDimensions;
      this.bubbleValues = bubbleValues;
      this.bubblesGap = bubblesGap;
      this.bubbleFieldType = bubbleFieldType;
      this.scanBoxes = [];
      this.setupScanBoxes();
    }
    /**
     * Generate scan boxes for each bubble value.
     *
     * Direction determines which axis the bubbles extend along:
     *   vertical   → h=1 → Y axis increments (bubbles stack top-to-bottom)
     *   horizontal → h=0 → X axis increments (bubbles go left-to-right)
     *
     * Ported from Python: src/processors/layout/field/bubble_field.py::BubbleField.setup_scan_boxes
     */
    setupScanBoxes() {
      const { bubbleValues, bubbleDimensions, bubblesGap, bubbleFieldType, direction } = this;
      if (!bubbleValues || bubbleValues.length === 0) {
        throw new Error("bubbleValues is required and must not be empty");
      }
      const h = direction === "vertical" ? 1 : 0;
      const bubblePoint = [this.origin[0], this.origin[1]];
      for (let fieldIndex = 0; fieldIndex < bubbleValues.length; fieldIndex++) {
        const bubbleValue = bubbleValues[fieldIndex];
        const bubbleOrigin = [bubblePoint[0], bubblePoint[1]];
        const scanBox = new BubblesScanBox(
          fieldIndex,
          this.fieldLabel,
          bubbleOrigin,
          bubbleDimensions,
          { ...ZERO_MARGINS },
          bubbleValue,
          bubbleFieldType
        );
        this.scanBoxes.push(scanBox);
        bubblePoint[h] += bubblesGap;
      }
    }
    resetAllShifts() {
      for (const scanBox of this.scanBoxes) {
        scanBox.resetShifts();
      }
    }
    toString() {
      return this.id;
    }
  };

  // src/template/FieldBlock.ts
  var FieldBlock = class {
    constructor(blockName, fieldBlockObject, fieldBlocksOffset) {
      this.name = blockName;
      this.shifts = [0, 0];
      this.setupFieldBlock(fieldBlockObject, fieldBlocksOffset);
      this.generateFields();
    }
    setupFieldBlock(obj, fieldBlocksOffset) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l, _m, _n, _o, _p, _q;
      const fieldDetectionType = (_b = (_a = obj.fieldDetectionType) != null ? _a : obj.field_detection_type) != null ? _b : "BUBBLES_THRESHOLD";
      if (fieldDetectionType !== "BUBBLES_THRESHOLD") {
        throw new OMRCheckerError(
          `Unsupported field detection type: ${fieldDetectionType}`,
          { field_detection_type: fieldDetectionType, block_name: this.name }
        );
      }
      this.direction = (_c = obj.direction) != null ? _c : "horizontal";
      this.emptyValue = (_e = (_d = obj.emptyValue) != null ? _d : obj.empty_value) != null ? _e : "";
      this.labelsGap = (_g = (_f = obj.labelsGap) != null ? _f : obj.labels_gap) != null ? _g : 0;
      const [ox, oy] = obj.origin;
      const [offsetX, offsetY] = fieldBlocksOffset;
      this.origin = [ox + offsetX, oy + offsetY];
      const fieldLabels = (_i = (_h = obj.fieldLabels) != null ? _h : obj.field_labels) != null ? _i : [];
      this.parsedFieldLabels = parseFields(`Field Block Labels: ${this.name}`, fieldLabels);
      this.bubbleDimensions = (_k = (_j = obj.bubbleDimensions) != null ? _j : obj.bubble_dimensions) != null ? _k : [10, 10];
      this.bubbleValues = (_m = (_l = obj.bubbleValues) != null ? _l : obj.bubble_values) != null ? _m : [];
      this.bubblesGap = (_o = (_n = obj.bubblesGap) != null ? _n : obj.bubbles_gap) != null ? _o : 0;
      this.bubbleFieldType = (_q = (_p = obj.bubbleFieldType) != null ? _p : obj.bubble_field_type) != null ? _q : "";
    }
    /**
     * Generate BubbleField instances, one per parsed label.
     *
     * Direction determines which axis labels extend along:
     *   vertical   → v=0 → X axis increments (labels stack left-to-right)
     *   horizontal → v=1 → Y axis increments (labels go top-to-bottom)
     *
     * Ported from Python: src/processors/layout/field_block/base.py::FieldBlock.generate_fields
     */
    generateFields() {
      const { direction, emptyValue, labelsGap, parsedFieldLabels } = this;
      const { bubbleDimensions, bubbleValues, bubblesGap, bubbleFieldType } = this;
      const v = direction === "vertical" ? 0 : 1;
      this.fields = [];
      const leadPoint = [
        parseFloat(String(this.origin[0])),
        parseFloat(String(this.origin[1]))
      ];
      for (const fieldLabel of parsedFieldLabels) {
        const origin = [leadPoint[0], leadPoint[1]];
        const field = new BubbleField(
          direction,
          emptyValue,
          this.name,
          bubbleDimensions,
          bubbleValues,
          bubblesGap,
          bubbleFieldType,
          fieldLabel,
          origin
        );
        this.fields.push(field);
        leadPoint[v] += labelsGap;
      }
      this.updateBoundingBox();
    }
    /**
     * Update the bounding box that covers all scan boxes in this field block.
     *
     * Ported from Python: src/processors/layout/field_block/base.py::FieldBlock.update_bounding_box
     */
    updateBoundingBox() {
      const allScanBoxes = this.fields.flatMap((field) => field.scanBoxes);
      if (allScanBoxes.length === 0) {
        this.boundingBoxOrigin = [this.origin[0], this.origin[1]];
        this.boundingBoxDimensions = [0, 0];
        return;
      }
      const minX = Math.min(...allScanBoxes.map((sb) => sb.origin[0]));
      const minY = Math.min(...allScanBoxes.map((sb) => sb.origin[1]));
      const maxX = Math.max(...allScanBoxes.map((sb) => sb.origin[0] + sb.dimensions[0]));
      const maxY = Math.max(...allScanBoxes.map((sb) => sb.origin[1] + sb.dimensions[1]));
      this.boundingBoxOrigin = [minX, minY];
      this.boundingBoxDimensions = [
        Math.round((maxX - minX) * 100) / 100,
        Math.round((maxY - minY) * 100) / 100
      ];
    }
    getShiftedOrigin() {
      return [
        this.origin[0] + this.shifts[0],
        this.origin[1] + this.shifts[1]
      ];
    }
    resetAllShifts() {
      this.shifts = [0, 0];
      for (const field of this.fields) {
        field.resetAllShifts();
      }
    }
  };

  // src/template/Template.ts
  var Template = class _Template {
    constructor(templateJson) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l, _m;
      this.templateDimensions = templateJson.templateDimensions;
      this.bubbleDimensions = templateJson.bubbleDimensions;
      const [pageWidth, pageHeight] = this.templateDimensions;
      const processingImageShapeRaw = (_b = (_a = templateJson.processingImageShape) != null ? _a : templateJson.processing_image_shape) != null ? _b : [pageHeight, pageWidth];
      this.processingImageShape = processingImageShapeRaw;
      this.fieldBlocksOffset = (_d = (_c = templateJson.fieldBlocksOffset) != null ? _c : templateJson.field_blocks_offset) != null ? _d : [0, 0];
      this.globalEmptyVal = (_f = (_e = templateJson.emptyValue) != null ? _e : templateJson.empty_value) != null ? _f : "";
      this.preProcessorsConfig = (_h = (_g = templateJson.preProcessors) != null ? _g : templateJson.pre_processors) != null ? _h : [];
      this.parseCustomBubbleFieldTypes(
        (_i = templateJson.customBubbleFieldTypes) != null ? _i : templateJson.custom_bubble_field_types
      );
      const customLabelsRaw = (_k = (_j = templateJson.customLabels) != null ? _j : templateJson.custom_labels) != null ? _k : {};
      const fieldBlocksRaw = (_m = (_l = templateJson.fieldBlocks) != null ? _l : templateJson.field_blocks) != null ? _m : {};
      this.allParsedLabels = /* @__PURE__ */ new Set();
      this.fieldBlocks = [];
      this.allFields = [];
      this.validateFieldBlocks(fieldBlocksRaw);
      this.setupLayout(fieldBlocksRaw);
      this.parseCustomLabels(customLabelsRaw);
      this.outputColumns = this.buildDefaultOutputColumns();
    }
    /**
     * Merge builtin bubble field types with any user-defined custom types.
     *
     * Ported from Python: TemplateLayout.parse_custom_bubble_field_types
     */
    parseCustomBubbleFieldTypes(customBubbleFieldTypes) {
      var _a, _b, _c;
      if (!customBubbleFieldTypes || Object.keys(customBubbleFieldTypes).length === 0) {
        this.bubbleFieldTypesData = { ...BUILTIN_BUBBLE_FIELD_TYPES };
        return;
      }
      const converted = {};
      for (const [typeName, typeData] of Object.entries(customBubbleFieldTypes)) {
        const bubbleValues = (_b = (_a = typeData.bubbleValues) != null ? _a : typeData.bubble_values) != null ? _b : [];
        const direction = (_c = typeData.direction) != null ? _c : "horizontal";
        converted[typeName] = { bubble_values: bubbleValues, direction };
      }
      this.bubbleFieldTypesData = {
        ...BUILTIN_BUBBLE_FIELD_TYPES,
        ...converted
      };
    }
    /**
     * Validate that all field blocks reference known bubble field types.
     *
     * Ported from Python: TemplateLayout.validate_field_blocks
     */
    validateFieldBlocks(fieldBlocksObject) {
      var _a, _b, _c, _d, _e, _f;
      for (const [blockName, fieldBlockObject] of Object.entries(fieldBlocksObject)) {
        const fieldDetectionType = (_b = (_a = fieldBlockObject.fieldDetectionType) != null ? _a : fieldBlockObject.field_detection_type) != null ? _b : "BUBBLES_THRESHOLD";
        if (fieldDetectionType === "BUBBLES_THRESHOLD") {
          const bubbleFieldType = (_c = fieldBlockObject.bubbleFieldType) != null ? _c : fieldBlockObject.bubble_field_type;
          if (!bubbleFieldType || !(bubbleFieldType in this.bubbleFieldTypesData)) {
            throw new OMRCheckerError(
              `Invalid bubble field type: ${bubbleFieldType} in block ${blockName}. Have you defined customBubbleFieldTypes?`,
              { bubble_field_type: bubbleFieldType, block_name: blockName }
            );
          }
          const fieldLabels = (_e = (_d = fieldBlockObject.fieldLabels) != null ? _d : fieldBlockObject.field_labels) != null ? _e : [];
          const labelsGap = (_f = fieldBlockObject.labelsGap) != null ? _f : fieldBlockObject.labels_gap;
          if (fieldLabels.length > 1 && labelsGap == null) {
            throw new OMRCheckerError(
              `More than one fieldLabels(${fieldLabels}) provided, but labels_gap not present for block ${blockName}`,
              { field_labels: fieldLabels, block_name: blockName }
            );
          }
        }
      }
    }
    /**
     * Build all FieldBlock instances from the raw JSON.
     *
     * Ported from Python: TemplateLayout.setup_layout
     */
    setupLayout(fieldBlocksObject) {
      for (const [blockName, fieldBlockObject] of Object.entries(fieldBlocksObject)) {
        const blockInstance = this.parseAndAddFieldBlock(blockName, fieldBlockObject);
        this.allFields.push(...blockInstance.fields);
      }
    }
    /**
     * Enrich a raw field block object with type data, then construct a FieldBlock.
     *
     * Ported from Python: TemplateLayout.parse_and_add_field_block + prefill_field_block
     */
    parseAndAddFieldBlock(blockName, fieldBlockObject) {
      const enriched = this.prefillFieldBlock(fieldBlockObject);
      const blockInstance = new FieldBlock(blockName, enriched, this.fieldBlocksOffset);
      this.fieldBlocks.push(blockInstance);
      this.validateParsedFieldBlock(fieldBlockObject, blockInstance);
      return blockInstance;
    }
    /**
     * Enrich a field block object with defaults from the template and the bubble field type data.
     *
     * Ported from Python: TemplateLayout.prefill_field_block
     */
    prefillFieldBlock(fieldBlockObject) {
      var _a, _b, _c;
      const fieldDetectionType = (_b = (_a = fieldBlockObject.fieldDetectionType) != null ? _a : fieldBlockObject.field_detection_type) != null ? _b : "BUBBLES_THRESHOLD";
      if (fieldDetectionType !== "BUBBLES_THRESHOLD") {
        return { ...fieldBlockObject };
      }
      const bubbleFieldType = (_c = fieldBlockObject.bubbleFieldType) != null ? _c : fieldBlockObject.bubble_field_type;
      const fieldTypeData = this.bubbleFieldTypesData[bubbleFieldType];
      return {
        bubble_field_type: bubbleFieldType,
        empty_value: this.globalEmptyVal,
        bubble_dimensions: this.bubbleDimensions,
        ...fieldBlockObject,
        ...fieldTypeData
      };
    }
    /**
     * Validate that the parsed field labels don't overlap with previously seen labels,
     * and that the bounding box doesn't overflow the template dimensions.
     *
     * Ported from Python: TemplateLayout.validate_parsed_field_block
     */
    validateParsedFieldBlock(fieldBlockObjectRaw, blockInstance) {
      var _a, _b;
      const { parsedFieldLabels, name: blockName } = blockInstance;
      const fieldLabelsSet = new Set(parsedFieldLabels);
      const overlap = [...fieldLabelsSet].filter((l) => this.allParsedLabels.has(l));
      if (overlap.length > 0) {
        const fieldLabels = (_b = (_a = fieldBlockObjectRaw.fieldLabels) != null ? _a : fieldBlockObjectRaw.field_labels) != null ? _b : [];
        throw new OMRCheckerError(
          `The field strings for field block ${blockName} overlap with other existing fields: ${overlap}`,
          { block_name: blockName, field_labels: fieldLabels, overlap }
        );
      }
      for (const label of fieldLabelsSet) {
        this.allParsedLabels.add(label);
      }
      const [pageWidth, pageHeight] = this.templateDimensions;
      const [blockWidth, blockHeight] = blockInstance.boundingBoxDimensions;
      const [blockStartX, blockStartY] = blockInstance.boundingBoxOrigin;
      const blockEndX = blockStartX + blockWidth;
      const blockEndY = blockStartY + blockHeight;
      if (blockEndX >= pageWidth || blockEndY >= pageHeight || blockStartX < 0 || blockStartY < 0) {
        throw new OMRCheckerError(
          `Overflowing field block '${blockName}' with origin ${blockInstance.boundingBoxOrigin} and dimensions ${blockInstance.boundingBoxDimensions} in template with dimensions ${this.templateDimensions}`,
          {
            block_name: blockName,
            bounding_box_origin: blockInstance.boundingBoxOrigin,
            bounding_box_dimensions: blockInstance.boundingBoxDimensions,
            template_dimensions: this.templateDimensions
          }
        );
      }
    }
    /**
     * Parse custom labels from the template (user-defined output groupings).
     *
     * Ported from Python: TemplateLayout.parse_custom_labels
     */
    parseCustomLabels(customLabelsObject) {
      const allParsedCustomLabels = /* @__PURE__ */ new Set();
      this.customLabels = {};
      for (const [customLabel, labelStrings] of Object.entries(customLabelsObject)) {
        const parsedLabels = parseFields(`Custom Label: ${customLabel}`, labelStrings);
        const parsedLabelsSet = new Set(parsedLabels);
        this.customLabels[customLabel] = parsedLabels;
        const missingCustomLabels = [...parsedLabelsSet].filter(
          (l) => !this.allParsedLabels.has(l)
        );
        if (missingCustomLabels.length > 0) {
          throw new OMRCheckerError(
            `Missing field block label(s) in the given template for ${missingCustomLabels} from '${customLabel}'`,
            { custom_label: customLabel, missing_labels: missingCustomLabels }
          );
        }
        const overlap = [...parsedLabelsSet].filter((l) => allParsedCustomLabels.has(l));
        if (overlap.length > 0) {
          throw new OMRCheckerError(
            `The field strings for custom label '${customLabel}' overlap with other existing custom labels`,
            { custom_label: customLabel, label_strings: labelStrings }
          );
        }
        for (const l of parsedLabelsSet) {
          allParsedCustomLabels.add(l);
        }
      }
      this.nonCustomLabels = new Set(
        [...this.allParsedLabels].filter((l) => !allParsedCustomLabels.has(l))
      );
    }
    /**
     * Build default output columns by sorting non-custom and custom label names alphanumerically.
     *
     * Ported from Python: TemplateLayout.fill_output_columns
     */
    buildDefaultOutputColumns() {
      const nonCustomColumns = [...this.nonCustomLabels];
      const allCustomColumns = Object.keys(this.customLabels);
      const allColumns = [...nonCustomColumns, ...allCustomColumns];
      return allColumns.sort((a, b) => a.localeCompare(b, void 0, { numeric: true }));
    }
    resetAllShifts() {
      for (const fieldBlock of this.fieldBlocks) {
        fieldBlock.resetAllShifts();
      }
    }
    /**
     * Construct a Template from a parsed JSON object.
     */
    static fromJSON(json) {
      return new _Template(json);
    }
    /**
     * Construct a Template from a JSON string.
     */
    static fromJSONString(jsonString) {
      const json = JSON.parse(jsonString);
      return new _Template(json);
    }
  };

  // src/detection/BubbleReader.ts
  var BubbleReader = class {
    constructor(userConfig = {}) {
      var _a, _b, _c, _d;
      this.config = {
        minJump: (_a = userConfig.minJump) != null ? _a : 30,
        minGapTwoBubbles: (_b = userConfig.minGapTwoBubbles) != null ? _b : 20,
        minJumpSurplusForGlobalFallback: (_c = userConfig.minJumpSurplusForGlobalFallback) != null ? _c : 10,
        globalFallbackThreshold: (_d = userConfig.globalFallbackThreshold) != null ? _d : 127.5
      };
    }
    /**
     * Process all fields in a template against a pre-processed grayscale image.
     *
     * Ported from Python:
     *   BubblesFieldDetection.run_detection  (pixel means)
     *   LocalThresholdStrategy.calculate_threshold
     *   BubblesFieldInterpretation.get_field_interpretation_string
     *
     * @param grayImage - Pre-processed grayscale cv.Mat (must not be deleted during call)
     * @param template  - Parsed Template (provides allFields with scanBoxes)
     * @returns OMRResponse mapping each fieldLabel → detected value string
     */
    readBubbles(grayImage, template) {
      const response = {};
      const { globalFallbackThreshold } = this.config;
      for (const field of template.allFields) {
        const bubbleMeans = [];
        for (const scanBox of field.scanBoxes) {
          const [w, h] = scanBox.bubbleDimensions;
          const [shiftedX, shiftedY] = scanBox.getShiftedPosition();
          const x = Math.round(shiftedX);
          const y = Math.round(shiftedY);
          const safeX = Math.max(0, Math.min(x, grayImage.cols - 1));
          const safeY = Math.max(0, Math.min(y, grayImage.rows - 1));
          const safeW = Math.min(w, grayImage.cols - safeX);
          const safeH = Math.min(h, grayImage.rows - safeY);
          if (safeW <= 0 || safeH <= 0) {
            bubbleMeans.push(255);
            continue;
          }
          const roi = grayImage.roi(new cv_shim_default.Rect(safeX, safeY, safeW, safeH));
          try {
            const meanVal = cv_shim_default.mean(roi)[0];
            bubbleMeans.push(meanVal);
          } finally {
            roi.delete();
          }
        }
        const threshold = this.localThreshold(bubbleMeans, globalFallbackThreshold);
        const markedBubbles = field.scanBoxes.filter((_sb, i) => bubbleMeans[i] < threshold);
        let fieldValue;
        if (markedBubbles.length === 0 || markedBubbles.length === field.scanBoxes.length) {
          fieldValue = field.emptyValue;
        } else {
          fieldValue = markedBubbles.map((sb) => sb.bubbleValue).join("");
        }
        response[field.fieldLabel] = fieldValue;
      }
      return response;
    }
    /**
     * Calculate the local threshold for a set of bubble mean values.
     *
     * Falls back to the global threshold when local confidence is too low.
     *
     * Ported from Python: LocalThresholdStrategy.calculate_threshold
     *
     * Algorithm:
     *   - 0 or 1 bubbles  → globalFallback
     *   - 2 bubbles       → midpoint if gap ≥ minGapTwoBubbles, else globalFallback
     *   - 3+ bubbles      → midpoint of largest inter-bubble jump;
     *                       use globalFallback when maxJump < minJump + minJumpSurplusForGlobalFallback
     *
     * @param bubbleMeans    - Mean pixel intensity per bubble (0–255)
     * @param globalFallback - Fallback threshold to use when local confidence is low
     * @returns Threshold value (bubble mean < threshold → bubble is marked)
     */
    localThreshold(bubbleMeans, globalFallback) {
      if (bubbleMeans.length < 2) {
        return globalFallback;
      }
      const sorted = [...bubbleMeans].sort((a, b) => a - b);
      if (sorted.length === 2) {
        const gap = sorted[1] - sorted[0];
        if (gap < this.config.minGapTwoBubbles) {
          return globalFallback;
        }
        return (sorted[0] + sorted[1]) / 2;
      }
      let maxJump = 0;
      let localThreshold = globalFallback;
      for (let i = 1; i < sorted.length - 1; i++) {
        const jump = sorted[i + 1] - sorted[i - 1];
        if (jump > maxJump) {
          maxJump = jump;
          localThreshold = sorted[i - 1] + jump / 2;
        }
      }
      const confidentJump = this.config.minJump + this.config.minJumpSurplusForGlobalFallback;
      if (maxJump < confidentJump) {
        return globalFallback;
      }
      return localThreshold;
    }
  };

  // src/OMRChecker.ts
  var OMRChecker = class _OMRChecker {
    /**
     * Process a single OMR sheet image and return the detected bubble responses.
     *
     * This is the primary entry point for the OMRChecker JS port, equivalent to
     * Python's process_single_file().
     *
     * @throws {Error} if image decoding, pre-processing, or bubble reading fails
     */
    static async processSingleFile(options) {
      var _a, _b, _c;
      const { imageBase64, templateJson, assets = {}, bubbleReaderConfig = {} } = options;
      const sourceGray = await _OMRChecker.decodeBase64ToGray(imageBase64);
      try {
        const template = Template.fromJSON(templateJson);
        let processedImage = sourceGray.clone();
        for (const ppConfig of template.preProcessorsConfig) {
          const ppOptions = (_a = ppConfig.options) != null ? _a : {};
          const ppShape = (_c = (_b = ppOptions.processingImageShape) != null ? _b : ppOptions.processing_image_shape) != null ? _c : void 0;
          if (ppShape) {
            const [ppH, ppW] = ppShape;
            const ppResized = new cv_shim_default.Mat();
            cv_shim_default.resize(processedImage, ppResized, new cv_shim_default.Size(ppW, ppH));
            processedImage.delete();
            processedImage = ppResized;
          }
          const next = await _OMRChecker.runPreprocessor(ppConfig, processedImage, assets);
          processedImage.delete();
          processedImage = next;
        }
        const [templateW, templateH] = template.templateDimensions;
        const resized = new cv_shim_default.Mat();
        cv_shim_default.resize(processedImage, resized, new cv_shim_default.Size(templateW, templateH));
        processedImage.delete();
        const reader = new BubbleReader(bubbleReaderConfig);
        const response = reader.readBubbles(resized, template);
        const dims = [resized.cols, resized.rows];
        resized.delete();
        return { response, processedImageDimensions: dims };
      } catch (err) {
        try {
          if (!sourceGray.isDeleted()) {
            sourceGray.delete();
          }
        } catch {
        }
        throw err;
      }
    }
    // ── Private helpers ───────────────────────────────────────────────────────────
    /**
     * Decode a base64-encoded image to a grayscale cv.Mat via the browser's
     * Image / canvas API (works without cv.imdecode / Node file I/O).
     *
     * @param base64 - Base64 string or data URL (data:image/…;base64,…)
     * @returns Promise<cv.Mat> grayscale image — caller must delete
     */
    static decodeBase64ToGray(base64) {
      const dataUrl = base64.startsWith("data:") ? base64 : `data:image/jpeg;base64,${base64}`;
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            reject(new Error("OMRChecker: could not create canvas 2D context"));
            return;
          }
          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, img.width, img.height);
          const rgba = cv_shim_default.matFromImageData(imageData);
          const gray = new cv_shim_default.Mat();
          cv_shim_default.cvtColor(rgba, gray, cv_shim_default.COLOR_RGBA2GRAY);
          rgba.delete();
          resolve(gray);
        };
        img.onerror = () => reject(new Error("OMRChecker: failed to load image from base64 data"));
        img.src = dataUrl;
      });
    }
    /**
     * Apply a single pre-processor by name to an image.
     *
     * Supported processors:
     *   - CropPage        (no async factory needed)
     *   - CropOnMarkers   (async factory; needs base64 asset for marker image)
     *
     * Unknown processors are silently skipped with a console.warn.
     *
     * @param ppConfig      - Pre-processor config from template.preProcessorsConfig
     * @param image         - Current grayscale cv.Mat (caller retains ownership)
     * @param assets        - Base64 assets map (filename → base64)
     * @returns New grayscale cv.Mat (caller must delete)
     */
    static async runPreprocessor(ppConfig, image, assets) {
      const { name, options } = ppConfig;
      if (name === "CropPage") {
        const cropPage = new CropPage(options);
        try {
          const [warped] = cropPage.applyFilter(image, null, null, "");
          return warped;
        } finally {
          cropPage.dispose();
        }
      }
      if (name === "CropOnMarkers") {
        const normalized = _OMRChecker.normalizeCropOnMarkersOptions(options);
        const cropOnMarkers = await CropOnMarkers.fromBase64(normalized, assets);
        try {
          const [warped] = cropOnMarkers.applyFilter(image, null, null, "");
          return warped;
        } finally {
          cropOnMarkers.dispose();
        }
      }
      console.warn(`OMRChecker: unknown pre-processor '${name}', skipping`);
      return image.clone();
    }
    /**
     * Normalize CropOnMarkers options from template.json (camelCase) to the
     * snake_case keys expected by the CropOnMarkers constructor.
     *
     * template.json uses:
     *   referenceImage, markerDimensions, tuningOptions.markerRescaleRange, etc.
     *
     * CropOnMarkers expects:
     *   reference_image, marker_dimensions, tuning_options.marker_rescale_range, etc.
     *
     * Falls back to already-snake_case keys when present (idempotent).
     */
    static normalizeCropOnMarkersOptions(options) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l;
      const type = (_a = options.type) != null ? _a : "FOUR_MARKERS";
      const referenceImage = (_c = (_b = options.reference_image) != null ? _b : options.referenceImage) != null ? _c : "";
      const markerDimensions = (_e = (_d = options.marker_dimensions) != null ? _d : options.markerDimensions) != null ? _e : void 0;
      const rawTuning = (_g = (_f = options.tuning_options) != null ? _f : options.tuningOptions) != null ? _g : {};
      const tuning_options = {
        warp_method: (_h = rawTuning.warp_method) != null ? _h : rawTuning.warpMethod,
        min_matching_threshold: (_i = rawTuning.min_matching_threshold) != null ? _i : rawTuning.minMatchingThreshold,
        marker_rescale_range: (_j = rawTuning.marker_rescale_range) != null ? _j : rawTuning.markerRescaleRange,
        marker_rescale_steps: (_k = rawTuning.marker_rescale_steps) != null ? _k : rawTuning.markerRescaleSteps,
        apply_erode_subtract: (_l = rawTuning.apply_erode_subtract) != null ? _l : rawTuning.applyErodeSubtract
      };
      for (const key of Object.keys(tuning_options)) {
        if (tuning_options[key] === void 0) {
          delete tuning_options[key];
        }
      }
      return {
        type,
        reference_image: referenceImage,
        ...markerDimensions !== void 0 ? { marker_dimensions: markerDimensions } : {},
        ...Object.keys(tuning_options).length > 0 ? { tuning_options } : {}
      };
    }
  };
  return __toCommonJS(index_exports);
})();
//# sourceMappingURL=omrchecker.iife.js.map
