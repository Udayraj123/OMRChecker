// Auto-generated from Python dataclasses
// Do not edit manually - regenerate with 2-generate-interfaces.js

/**
 * ProcessingContext
 * Source: processors/base.py
 */
export interface ProcessingContext {
  filePath: string | string;  // Python: file_path
  grayImage: MatLike;  // Python: gray_image
  coloredImage: MatLike;  // Python: colored_image
  omrResponse?: Record<[str, str], string>;  // Python: omr_response
  isMultiMarked?: boolean;  // Python: is_multi_marked
  fieldIdToInterpretation?: Record<[str, Any], string>;  // Python: field_id_to_interpretation
  score?: number;
  evaluationMeta?: Record<[str, Any], string>;  // Python: evaluation_meta
  evaluationConfigForResponse?: any;  // Python: evaluation_config_for_response
  defaultAnswersSummary?: string;  // Python: default_answers_summary
  metadata?: Record<[str, Any], string>;
}

/**
 * BubbleMeanValue
 * Source: processors/detection/models/detection_results.py
 */
export interface BubbleMeanValue {
  meanValue: number;  // Python: mean_value
  unitBubble: Any  # BubblesScanBox - avoiding circular import;  // Python: unit_bubble
  position?: [number, number];
}

/**
 * BubbleFieldDetectionResult
 * Source: processors/detection/models/detection_results.py
 */
export interface BubbleFieldDetectionResult {
  fieldId: string;  // Python: field_id
  fieldLabel: string;  // Python: field_label
  bubbleMeans: BubbleMeanValue[];  // Python: bubble_means
  timestamp?: datetime;
}

/**
 * OCRFieldDetectionResult
 * Source: processors/detection/models/detection_results.py
 */
export interface OCRFieldDetectionResult {
  fieldId: string;  // Python: field_id
  fieldLabel: string;  // Python: field_label
  detections: any[];
  confidence?: number;
  timestamp?: datetime;
}

/**
 * BarcodeFieldDetectionResult
 * Source: processors/detection/models/detection_results.py
 */
export interface BarcodeFieldDetectionResult {
  fieldId: string;  // Python: field_id
  fieldLabel: string;  // Python: field_label
  detections: any[];
  timestamp?: datetime;
}

/**
 * ThresholdResult
 * Source: processors/threshold/threshold_result.py
 */
export interface ThresholdResult {
  thresholdValue: number;  // Python: threshold_value
  confidence: float  # 0.0 to 1.0;
  maxJump: number;  // Python: max_jump
  methodUsed: string;  // Python: method_used
  fallbackUsed?: boolean;  // Python: fallback_used
  metadata?: Record;
}

/**
 * ProcessorState
 * Source: processors/visualization/workflow_session.py
 */
export interface ProcessorState {
  Attributes: name: Human-readable name of the processor;
  timestamp: ISO format timestamp when processor executed;
  durationMs: Time taken to execute in milliseconds;  // Python: duration_ms
  grayImageBase64: Base64-encoded JPEG of grayscale output image;  // Python: gray_image_base64
  metadata: Additional processor-specific metadata;
  success: Whether the processor executed successfully;
  errorMessage: Error message if processor failed;  // Python: error_message
  name: string;
  order: number;
  timestamp: string;
  durationMs: number;  // Python: duration_ms
  imageShape: [number, ...];  // Python: image_shape
  grayImageBase64?: string | null;  // Python: gray_image_base64
  coloredImageBase64?: string | null;  // Python: colored_image_base64
  metadata?: Record<[str, Any], string>;
  success?: boolean;
  errorMessage?: string | null;  // Python: error_message
}

/**
 * WorkflowGraph
 * Source: processors/visualization/workflow_session.py
 */
export interface WorkflowGraph {
  Attributes: nodes: List of node definitions with id, label, and metadata;
  edges: List of edge definitions connecting nodes;
  nodes?: Record<[str, Any], string>[];
  edges?: Record<[str, Any], string>[];
  Args: node_id: Unique identifier for the node;
  label: Display label for the node;
  metadata: Additional node metadata;
  Args: from_id: Source node ID;
  toId: Target node ID;  // Python: to_id
  label?: Optional edge label;
}

/**
 * WorkflowSession
 * Source: processors/visualization/workflow_session.py
 */
export interface WorkflowSession {
  Attributes: session_id: Unique identifier for this session;
  filePath: Path to the input file being processed;  // Python: file_path
  templateName: Name of the template used;  // Python: template_name
  startTime: ISO format timestamp when session started;  // Python: start_time
  endTime: ISO format timestamp when session ended;  // Python: end_time
  totalDurationMs: Total execution time in milliseconds;  // Python: total_duration_ms
  processorStates: List of processor states in execution order;  // Python: processor_states
  graph: Workflow graph structure;
  config: Configuration used for this session;
  metadata: Additional session metadata;
  sessionId: string;  // Python: session_id
  filePath: string;  // Python: file_path
  templateName: string;  // Python: template_name
  startTime: string;  // Python: start_time
  endTime?: string | null;  // Python: end_time
  totalDurationMs?: number | null;  // Python: total_duration_ms
  processorStates?: ProcessorState[];  // Python: processor_states
  graph?: WorkflowGraph;
  config?: Record<[str, Any], string>;
  metadata?: Record<[str, Any], string>;
  Args: state: ProcessorState to add;
  Args: end_time: ISO format timestamp when session ended;
  totalDurationMs: Total execution time in milliseconds;  // Python: total_duration_ms
  Args: indent: Number of spaces for JSON indentation;
  Returns: JSON string representation;
  Args: file_path: Path where to save the JSON file;
  Args: data: Dictionary containing session data;
  Returns: WorkflowSession instance;
  Args: json_str: JSON string containing session data;
  Returns: WorkflowSession instance;
  Args: file_path: Path to the JSON file;
  Returns: WorkflowSession instance;
}
