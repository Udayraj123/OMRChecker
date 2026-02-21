// Auto-generated from Python dataclasses
// Do not edit manually - regenerate with 2-generate-interfaces.js

import type { MatLike } from './opencv';

/**
 * ProcessingContext
 * Source: processors/base.py
 */
export interface ProcessingContext {
  filePath: string;  // Python: file_path
  grayImage: MatLike;  // Python: gray_image
  coloredImage: MatLike;  // Python: colored_image
  omrResponse?: Record<string, string>;  // Python: omr_response
  isMultiMarked?: boolean;  // Python: is_multi_marked
  fieldIdToInterpretation?: Record<string, any>;  // Python: field_id_to_interpretation
  score?: number;
  evaluationMeta?: Record<string, any>;  // Python: evaluation_meta
  evaluationConfigForResponse?: any;  // Python: evaluation_config_for_response
  defaultAnswersSummary?: string;  // Python: default_answers_summary
  metadata?: Record<string, any>;
}

/**
 * BubbleMeanValue
 * Source: processors/detection/models/detection_results.py
 */
export interface BubbleMeanValue {
  meanValue: number;  // Python: mean_value
  unitBubble: any;  // BubblesScanBox - avoiding circular import | Python: unit_bubble
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
  timestamp?: Date | string;
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
  timestamp?: Date | string;
}

/**
 * BarcodeFieldDetectionResult
 * Source: processors/detection/models/detection_results.py
 */
export interface BarcodeFieldDetectionResult {
  fieldId: string;  // Python: field_id
  fieldLabel: string;  // Python: field_label
  detections: any[];
  timestamp?: Date | string;
}

/**
 * ThresholdResult
 * Source: processors/threshold/threshold_result.py
 */
export interface ThresholdResult {
  thresholdValue: number;  // Python: threshold_value
  confidence: number;  // 0.0 to 1.0
  maxJump: number;  // Python: max_jump
  methodUsed: string;  // Python: method_used
  fallbackUsed?: boolean;  // Python: fallback_used
  metadata?: Record<string, any>;
}

/**
 * ProcessorState
 * Source: processors/visualization/workflow_session.py
 */
export interface ProcessorState {
  name: string;  // Human-readable name of the processor
  order: number;
  timestamp: string;  // ISO format timestamp when processor executed
  durationMs: number;  // Time taken to execute in milliseconds | Python: duration_ms
  imageShape: number[];  // Python: image_shape
  grayImageBase64?: string | null;  // Base64-encoded JPEG of grayscale output image | Python: gray_image_base64
  coloredImageBase64?: string | null;  // Python: colored_image_base64
  metadata?: Record<string, any>;  // Additional processor-specific metadata
  success?: boolean;  // Whether the processor executed successfully
  errorMessage?: string | null;  // Error message if processor failed | Python: error_message
}

/**
 * WorkflowGraph
 * Source: processors/visualization/workflow_session.py
 */
export interface WorkflowGraph {
  nodes?: Record<string, any>[];  // List of node definitions with id, label, and metadata
  edges?: Record<string, any>[];  // List of edge definitions connecting nodes
}

/**
 * WorkflowSession
 * Source: processors/visualization/workflow_session.py
 */
export interface WorkflowSession {
  sessionId: string;  // Unique identifier for this session | Python: session_id
  filePath: string;  // Path to the input file being processed | Python: file_path
  templateName: string;  // Name of the template used | Python: template_name
  startTime: string;  // ISO format timestamp when session started | Python: start_time
  endTime?: string | null;  // ISO format timestamp when session ended | Python: end_time
  totalDurationMs?: number | null;  // Total execution time in milliseconds | Python: total_duration_ms
  processorStates?: ProcessorState[];  // List of processor states in execution order | Python: processor_states
  graph?: WorkflowGraph;  // Workflow graph structure
  config?: Record<string, any>;  // Configuration used for this session
  metadata?: Record<string, any>;  // Additional session metadata
}

/**
 * ProcessingResult
 * Result of processing an OMR image through the complete pipeline
 */
export interface ProcessingResult {
  filePath: string;  // Path to the processed file
  omrResponse: Record<string, string>;  // Field ID to detected value mapping
  isMultiMarked: boolean;  // Whether any multi-marked bubbles were detected
  score?: number;  // Total score (if evaluation is enabled)
  evaluationMeta?: Record<string, any>;  // Detailed evaluation results
  processingTime?: number;  // Time taken to process in milliseconds
  errors?: string[];  // Any errors encountered during processing
}
