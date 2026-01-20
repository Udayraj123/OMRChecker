/**
 * Bubbles threshold file runner.
 *
 * TypeScript port of src/processors/detection/bubbles_threshold/file_runner.py
 * Extends FieldTypeFileLevelRunner for bubble field processing.
 */

import { FieldDetectionType } from '../../constants';
import { FieldTypeFileLevelRunner } from '../base/fileRunner';
import { BubblesThresholdDetectionPass } from './detectionPass';
import { BubblesThresholdInterpretationPass } from './interpretationPass';
import type { TuningConfig } from '../base/commonPass';
import { DetectionRepository } from '../../repositories/DetectionRepository';

/**
 * File runner for bubbles threshold detection and interpretation.
 *
 * Instantiates BubblesThresholdDetectionPass and BubblesThresholdInterpretationPass
 * and coordinates them for processing bubble fields.
 */
export class BubblesThresholdFileRunner extends FieldTypeFileLevelRunner {
  public repository: DetectionRepository;

  constructor(tuningConfig: TuningConfig, repository: DetectionRepository) {
    const fieldDetectionType = FieldDetectionType.BUBBLES_THRESHOLD;
    const detectionPass = new BubblesThresholdDetectionPass(
      tuningConfig,
      fieldDetectionType,
      repository
    );
    const interpretationPass = new BubblesThresholdInterpretationPass(
      tuningConfig,
      fieldDetectionType,
      repository
    );

    super(tuningConfig, fieldDetectionType, detectionPass, interpretationPass);
    this.repository = repository;
  }
}

