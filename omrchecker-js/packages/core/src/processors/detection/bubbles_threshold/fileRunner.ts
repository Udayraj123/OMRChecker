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

/**
 * File runner for bubbles threshold detection and interpretation.
 *
 * Instantiates BubblesThresholdDetectionPass and BubblesThresholdInterpretationPass
 * and coordinates them for processing bubble fields.
 */
export class BubblesThresholdFileRunner extends FieldTypeFileLevelRunner {
  constructor(tuningConfig: TuningConfig) {
    const detectionPass = new BubblesThresholdDetectionPass(tuningConfig);
    const interpretationPass = new BubblesThresholdInterpretationPass(tuningConfig);

    super(
      tuningConfig,
      FieldDetectionType.BUBBLES_THRESHOLD,
      detectionPass,
      interpretationPass
    );
  }
}

