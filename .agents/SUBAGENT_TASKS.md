# Subagent Task Assignments - Phase 1 Migration

**Created**: 2026-02-28  
**Phase**: Phase 1 - TypeScript Migration Pilot  
**Estimated Duration**: 10-14 hours with 4 parallel agents

---

## Overview

4 specialized subagents will work in parallel on Phase 1 migration tasks. Each agent has a clear set of files to migrate, with dependencies managed to prevent conflicts.

---

## Agent 1: Foundation-Alpha 🔷

**Specialization**: Core utility modules and checksums  
**Priority**: HIGH (other agents depend on these)  
**Estimated Time**: 2-3 hours

### Tasks

#### Task 1.1: Checksum Utility (✅ COMPLETED)
- **File**: `src/utils/checksum.py` → `omrchecker-js/packages/core/src/utils/checksum.ts`
- **Status**: ✅ Completed (test migration)
- **Score**: 80%
- **Notes**: Already migrated as proof-of-concept

#### Task 1.2: Math Utilities
- **File**: `src/utils/math.py` → `omrchecker-js/packages/core/src/utils/math.ts`
- **Description**: Mathematical utility functions (distance calculations, interpolation, array operations)
- **Dependencies**: None
- **Key Functions**:
  - `distance(point1, point2)` - Euclidean distance
  - `interpolate(start, end, factor)` - Linear interpolation
  - `clamp(value, min, max)` - Value clamping
- **Migration Notes**:
  - Pure functions, no OpenCV
  - Type: All use `number` or `[number, number]` for points
  - Edge cases: Division by zero, negative distances

#### Task 1.3: Geometry Utilities
- **File**: `src/utils/geometry.py` → `omrchecker-js/packages/core/src/utils/geometry.ts`
- **Description**: Geometric calculations (bounding boxes, polygons, intersections)
- **Dependencies**: `math.ts` (Task 1.2)
- **Key Functions**:
  - `getBoundingRect(points)` - Min bounding rectangle
  - `getPolygonArea(points)` - Area calculation
  - `doPolygonsIntersect(poly1, poly2)` - Intersection test
- **Migration Notes**:
  - Depends on `distance()` from math.ts
  - Point type: `type Point = [number, number]`
  - Return rectangles as `[x, y, width, height]`

#### Task 1.4: Stats Utilities
- **File**: `src/utils/stats.py` → `omrchecker-js/packages/core/src/utils/stats.ts`
- **Description**: Statistical functions (mean, median, std deviation)
- **Dependencies**: None
- **Key Functions**:
  - `mean(values)` - Average
  - `median(values)` - Middle value
  - `stdDev(values)` - Standard deviation
- **Migration Notes**:
  - Pure math, no dependencies
  - Handle empty arrays gracefully
  - Use TypeScript generics for numeric arrays

**Deliverables**:
- 4 TypeScript files (1 already done)
- Validation scores ≥80% for each
- Unit tests for pure functions
- Commit after each file

---

## Agent 2: Schema-Beta 🔶

**Specialization**: Pydantic schemas and data models  
**Priority**: HIGH (required for template loading)  
**Estimated Time**: 3-4 hours

### Tasks

#### Task 2.1: Template Schema
- **File**: `src/schemas/models/template.py` → `omrchecker-js/packages/core/src/schemas/template.ts`
- **Description**: Template.json structure validation (Pydantic → Zod/TypeScript interfaces)
- **Dependencies**: None (pure schema)
- **Key Models**:
  - `TemplateModel` - Root template structure
  - `FieldBlockModel` - Field block definition
  - `FieldModel` - Individual field specs
  - `PreprocessorModel` - Image preprocessing config
- **Migration Notes**:
  - Convert Pydantic validators → Zod schemas OR TypeScript interfaces
  - Keep field name mappings (snake_case → camelCase)
  - Preserve default values
  - Document all validation rules
- **Special Attention**:
  - Enum types: `QType`, `PreprocessorType`
  - Optional fields with defaults
  - Nested object validation

#### Task 2.2: Config Schema
- **File**: `src/schemas/models/config.py` → `omrchecker-js/packages/core/src/schemas/config.ts`
- **Description**: Config.json structure (tuning parameters)
- **Dependencies**: None (pure schema)
- **Key Models**:
  - `ConfigModel` - Root config
  - `ThresholdConfig` - Thresholding parameters
  - `AlignmentConfig` - Alignment settings
  - `DetectionConfig` - Detection tuning
- **Migration Notes**:
  - Numeric ranges: Add min/max validation
  - Boolean flags: Document behavior
  - Optional vs required fields
  - Default value hierarchy

#### Task 2.3: Evaluation Schema
- **File**: `src/schemas/models/evaluation.py` → `omrchecker-js/packages/core/src/schemas/evaluation.ts`
- **Description**: Evaluation.json structure (answer keys, scoring)
- **Dependencies**: Template schema (Task 2.1)
- **Key Models**:
  - `EvaluationModel` - Root evaluation
  - `AnswerKeyModel` - Correct answers
  - `ScoringRuleModel` - Point allocation
  - `GroupingModel` - Question grouping
- **Migration Notes**:
  - Answer key format: Array or dict
  - Scoring rules: Positive/negative marks
  - Multi-mark handling
  - Custom evaluation functions

**Deliverables**:
- 3 schema/interface files
- Validation examples for each
- Type guards if using interfaces
- Integration tests with sample JSON files

---

## Agent 3: Image-Gamma 🔵

**Specialization**: Image processing and OpenCV operations  
**Priority**: MEDIUM (needed for preprocessing)  
**Estimated Time**: 4-5 hours

### Tasks

#### Task 3.1: Image Utils
- **File**: `src/utils/image.py` → `omrchecker-js/packages/core/src/utils/image.ts`
- **Description**: Core image operations (resize, rotate, convert)
- **Dependencies**: None (uses cv directly)
- **Key Functions**:
  - `resizeImage(img, width, height)` - Resize with interpolation
  - `rotateImage(img, angle)` - Rotation with border handling
  - `convertColorSpace(img, code)` - Color conversion
  - `thresholdImage(img, thresh, type)` - Thresholding
- **Migration Notes**:
  - Python: `cv2.resize()` → TypeScript: `cv.resize()`
  - **Memory management**: Wrap all cv.Mat operations in try/finally
  - Use `cv.Size` constructor for dimensions
  - Clean up intermediate Mat objects
- **OpenCV.js Patterns**:
  ```typescript
  const result = new cv.Mat();
  try {
    cv.resize(src, result, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR);
    return result.clone();
  } finally {
    result.delete();
  }
  ```

#### Task 3.2: Image Warp
- **File**: `src/utils/image_warp.py` → `omrchecker-js/packages/core/src/utils/imageWarp.ts`
- **Description**: Geometric transformations (warp perspective, homography)
- **Dependencies**: `image.ts` (Task 3.1), `geometry.ts` (Agent 1)
- **Key Functions**:
  - `warpPerspective(img, M, size)` - Apply homography
  - `getPerspectiveTransform(src, dst)` - Compute transform matrix
  - `applyHomography(img, H)` - Apply transformation
- **Migration Notes**:
  - Matrix handling: cv.Mat for 3x3 matrices
  - Point ordering: Ensure clockwise/counter-clockwise consistency
  - Border modes: cv.BORDER_CONSTANT, cv.BORDER_REPLICATE
  - Memory: Delete transformation matrices

#### Task 3.3: Drawing Utils
- **File**: `src/utils/drawing.py` → `omrchecker-js/packages/core/src/utils/drawing.ts`
- **Description**: Visualization utilities (draw rectangles, circles, text)
- **Dependencies**: `image.ts` (Task 3.1)
- **Key Functions**:
  - `drawRectangle(img, rect, color, thickness)` - Draw bbox
  - `drawCircle(img, center, radius, color)` - Draw circle
  - `drawText(img, text, position, font, color)` - Render text
  - `drawContours(img, contours, color)` - Draw shapes
- **Migration Notes**:
  - Color format: cv.Scalar for RGBA
  - Thickness: -1 for filled shapes
  - Font: cv.FONT_HERSHEY_SIMPLEX
  - Text positioning: Bottom-left corner

**Deliverables**:
- 3 TypeScript files with OpenCV operations
- Memory leak tests (check Mat cleanup)
- Visual validation with sample images
- Browser compatibility checks

---

## Agent 4: Processor-Delta 🔴

**Specialization**: Processing pipeline and base classes  
**Priority**: HIGH (foundation for all processors)  
**Estimated Time**: 3-4 hours

### Tasks

#### Task 4.1: Base Processor
- **File**: `src/processors/base.py` → `omrchecker-js/packages/core/src/processors/base.ts`
- **Description**: Abstract processor interface and context
- **Dependencies**: None (interface definition)
- **Key Classes**:
  - `Processor` (abstract) - Base processor interface
    - `getName()` - Processor identification
    - `process(context)` - Main processing method
  - `ProcessingContext` (interface) - State container
- **Migration Notes**:
  - Use TypeScript abstract class for Processor
  - ProcessingContext as interface (not class)
  - Preserve immutability patterns
  - Add type parameters for extensibility
- **Interface**:
  ```typescript
  export abstract class Processor {
    abstract getName(): string;
    abstract process(context: ProcessingContext): Promise<ProcessingContext>;
  }
  
  export interface ProcessingContext {
    originalImage: cv.Mat;
    processedImage: cv.Mat;
    metadata: Record<string, any>;
    // ... more fields
  }
  ```

#### Task 4.2: Processing Pipeline
- **File**: `src/processors/pipeline.py` → `omrchecker-js/packages/core/src/processors/Pipeline.ts`
- **Description**: Pipeline orchestrator (chains processors)
- **Dependencies**: `base.ts` (Task 4.1)
- **Key Classes**:
  - `ProcessingPipeline` - Main orchestrator
    - `addProcessor(processor)` - Register processor
    - `removeProcessor(name)` - Unregister
    - `processFile(imagePath)` - Execute pipeline
    - `getProcessorNames()` - List registered
- **Migration Notes**:
  - Support both sync and async processors
  - Error handling: Try-catch per processor
  - Context cloning: Deep copy between stages
  - Processor ordering: Maintain insertion order
  - Logging: Console.log processor names and timings

#### Task 4.3: Preprocessing Coordinator
- **File**: `src/processors/image/coordinator.py` → `omrchecker-js/packages/core/src/processors/image/coordinator.ts`
- **Description**: Coordinates image preprocessing steps
- **Dependencies**: `base.ts`, `Pipeline.ts`, image utils (Agent 3)
- **Key Classes**:
  - `PreprocessingProcessor` - Coordinates preprocessors
    - Runs: Rotation → Crop → Filters in sequence
    - Handles config-driven preprocessing
- **Migration Notes**:
  - Load preprocessor config from template
  - Skip InteractionUtils.show() (browser mode)
  - Preserve intermediate images for debugging
  - Memory management: Clean up after each step

**Deliverables**:
- 3 TypeScript files (base, pipeline, coordinator)
- Integration tests with mock processors
- End-to-end pipeline test
- Performance benchmarks

---

## Coordination & Communication

### Agent Sync Points

**Daily Stand-ups** (async):
- Update `.migration-tasks.jsonl` after each completion
- Report blockers in shared channel
- Share learned patterns (type mappings, OpenCV tricks)

**Dependency Order**:
1. **Start immediately**: Agent 1 (utils), Agent 2 (schemas), Agent 4 (base/pipeline)
2. **After Agent 1 completes geometry**: Agent 3 starts image warp
3. **After Agent 3 completes image.ts**: Agent 4 starts coordinator
4. **After Agent 4 completes base.ts**: Agent 3 can use Processor interface

### Communication Protocol

**Use migration_tasks.py**:
```bash
# Get your next task
TASK=$(uv run scripts/migration_tasks.py next | jq -r '.id')

# Claim it with your agent name
uv run scripts/migration_tasks.py claim $TASK --agent <your-name>

# Complete when done
uv run scripts/migration_tasks.py complete $TASK --score <validation-score>
```

**Update after each file**:
```bash
git commit -m "feat(ts-migrate): <task-id> - <file-name> (<agent-name>)"
git push
```

**Report issues**:
- Create issue in `.migration-tasks.jsonl` notes
- Tag dependent tasks
- Share in team channel

---

## Success Criteria

### Per-Agent Goals

- ✅ All assigned files migrated
- ✅ Validation scores ≥80% average
- ✅ Zero TypeScript compilation errors
- ✅ Memory leaks addressed (for OpenCV code)
- ✅ Tests passing for pure functions
- ✅ Git commits follow conventions

### Team Goals

- ✅ 12 files migrated in 10-14 hours
- ✅ No merge conflicts
- ✅ Documentation updated (FILE_MAPPING.json)
- ✅ Integration tests pass
- ✅ Browser demo still works

---

## Agent Profiles

### Foundation-Alpha 🔷
**Strengths**: Pure functions, mathematical operations  
**Style**: Test-driven, comprehensive edge cases  
**Focus**: Type safety, zero dependencies

### Schema-Beta 🔶
**Strengths**: Data modeling, validation  
**Style**: Schema-first design  
**Focus**: Type correctness, validation coverage

### Image-Gamma 🔵
**Strengths**: OpenCV, visual algorithms  
**Style**: Performance-conscious, memory-aware  
**Focus**: Browser compatibility, memory management

### Processor-Delta 🔴
**Strengths**: Architecture, interfaces  
**Style**: Clean abstractions, extensibility  
**Focus**: Pipeline patterns, error handling

---

## Quick Start for Each Agent

1. **Read the migration skill**: `.agents/skills/python-to-typescript-migration/SKILL.md`
2. **Read your task section above**
3. **Initialize your tasks**:
   ```bash
   uv run scripts/migration_tasks.py init --phase 1
   ```
4. **Get your first task**:
   ```bash
   uv run scripts/migration_tasks.py next
   ```
5. **Follow the 9-step workflow** in the skill document
6. **Report progress** via `migration_tasks.py complete`

---

**Status**: Ready for Agent Deployment  
**Last Updated**: 2026-02-28  
**Coordinator**: Human (you) + Oz (me, if needed)
