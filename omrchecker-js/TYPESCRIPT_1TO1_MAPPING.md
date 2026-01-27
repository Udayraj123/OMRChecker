# TypeScript Port - 1:1 File Mapping Structure

## Overview
This document describes the 1:1 correspondence between Python and TypeScript processor files after refactoring.

## Core Utilities

| Python File | TypeScript File | Test File |
|------------|----------------|-----------|
| `src/utils/json_conversion.py` | `packages/core/src/utils/jsonConversion.ts` | `packages/core/src/utils/__tests__/jsonConversion.test.ts` |

### JSON Conversion Utilities
These utilities provide:
- `camelCase` ↔ `snake_case` conversion for JSON keys
- Validation to detect clashing keys (both camelCase and snake_case versions of same key)
- Recursive dictionary key transformation

## Image Processors

### Basic Filters
| Python File | TypeScript File | Test File |
|------------|----------------|-----------|
| `src/processors/image/GaussianBlur.py` | `packages/core/src/processors/image/GaussianBlur.ts` | `packages/core/src/processors/image/__tests__/GaussianBlur.test.ts` |
| `src/processors/image/MedianBlur.py` | `packages/core/src/processors/image/MedianBlur.ts` | `packages/core/src/processors/image/__tests__/MedianBlur.test.ts` |
| `src/processors/image/Contrast.py` | `packages/core/src/processors/image/Contrast.ts` | `packages/core/src/processors/image/__tests__/Contrast.test.ts` |

### Advanced Processors
| Python File | TypeScript File | Test File |
|------------|----------------|-----------|
| `src/processors/image/AutoRotate.py` | `packages/core/src/processors/image/AutoRotate.ts` | `packages/core/src/processors/image/__tests__/AutoRotate.test.ts` |
| `src/processors/image/Levels.py` | `packages/core/src/processors/image/Levels.ts` | `packages/core/src/processors/image/__tests__/Levels.test.ts` |

## Threshold Strategies

**Note:** In Python, all threshold strategies are in a single file `src/processors/threshold/strategies.py`, but in TypeScript, they are split into separate files for better maintainability and 1:1 class-to-file mapping.

| Python Class (in strategies.py) | TypeScript File | Test File |
|--------------------------------|----------------|-----------|
| `GlobalThresholdStrategy` | `packages/core/src/processors/threshold/GlobalThreshold.ts` | `packages/core/src/processors/threshold/__tests__/GlobalThreshold.test.ts` |
| `LocalThresholdStrategy` | `packages/core/src/processors/threshold/LocalThreshold.ts` | `packages/core/src/processors/threshold/__tests__/LocalThreshold.test.ts` |
| `AdaptiveThresholdStrategy` | `packages/core/src/processors/threshold/AdaptiveThreshold.ts` | `packages/core/src/processors/threshold/__tests__/AdaptiveThreshold.test.ts` |

## Benefits of This Structure

1. **1:1 Correspondence**: Each Python processor class has a corresponding TypeScript file
2. **Easy Maintenance**: Changes in Python can be easily tracked to the corresponding TypeScript file
3. **Clear Testing**: Each processor has its own test file, making it easy to verify functionality
4. **Better IDE Support**: Smaller files are easier to navigate and understand
5. **Independent Development**: Processors can be developed and tested independently

## File Organization

```
omrchecker-js/packages/core/src/
├── processors/
│   ├── image/
│   │   ├── GaussianBlur.ts
│   │   ├── MedianBlur.ts
│   │   ├── Contrast.ts
│   │   ├── AutoRotate.ts
│   │   ├── Levels.ts
│   │   └── __tests__/
│   │       ├── GaussianBlur.test.ts
│   │       ├── MedianBlur.test.ts
│   │       ├── Contrast.test.ts
│   │       ├── AutoRotate.test.ts
│   │       └── Levels.test.ts
│   └── threshold/
│       ├── GlobalThreshold.ts
│       ├── LocalThreshold.ts
│       ├── AdaptiveThreshold.ts
│       └── __tests__/
│           ├── GlobalThreshold.test.ts
│           ├── LocalThreshold.test.ts
│           └── AdaptiveThreshold.test.ts
└── index.ts (exports all processors)
```

## Naming Conventions

- **Python**: `ClassNameInCamelCase.py` (e.g., `GaussianBlur.py`)
- **TypeScript**: `ClassNameInCamelCase.ts` (e.g., `GaussianBlur.ts`)
- **Tests**: `ClassNameInCamelCase.test.ts` (e.g., `GaussianBlur.test.ts`)

This maintains consistency with both codebases while ensuring clear 1:1 mapping for maintenance and review.

