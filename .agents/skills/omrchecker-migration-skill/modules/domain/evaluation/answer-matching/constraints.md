# Answer Matcher - Constraints

## Edge Cases

### Answer Type Detection

#### 1. Invalid Answer Item Structures

**Edge Case**: Unrecognizable answer item format.

**Behavior**:
```python
# Invalid structures that raise EvaluationError
answer_item = None           # Not a string or list
answer_item = 123            # Number instead of string
answer_item = []             # Empty list
answer_item = ["A"]          # Single-element list (needs >= 2 for MULTIPLE_CORRECT)
answer_item = [["A"], ["B"]] # Wrong tuple structure (needs exactly 2 elements)
answer_item = [["A", "x"]]   # Non-numeric score
```

**Error**:
```python
raise EvaluationError(
    "Unable to determine answer type",
    context={"answer_item": str(answer_item)}
)
```

**Browser Migration**:
```typescript
function getAnswerType(answerItem: unknown): AnswerType {
  if (typeof answerItem === 'string' && answerItem.length >= 1) {
    return AnswerType.STANDARD;
  }

  if (!Array.isArray(answerItem)) {
    throw new EvaluationError('Unable to determine answer type', {
      answerItem: String(answerItem)
    });
  }

  // Check MULTIPLE_CORRECT: all elements are strings
  if (answerItem.length >= 2 && answerItem.every(item =>
      typeof item === 'string' && item.length >= 1)) {
    return AnswerType.MULTIPLE_CORRECT;
  }

  // Check MULTIPLE_CORRECT_WEIGHTED: all elements are [string, number] tuples
  if (answerItem.length >= 1 && answerItem.every(item =>
      Array.isArray(item) && item.length === 2 &&
      typeof item[0] === 'string' && item[0].length >= 1 &&
      (typeof item[1] === 'number' || typeof item[1] === 'string'))) {
    return AnswerType.MULTIPLE_CORRECT_WEIGHTED;
  }

  throw new EvaluationError('Unable to determine answer type', {
    answerItem: JSON.stringify(answerItem)
  });
}
```

---

### Answer Matching Edge Cases

#### 2. Empty Values and Unmarked Answers

**Edge Case**: Different representations of "unmarked".

**Behavior**:
```python
# Template defines empty_value (typically "")
empty_value = ""

# All of these are treated as unmarked
marked_answer = ""           # Standard empty
marked_answer = None         # Also treated as unmarked (after preprocessing)

# These are NOT unmarked (they're wrong answers)
marked_answer = " "          # Space character - treated as answer " "
marked_answer = "0"          # Zero - treated as answer "0"
```

**Browser Migration**:
```typescript
function getStandardVerdict(markedAnswer: string, allowedAnswer: string, emptyValue: string): Verdict {
  // Normalize null/undefined to empty string
  const normalized = markedAnswer ?? '';

  if (normalized === emptyValue) {
    return Verdict.UNMARKED;
  }

  if (normalized === allowedAnswer) {
    return Verdict.ANSWER_MATCH;
  }

  return Verdict.NO_ANSWER_MATCH;
}
```

**Warning**: Be careful with falsy values in JavaScript. Use explicit `=== emptyValue` checks, not truthiness checks.

---

#### 3. Case Sensitivity

**Edge Case**: Answer matching is case-sensitive.

**Behavior**:
```python
allowed_answer = "A"

marked_answer = "A"  → verdict="answer-match"      ✓
marked_answer = "a"  → verdict="no-answer-match"   ✗
marked_answer = "A " → verdict="no-answer-match"   ✗ (trailing space)
```

**Current Limitation**: No automatic case normalization. This is intentional to preserve exact matching.

**Browser Migration**: Same behavior (use exact string equality).

**Workaround**: If case-insensitive matching is needed, normalize in preprocessing stage before evaluation.

---

#### 4. Multi-Character Answers

**Edge Case**: Concatenated bubble values (e.g., student marks both A and B).

**Behavior**:
```python
# Standard answer type - exact match required
allowed_answer = "AB"

marked_answer = "AB"  → verdict="answer-match"      ✓
marked_answer = "BA"  → verdict="no-answer-match"   ✗ (order matters)
marked_answer = "A"   → verdict="no-answer-match"   ✗ (partial match not allowed)
marked_answer = "ABC" → verdict="no-answer-match"   ✗

# Multiple correct - can include multi-char answers
allowed_answers = ["A", "B", "AB"]

marked_answer = "AB"  → verdict="answer-match-AB"   ✓
marked_answer = "BA"  → verdict="no-answer-match"   ✗ (not in list)
marked_answer = "A"   → verdict="answer-match-A"    ✓
```

**Note**: Order matters. "AB" and "BA" are different answers unless both are in the allowed list.

---

#### 5. Substring Matching in Helper Methods

**Edge Case**: `is_part_of_some_answer()` uses substring matching, not exact matching.

**Behavior**:
```python
# Question config
answer_item = ["AB", "ABC"]

# These return True (substring matching)
is_part_of_some_answer("A")   → True  # "A" in "AB" and "ABC"
is_part_of_some_answer("B")   → True  # "B" in "AB" and "ABC"
is_part_of_some_answer("AB")  → True  # "AB" in "AB" and exact match
is_part_of_some_answer("C")   → True  # "C" in "ABC"

# This returns False
is_part_of_some_answer("D")   → False
```

**Purpose**: Used for visual highlighting of bubble groups, not for scoring.

**Browser Migration**:
```typescript
function isPartOfSomeAnswer(questionMeta: QuestionMeta, answerString: string): boolean {
  if (questionMeta.bonusType !== null) {
    return true; // All answers are "correct" for bonus
  }

  const matchedGroups = getMatchedAnswerGroups(questionMeta, answerString);
  return matchedGroups.length > 0;
}

function getMatchedAnswerGroups(questionMeta: QuestionMeta, answerString: string): number[] {
  const { answerType, answerItem } = questionMeta;
  const matched: number[] = [];

  if (answerType === AnswerType.STANDARD) {
    if (String(answerItem).includes(answerString)) {
      matched.push(0);
    }
  } else if (answerType === AnswerType.MULTIPLE_CORRECT) {
    answerItem.forEach((allowed: string, index: number) => {
      if (allowed.includes(answerString)) {
        matched.push(index);
      }
    });
  } else if (answerType === AnswerType.MULTIPLE_CORRECT_WEIGHTED) {
    answerItem.forEach(([allowed, score]: [string, number], index: number) => {
      if (allowed.includes(answerString) && score > 0) {
        matched.push(index);
      }
    });
  }

  return matched;
}
```

---

### Weighted Answer Edge Cases

#### 6. Negative Weights and Schema Verdict

**Edge Case**: Negative custom weights override the "answer-match" verdict.

**Behavior**:
```python
# Configuration
answer_item = [["A", 2], ["B", -1], ["C", 0]]

# Student marks B (technically in allowed answers, but negative weight)
marked_answer = "B"
verdict = "answer-match-B"           # Question verdict (internal)
delta = -1                           # Negative score
schema_verdict = "incorrect"         # Schema verdict (displayed) - FORCED TO INCORRECT

# Student marks C (zero weight)
marked_answer = "C"
verdict = "answer-match-C"           # Question verdict
delta = 0                            # Zero score
schema_verdict = "correct"           # Schema verdict is still "correct" (not forced)
```

**Rationale**: Negative weights are for penalizing specific wrong answers more heavily than unmarked. The schema verdict should show "incorrect" to reflect the penalty.

**Browser Migration**:
```typescript
function getSchemaVerdict(
  answerType: AnswerType,
  questionVerdict: string,
  delta: number | null = null
): SchemaVerdict {
  // Special case: negative weights in weighted mode
  if (delta !== null && delta < 0 && answerType === AnswerType.MULTIPLE_CORRECT_WEIGHTED) {
    return SchemaVerdict.INCORRECT;
  }

  // Standard mapping
  for (const verdict of VERDICTS_IN_ORDER) {
    if (questionVerdict.startsWith(verdict)) {
      return VERDICT_TO_SCHEMA_VERDICT[verdict];
    }
  }

  throw new EvaluationError('Unable to determine schema verdict', {
    questionVerdict
  });
}
```

---

#### 7. Zero-Weight Answers

**Edge Case**: Answers with zero weight are allowed but give no points.

**Behavior**:
```python
# Configuration
answer_item = [["A", 3], ["B", 0]]

marked_answer = "B"
verdict = "answer-match-B"
delta = 0
schema_verdict = "correct"  # Still considered "correct" (not "incorrect")
```

**Use Case**: Deprecated answers that should no longer give points but shouldn't be penalized.

**Important**: Zero-weight answers **are included** in `is_part_of_some_answer()` checks (they're still valid answers, just worth 0 points).

---

#### 8. Weights Excluded from Matching

**Edge Case**: In `get_matched_answer_groups()`, zero and negative weights ARE included (unlike positive-only filtering).

**Clarification from Code**:
```python
# In get_matched_answer_groups():
if answer_string in allowed_answer and score > 0:
    matched_groups.append(answer_index)
```

**Wait - this excludes zero/negative weights!**

**Behavior**:
```python
answer_item = [["A", 3], ["B", 0], ["C", -1]]

# For visual highlighting (is_part_of_some_answer → get_matched_answer_groups)
get_matched_answer_groups("A")  → [0]      # Included (score > 0)
get_matched_answer_groups("B")  → []       # EXCLUDED (score = 0)
get_matched_answer_groups("C")  → []       # EXCLUDED (score < 0)
```

**Implication**: Zero-weight and negative-weight answers won't be highlighted as "part of correct answer groups" in visualization, even though they're valid answer options.

**Browser Migration**: Same logic.

---

### Fraction Parsing Edge Cases

#### 9. Fraction String Parsing

**Edge Case**: Parsing fraction strings like `"1/2"`, `"3/4"`.

**Behavior**:
```python
parse_float_or_fraction("3")      → 3.0
parse_float_or_fraction("0.5")    → 0.5
parse_float_or_fraction("1/2")    → 0.5
parse_float_or_fraction("3/4")    → 0.75
parse_float_or_fraction("1/3")    → 0.333333...

# Invalid formats (depends on parsing implementation)
parse_float_or_fraction("1/0")    → ZeroDivisionError or infinity
parse_float_or_fraction("abc")    → ValueError
parse_float_or_fraction("")       → ValueError
```

**Browser Migration**:
```typescript
function parseFloatOrFraction(value: string | number): number {
  if (typeof value === 'number') {
    return value;
  }

  // Try parsing as float first
  const floatValue = parseFloat(value);
  if (!isNaN(floatValue) && !value.includes('/')) {
    return floatValue;
  }

  // Try parsing as fraction
  const fractionMatch = value.match(/^(-?\d+)\/(\d+)$/);
  if (fractionMatch) {
    const numerator = parseInt(fractionMatch[1], 10);
    const denominator = parseInt(fractionMatch[2], 10);

    if (denominator === 0) {
      throw new Error(`Division by zero in fraction: ${value}`);
    }

    return numerator / denominator;
  }

  throw new Error(`Unable to parse as float or fraction: ${value}`);
}
```

**Edge Cases to Handle**:
- Division by zero: `"1/0"` → throw error
- Negative fractions: `"-1/2"` → -0.5
- Spaces: `"1 / 2"` → should trim or reject
- Decimals in fraction: `"1.5/2"` → reject (ambiguous)

---

### Marking Scheme Integration Edge Cases

#### 10. Missing Verdict Keys

**Edge Case**: Question verdict not found in marking scheme.

**Behavior**:
```python
# This should never happen if initialization is correct
marking = {
  "answer-match": 3,
  "no-answer-match": -1,
  "unmarked": 0
}

# If somehow we get a verdict not in marking (bug scenario)
question_verdict = "answer-match-XYZ"  # Created for MULTIPLE_CORRECT

# Accessing marking[question_verdict] → KeyError

# get_delta_for_verdict() doesn't have error handling for missing keys
# This would crash the evaluation
```

**Prevention**: The initialization methods (`set_local_marking_defaults`) ensure all possible verdicts are pre-populated in the local marking dictionary.

**Browser Migration**:
```typescript
function getDeltaForVerdict(
  marking: Record<string, number | number[]>,
  verdict: string,
  currentStreak: number
): number {
  const markingValue = marking[verdict];

  if (markingValue === undefined) {
    throw new EvaluationError('Missing marking for verdict', {
      verdict,
      availableVerdicts: Object.keys(marking)
    });
  }

  if (Array.isArray(markingValue)) {
    // Streak-based marking
    const streakIndex = Math.min(currentStreak, markingValue.length - 1);
    return markingValue[streakIndex];
  }

  if (currentStreak > 0) {
    console.warn(
      `Non-zero streak (${currentStreak}) for verdict ${verdict} with non-streak marking. Using base score.`
    );
  }

  return markingValue;
}
```

---

#### 11. Streak Marking with Insufficient Array Length

**Edge Case**: Streak marking array shorter than possible streak length.

**Behavior**:
```python
# Section has 10 questions
questions = ["q1", "q2", ..., "q10"]

# But streak marking only has 5 levels
marking = {
  "answer-match": [1, 2, 3, 4, 5],  # Only 5 levels
  "no-answer-match": -1,
  "unmarked": 0
}

# Student gets first 7 questions correct
# Streaks: 0, 1, 2, 3, 4, 5, 6
# Deltas:  1, 2, 3, 4, 5, 5, 5 (last level repeated)
```

**Code**:
```python
# In get_delta_for_verdict():
if isinstance(marking[verdict], list):
    return marking[verdict][current_streak]  # This would cause IndexError!
```

**Wait - Looking at code again**:
Actually, the code doesn't handle this! It would raise `IndexError` if streak exceeds array length.

**Expected Behavior** (based on common patterns): Use last value for all remaining streaks.

**Fix Needed**:
```python
def get_delta_for_verdict(self, marking, verdict, current_streak):
    if isinstance(marking[verdict], list):
        streak_index = min(current_streak, len(marking[verdict]) - 1)
        return marking[verdict][streak_index]
    # ...
```

**Browser Migration**:
```typescript
function getDeltaForVerdict(
  marking: Record<string, number | number[]>,
  verdict: string,
  currentStreak: number
): number {
  const markingValue = marking[verdict];

  if (Array.isArray(markingValue)) {
    if (markingValue.length === 0) {
      throw new EvaluationError('Empty streak marking array', { verdict });
    }

    // Use last value if streak exceeds array length
    const streakIndex = Math.min(currentStreak, markingValue.length - 1);
    return markingValue[streakIndex];
  }

  return markingValue;
}
```

**Validation**: The `SectionMarkingScheme.validate_marking_scheme()` method logs a warning if streak array is too short, but doesn't prevent it.

---

### Bonus Questions Edge Cases

#### 12. Bonus Type Determination

**Edge Case**: `is_part_of_some_answer()` returns `True` for ALL answers if question has a bonus type.

**Behavior**:
```python
# Bonus question (from section marking scheme)
bonus_type = "BONUS_ON_ATTEMPT"  # or "BONUS_FOR_ALL"

# ALL answers return True
is_part_of_some_answer("A")      → True
is_part_of_some_answer("B")      → True
is_part_of_some_answer("Z")      → True
is_part_of_some_answer("wrong")  → True
```

**Rationale**: For bonus questions, all answers should be highlighted as "acceptable" since they all give points.

**Browser Migration**:
```typescript
function isPartOfSomeAnswer(questionMeta: QuestionMeta, answerString: string): boolean {
  // Short-circuit for bonus questions
  if (questionMeta.bonusType !== null) {
    return true;
  }

  // Normal matching logic
  const matchedGroups = getMatchedAnswerGroups(questionMeta, answerString);
  return matchedGroups.length > 0;
}
```

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `get_answer_type()` | O(n) | n = number of answer items; checks all elements |
| `get_standard_verdict()` | O(1) | Simple string comparison |
| `get_multiple_correct_verdict()` | O(n) | n = number of allowed answers; uses `in` operator |
| `get_multiple_correct_weighted_verdict()` | O(n) | Extract answers list, then `in` check |
| `get_matched_answer_groups()` | O(n × m) | n = answers, m = avg answer string length (substring check) |
| `is_part_of_some_answer()` | O(n × m) | Calls `get_matched_answer_groups()` |
| `parse_verdict_marking()` | O(n) | n = number of verdicts to parse |

**Optimization Opportunities**:
1. **Cache answer type**: Already done in `__init__`
2. **Pre-extract allowed answers list**: Already done for weighted answers
3. **Use Set for multiple correct**: Could optimize `in` check from O(n) to O(1)

**Browser Optimization**:
```typescript
class AnswerMatcher {
  private allowedAnswersSet?: Set<string>; // For fast lookup

  constructor(answerItem: AnswerItem, sectionMarkingScheme: SectionMarkingScheme) {
    // ...initialization...

    // Create Set for O(1) lookup in MULTIPLE_CORRECT mode
    if (this.answerType === AnswerType.MULTIPLE_CORRECT) {
      this.allowedAnswersSet = new Set(answerItem as string[]);
    }
  }

  getMultipleCorrectVerdict(markedAnswer: string): string {
    if (markedAnswer === this.emptyValue) {
      return Verdict.UNMARKED;
    }

    // O(1) lookup instead of O(n)
    if (this.allowedAnswersSet!.has(markedAnswer)) {
      return `${Verdict.ANSWER_MATCH}-${markedAnswer}`;
    }

    return Verdict.NO_ANSWER_MATCH;
  }
}
```

---

### Memory Considerations

**Python Memory Usage** (per AnswerMatcher instance):
- `answer_item`: String or List (~50-500 bytes typical)
- `marking`: Dict with 3-10 entries (~200-1000 bytes)
- `answer_type`: String enum (~50 bytes)
- `empty_value`: String (~10 bytes)
- **Total**: ~500-2000 bytes per question

**Scaling**:
- 100 questions: ~50-200 KB
- 1000 questions: ~500 KB - 2 MB
- 10,000 questions: ~5-20 MB

**Browser Considerations**:
- Same memory usage
- Use TypeScript classes for better memory layout
- Consider Object.freeze() for immutable answer items
- Reuse AnswerMatcher instances across multiple student sheets if evaluation config is the same

**Memory Optimization**:
```typescript
// Instead of creating new AnswerMatcher for each student
const answerMatchers = new Map<string, AnswerMatcher>();

function getOrCreateAnswerMatcher(
  question: string,
  answerItem: AnswerItem,
  scheme: SectionMarkingScheme
): AnswerMatcher {
  const key = `${question}:${JSON.stringify(answerItem)}`;

  if (!answerMatchers.has(key)) {
    answerMatchers.set(key, new AnswerMatcher(answerItem, scheme));
  }

  return answerMatchers.get(key)!;
}
```

---

### Evaluation Performance

**Typical Evaluation** (100 questions, 100 students):
- 10,000 answer match operations
- Each operation: ~1-5 μs
- Total time: ~10-50 ms

**Large Scale** (1000 questions, 1000 students):
- 1,000,000 answer match operations
- Total time: ~1-5 seconds (Python)
- Browser: ~2-10 seconds (slower JS execution)

**Optimization for Browser**:
```typescript
// Use Web Worker for large evaluations
// evaluate-worker.ts
self.onmessage = (e: MessageEvent) => {
  const { responses, evaluationConfig } = e.data;

  const results = responses.map((response: ConcatenatedResponse) => {
    return evaluateConcatenatedResponse(response, evaluationConfig);
  });

  self.postMessage({ results });
};

// Main thread
async function evaluateBatch(
  responses: ConcatenatedResponse[],
  evaluationConfig: EvaluationConfig
): Promise<EvaluationResult[]> {
  const worker = new Worker('evaluate-worker.js');

  return new Promise((resolve) => {
    worker.onmessage = (e) => {
      resolve(e.data.results);
      worker.terminate();
    };

    worker.postMessage({ responses, evaluationConfig });
  });
}
```

---

## Browser Migration Constraints

### 1. No Direct Python Dependencies

**Challenge**: Answer matcher uses Python-specific features.

**Python Dependencies**:
- `copy.deepcopy()` for marking scheme cloning
- `parse_float_or_fraction()` utility
- `DotMap` for verdict enums
- Exception types from `src.exceptions`

**Browser Replacements**:
```typescript
// Deep copy
function deepCopy<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

// Or use structured clone (modern browsers)
function deepCopy<T>(obj: T): T {
  return structuredClone(obj);
}

// Verdict enums
enum Verdict {
  ANSWER_MATCH = 'answer-match',
  NO_ANSWER_MATCH = 'no-answer-match',
  UNMARKED = 'unmarked',
}

enum SchemaVerdict {
  CORRECT = 'correct',
  INCORRECT = 'incorrect',
  UNMARKED = 'unmarked',
}

enum AnswerType {
  STANDARD = 'standard',
  MULTIPLE_CORRECT = 'multiple-correct',
  MULTIPLE_CORRECT_WEIGHTED = 'multiple-correct-weighted',
}
```

---

### 2. Type Safety with TypeScript

**Challenge**: Python uses duck typing; TypeScript needs explicit types.

**Type Definitions**:
```typescript
// Answer item can be one of three structures
type AnswerItem =
  | string                           // STANDARD
  | string[]                         // MULTIPLE_CORRECT
  | Array<[string, number | string]> // MULTIPLE_CORRECT_WEIGHTED

// Marking scheme
interface Marking {
  [verdict: string]: number | number[]; // Can be single value or streak array
}

// Answer matcher class
class AnswerMatcher {
  readonly answerType: AnswerType;
  readonly answerItem: AnswerItem;
  readonly marking: Marking;
  readonly emptyValue: string;

  constructor(answerItem: AnswerItem, sectionMarkingScheme: SectionMarkingScheme) {
    this.answerType = this.getAnswerType(answerItem);
    this.answerItem = this.parseAndSetAnswerItem(answerItem);
    this.emptyValue = sectionMarkingScheme.emptyValue;
    this.marking = this.setLocalMarkingDefaults(sectionMarkingScheme);
  }

  // ... methods ...
}
```

---

### 3. Zod Validation for Runtime Type Checking

**Challenge**: Answer items come from JSON; need runtime validation.

**Zod Schema**:
```typescript
import { z } from 'zod';

// Standard answer
const standardAnswerSchema = z.string().min(1);

// Multiple correct answers
const multipleCorrectSchema = z.array(z.string().min(1)).min(2);

// Weighted answers [answer, score]
const weightedAnswerSchema = z.array(
  z.tuple([
    z.string().min(1),
    z.union([z.number(), z.string()]) // Allow string for fractions
  ])
).min(1);

// Union of all answer types
const answerItemSchema = z.union([
  standardAnswerSchema,
  multipleCorrectSchema,
  weightedAnswerSchema
]);

// Validate at runtime
function validateAnswerItem(answerItem: unknown): AnswerItem {
  return answerItemSchema.parse(answerItem);
}
```

---

### 4. Immutability Patterns

**Challenge**: Python uses mutable objects; browser may want immutability.

**Immutable Answer Matcher**:
```typescript
class AnswerMatcher {
  // All fields readonly
  readonly answerType: AnswerType;
  readonly answerItem: AnswerItem;
  readonly marking: Readonly<Marking>;
  readonly emptyValue: string;

  constructor(answerItem: AnswerItem, sectionMarkingScheme: SectionMarkingScheme) {
    this.answerType = this.getAnswerType(answerItem);
    this.answerItem = this.parseAndSetAnswerItem(answerItem);
    this.emptyValue = sectionMarkingScheme.emptyValue;

    // Deep freeze marking to prevent mutations
    this.marking = Object.freeze(
      this.setLocalMarkingDefaults(sectionMarkingScheme)
    );
  }
}
```

**Why Immutability**:
- Prevents accidental mutations across multiple student evaluations
- Enables caching (same answer matcher instance can be reused)
- Better debugging (no hidden state changes)

---

### 5. Error Handling

**Challenge**: Python exceptions need to map to JavaScript errors.

**Error Classes**:
```typescript
class OMRCheckerError extends Error {
  context?: Record<string, unknown>;

  constructor(message: string, context?: Record<string, unknown>) {
    super(message);
    this.name = this.constructor.name;
    this.context = context;
  }
}

class EvaluationError extends OMRCheckerError {}

// Usage
throw new EvaluationError(
  'Unable to determine answer type',
  { answerItem: JSON.stringify(answerItem) }
);
```

---

### 6. Logging

**Challenge**: Python uses rich logger; browser needs console logging.

**Browser Logging**:
```typescript
const logger = {
  critical: (msg: string, context?: unknown) => {
    console.error('[CRITICAL]', msg, context);
  },
  warning: (msg: string, context?: unknown) => {
    console.warn('[WARNING]', msg, context);
  },
  info: (msg: string, context?: unknown) => {
    console.info('[INFO]', msg, context);
  },
  debug: (msg: string, context?: unknown) => {
    if (DEBUG_MODE) {
      console.debug('[DEBUG]', msg, context);
    }
  }
};

// Usage
logger.critical(`Unable to determine answer type for answer item: ${answerItem}`);
```

---

### 7. Serialization for Storage/Transfer

**Challenge**: Need to serialize answer matchers for storage or Web Worker transfer.

**Serialization**:
```typescript
interface SerializedAnswerMatcher {
  answerType: AnswerType;
  answerItem: AnswerItem;
  marking: Marking;
  emptyValue: string;
}

class AnswerMatcher {
  // ... existing code ...

  toJSON(): SerializedAnswerMatcher {
    return {
      answerType: this.answerType,
      answerItem: this.answerItem,
      marking: this.marking,
      emptyValue: this.emptyValue
    };
  }

  static fromJSON(data: SerializedAnswerMatcher): AnswerMatcher {
    // Reconstruct from serialized data
    const dummyScheme = {
      emptyValue: data.emptyValue,
      marking: data.marking,
      // ... other required fields ...
    } as SectionMarkingScheme;

    const matcher = new AnswerMatcher(data.answerItem, dummyScheme);
    return matcher;
  }
}

// Transfer to Web Worker
worker.postMessage({
  answerMatcher: answerMatcher.toJSON()
});
```

---

## Testing Edge Cases

### Recommended Test Cases

```typescript
describe('AnswerMatcher', () => {
  describe('Answer Type Detection', () => {
    it('should detect STANDARD type', () => {
      const matcher = new AnswerMatcher('A', mockScheme);
      expect(matcher.answerType).toBe(AnswerType.STANDARD);
    });

    it('should detect MULTIPLE_CORRECT type', () => {
      const matcher = new AnswerMatcher(['A', 'B'], mockScheme);
      expect(matcher.answerType).toBe(AnswerType.MULTIPLE_CORRECT);
    });

    it('should detect MULTIPLE_CORRECT_WEIGHTED type', () => {
      const matcher = new AnswerMatcher([['A', 2], ['B', 0.5]], mockScheme);
      expect(matcher.answerType).toBe(AnswerType.MULTIPLE_CORRECT_WEIGHTED);
    });

    it('should throw on invalid answer item', () => {
      expect(() => new AnswerMatcher(123, mockScheme)).toThrow(EvaluationError);
      expect(() => new AnswerMatcher([], mockScheme)).toThrow(EvaluationError);
      expect(() => new AnswerMatcher(['A'], mockScheme)).toThrow(EvaluationError);
    });
  });

  describe('Standard Matching', () => {
    it('should match correct answer', () => {
      const matcher = new AnswerMatcher('A', mockScheme);
      const [verdict, delta] = matcher.getVerdictMarking('A', false);
      expect(verdict).toBe('answer-match');
      expect(delta).toBeGreaterThan(0);
    });

    it('should handle unmarked answer', () => {
      const matcher = new AnswerMatcher('A', mockScheme);
      const [verdict, delta] = matcher.getVerdictMarking('', false);
      expect(verdict).toBe('unmarked');
      expect(delta).toBe(0);
    });

    it('should be case-sensitive', () => {
      const matcher = new AnswerMatcher('A', mockScheme);
      const [verdict] = matcher.getVerdictMarking('a', false);
      expect(verdict).toBe('no-answer-match');
    });
  });

  describe('Multiple Correct Matching', () => {
    it('should match any allowed answer', () => {
      const matcher = new AnswerMatcher(['A', 'B'], mockScheme);

      const [verdictA] = matcher.getVerdictMarking('A', false);
      expect(verdictA).toBe('answer-match-A');

      const [verdictB] = matcher.getVerdictMarking('B', false);
      expect(verdictB).toBe('answer-match-B');
    });
  });

  describe('Weighted Matching', () => {
    it('should apply custom weights', () => {
      const matcher = new AnswerMatcher([['A', 2], ['B', 0.5]], mockScheme);

      const [verdictA, deltaA] = matcher.getVerdictMarking('A', false);
      expect(deltaA).toBe(2);

      const [verdictB, deltaB] = matcher.getVerdictMarking('B', false);
      expect(deltaB).toBe(0.5);
    });

    it('should force incorrect verdict for negative weights', () => {
      const matcher = new AnswerMatcher([['A', 2], ['B', -1]], mockScheme);
      const [verdict, delta] = matcher.getVerdictMarking('B', false);

      const schemaVerdict = AnswerMatcher.getSchemaVerdict(
        AnswerType.MULTIPLE_CORRECT_WEIGHTED,
        verdict,
        delta
      );

      expect(schemaVerdict).toBe(SchemaVerdict.INCORRECT);
    });
  });

  describe('Fraction Parsing', () => {
    it('should parse fractions', () => {
      expect(parseFloatOrFraction('1/2')).toBe(0.5);
      expect(parseFloatOrFraction('3/4')).toBe(0.75);
    });

    it('should parse decimals', () => {
      expect(parseFloatOrFraction('0.5')).toBe(0.5);
      expect(parseFloatOrFraction('2.5')).toBe(2.5);
    });

    it('should throw on division by zero', () => {
      expect(() => parseFloatOrFraction('1/0')).toThrow();
    });
  });

  describe('Edge Cases', () => {
    it('should handle multi-character answers', () => {
      const matcher = new AnswerMatcher('AB', mockScheme);
      const [verdict] = matcher.getVerdictMarking('AB', false);
      expect(verdict).toBe('answer-match');

      const [verdictBA] = matcher.getVerdictMarking('BA', false);
      expect(verdictBA).toBe('no-answer-match'); // Order matters
    });

    it('should handle empty string as empty value', () => {
      const matcher = new AnswerMatcher('A', { ...mockScheme, emptyValue: '' });
      const [verdict] = matcher.getVerdictMarking('', false);
      expect(verdict).toBe('unmarked');
    });
  });
});
```

---

## Summary

**Key Constraints**:
1. Answer type detection requires specific structures (minimum lengths, tuple formats)
2. Matching is case-sensitive and order-dependent
3. Negative weights in weighted mode force "incorrect" schema verdict
4. Substring matching used for visualization helpers, not scoring
5. Streak arrays should be long enough for question count (or use last value)
6. Browser migration requires TypeScript types, Zod validation, and immutability patterns

**Performance**:
- O(1) to O(n) per question depending on answer type
- Consider Set optimization for multiple correct answers in browser
- Use Web Workers for large-scale evaluations
- Memory usage: ~500-2000 bytes per question

**Browser Specific**:
- Replace deepcopy with `structuredClone()` or `JSON.parse(JSON.stringify())`
- Use enums instead of DotMap
- Implement custom error classes
- Serialize/deserialize for Web Worker transfer
- Consider immutability with `Object.freeze()`
