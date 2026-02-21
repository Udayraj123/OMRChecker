# Section Marking Scheme Constraints

**Module**: Domain - Evaluation - Section Marking
**Python Reference**: `src/processors/evaluation/section_marking_scheme.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### section_key

**Type**: `str`
**Special Value**: `"DEFAULT"` (required, case-sensitive)

**Valid Examples**:
```python
"DEFAULT"                    # Required default scheme
"BONUS_SECTION"             # Custom section with BONUS_ prefix
"HARD_QUESTIONS"            # Custom section
"SECTION_A"                 # Custom section
"parent-SECTION_A"          # Inherited from parent (auto-generated)
```

**Constraints**:
- Must be unique within evaluation config
- `"DEFAULT"` is mandatory, others optional
- Custom sections must define `questions` array
- Prefix `"BONUS_"` triggers positive-marking validation warnings
- Prefix `"parent-"` indicates inherited section (from conditional sets)

**Invalid**:
```python
""                          # Empty string not allowed
"default"                   # Case-sensitive, must be "DEFAULT"
"DEFAULT "                  # No trailing spaces
```

### section_scheme

**Type**: `dict`
**Structure**: Varies by section type

**For DEFAULT Section**:
```python
{
    "correct": "3",              # Required
    "incorrect": "-1",           # Required
    "unmarked": "0"              # Required
}
```

**For Custom Section**:
```python
{
    "marking": {
        "correct": "3",
        "incorrect": "-1",
        "unmarked": "0"
    },
    "questions": ["q1..10"],     # Required
    "markingType": "default"     # Optional, default: "default"
}
```

**Constraints**:
- DEFAULT section: marking at top level (shorthand)
- Custom sections: marking nested under `"marking"` key
- Must include all three verdicts: correct, incorrect, unmarked
- `markingType` must be one of: `"default"`, `"verdict_level_streak"`, `"section_level_streak"`
- `questions` required for non-DEFAULT sections

### set_name

**Type**: `str`
**Default**: `"DEFAULT_SET"`

**Valid Examples**:
```python
"DEFAULT_SET"               # Default set
"SET_A"                     # Conditional set A
"SET_B"                     # Conditional set B
"MATH_SET"                  # Custom set name
```

**Constraints**:
- Must be unique within evaluation config
- Used for tracking which set was matched
- No special characters validation (accepts any string)

### empty_value

**Type**: `str`
**Source**: `template.global_empty_val`

**Valid Examples**:
```python
""                          # Empty string (most common)
"0"                         # Zero character
"_"                         # Underscore
"X"                         # Any single character
```

**Constraints**:
- Defines what constitutes an "unmarked" answer
- Must match template's empty value
- Used to identify UNMARKED verdict
- Case-sensitive comparison

---

## Marking Score Constraints

### Marking Score Format

**Valid Formats**:
```python
# 1. Integer
"3"                         # Positive integer
"-1"                        # Negative integer
"0"                         # Zero

# 2. Float
3.0                         # Positive float
-1.5                        # Negative float
0.0                         # Zero float

# 3. Fraction (string)
"1/2"                       # 0.5
"2/3"                       # 0.666...
"3/4"                       # 0.75
"-1/2"                      # -0.5

# 4. Streak Array
[3, 5, 7, 10]               # Integer array
[1.0, 2.0, 3.0]             # Float array
["1/2", "3/4", "1"]         # Fraction array (mixed)
```

**Parsed Values**:
```python
AnswerMatcher.parse_verdict_marking("3")      # → 3.0
AnswerMatcher.parse_verdict_marking("-1")     # → -1.0
AnswerMatcher.parse_verdict_marking("1/2")    # → 0.5
AnswerMatcher.parse_verdict_marking([3, 5])   # → [3.0, 5.0]
```

**Constraints**:
- Strings must match regex: `^-?(\d+)(\/(\d+))?$`
- Fractions: denominator cannot be zero (validated at parse time)
- Arrays: all elements must be valid marking scores
- Arrays: minimum length 1, no maximum enforced
- Range: typically -10 to +10, but no hard limits

### Marking Type Constraints

**Valid Types**:
```python
MarkingSchemeType.DEFAULT                    # "default"
MarkingSchemeType.VERDICT_LEVEL_STREAK      # "verdict_level_streak"
MarkingSchemeType.SECTION_LEVEL_STREAK      # "section_level_streak"
```

**Type-Specific Constraints**:

| Type | Marking Format | Streak Tracking | Questions Required |
|------|---------------|-----------------|-------------------|
| DEFAULT | Single value | None | Optional |
| VERDICT_LEVEL_STREAK | Single value OR array | Per verdict | Required |
| SECTION_LEVEL_STREAK | Single value OR array | Section-wide | Required |

**Validation Rules**:
```python
# DEFAULT marking type
If markingType == "default":
    - Can use single values only (arrays allowed but not recommended)
    - No streak tracking
    - Questions optional (DEFAULT section)

# VERDICT_LEVEL_STREAK
If markingType == "verdict_level_streak":
    - Can use arrays for streak bonuses
    - Tracks separate streaks for correct/incorrect/unmarked
    - Questions required (to validate array lengths)

# SECTION_LEVEL_STREAK
If markingType == "section_level_streak":
    - Can use arrays for streak bonuses
    - Tracks single section-wide streak
    - Questions required
```

---

## Questions Constraints

### questions Array

**Type**: `list[str] | None`
**For DEFAULT Section**: `None`
**For Custom Sections**: Required

**Valid Examples**:
```python
# DEFAULT section
questions = None

# Custom sections
["q1", "q2", "q3"]                          # Explicit list
parse_fields("q1..10")  # → ["q1", ..., "q10"]
parse_fields(["q5", "q7", "q9"])  # → ["q5", "q7", "q9"]
```

**Constraints**:
- Must be unique across all sections in same evaluation config
- No question can appear in multiple sections
- Questions must exist in template field blocks
- Question names must match template exactly (case-sensitive)
- Minimum: 1 question per custom section
- Maximum: No limit (but typically < 200 per section)

**Validation**:
```python
# Check for overlaps
section_questions = set()
for section in sections:
    current_set = set(section.questions)
    if not section_questions.isdisjoint(current_set):
        raise FieldDefinitionError("Overlapping questions")
    section_questions = section_questions.union(current_set)

# Check for missing questions
all_questions = set(questions_in_order)
missing = sorted(section_questions.difference(all_questions))
if len(missing) > 0:
    raise EvaluationError("Missing answer key")
```

---

## Streak Constraints

### Streak Array Length

**Constraint**: Array length should match or exceed number of questions in section

**Validation Logic**:
```python
If marking_type in {VERDICT_LEVEL_STREAK, SECTION_LEVEL_STREAK}:
    For each verdict with array marking:
        If len(array) < len(questions):
            logger.warning(
                f"Verdict '{verdict}' has {len(array)} streak levels "
                f"but section has {len(questions)} questions"
            )
```

**Behavior When Array Too Short**:
```python
# If streak exceeds array length, use last value
marking_array = [3, 5, 7]
questions = 10

# Question streaks:
# q1 (streak=0): marking_array[0] = 3
# q2 (streak=1): marking_array[1] = 5
# q3 (streak=2): marking_array[2] = 7
# q4 (streak=3): marking_array[2] = 7  # Reuses last value
# q5 (streak=4): marking_array[2] = 7  # Reuses last value
# ... all remaining use last value
```

**Recommended Array Lengths**:
```python
# Conservative (exact match)
len(streak_array) == len(questions)

# Safe (allow some buffer)
len(streak_array) >= len(questions)

# Minimal (assume most students won't get all correct)
len(streak_array) >= len(questions) * 0.7

# Practical example for 10 questions
[3, 5, 7, 10, 12, 15, 18, 20, 22, 25]  # 10 levels (exact)
[3, 5, 7, 10, 12]                       # 5 levels (assumes max 5 streak)
```

### Streak Counter Constraints

**VERDICT_LEVEL_STREAK**:
```python
# State
streaks = {
    "correct": int,      # Range: 0 to len(questions)
    "incorrect": int,    # Range: 0 to len(questions)
    "unmarked": int      # Range: 0 to len(questions)
}

# Constraints
- All counters reset when different verdict occurs
- Only current verdict counter increments
- Unmarked verdict: streak increments only if allow_streak=True
- Maximum streak: len(questions) in section
```

**SECTION_LEVEL_STREAK**:
```python
# State
section_level_streak: int              # Range: 0 to len(questions)
previous_streak_verdict: SchemaVerdict | None

# Constraints
- Counter increments only if same verdict as previous
- First question: counter increments regardless
- Verdict change: counter resets to 0
- Maximum streak: len(questions) in section
```

---

## Bonus Type Constraints

### Bonus Type Detection

**Possible Values**:
```python
None                        # Not a bonus section
"BONUS_FOR_ALL"            # Bonus for all answers (marked or unmarked)
"BONUS_ON_ATTEMPT"         # Bonus only for attempted answers
```

**Detection Logic**:
```python
def get_bonus_type():
    # Rule 1: Streak marking → not bonus
    if marking_type == VERDICT_LEVEL_STREAK:
        return None

    # Rule 2: Negative/zero incorrect → not bonus
    if marking[NO_ANSWER_MATCH] <= 0:
        return None

    # Rule 3: Positive incorrect + positive unmarked → BONUS_FOR_ALL
    if marking[UNMARKED] > 0:
        return "BONUS_FOR_ALL"

    # Rule 4: Positive incorrect + zero unmarked → BONUS_ON_ATTEMPT
    if marking[UNMARKED] == 0:
        return "BONUS_ON_ATTEMPT"

    # Rule 5: Positive incorrect + negative unmarked → not bonus
    return None
```

**Constraint Table**:

| incorrect | unmarked | Bonus Type | Description |
|-----------|----------|------------|-------------|
| ≤ 0 | Any | None | Standard (negative marking) |
| > 0 | > 0 | BONUS_FOR_ALL | Everyone gets marks |
| > 0 | = 0 | BONUS_ON_ATTEMPT | Marks for trying |
| > 0 | < 0 | None | Penalty for leaving blank |

**Examples**:
```python
# Standard marking (not bonus)
{"correct": 3, "incorrect": -1, "unmarked": 0}
→ None

# BONUS_FOR_ALL
{"correct": 3, "incorrect": 3, "unmarked": 3}
→ "BONUS_FOR_ALL"

# BONUS_ON_ATTEMPT
{"correct": 3, "incorrect": 3, "unmarked": 0}
→ "BONUS_ON_ATTEMPT"

# Penalty for blank (not bonus)
{"correct": 3, "incorrect": 1, "unmarked": -1}
→ None
```

---

## Validation Constraints

### Positive Marks for Incorrect Warning

**Trigger Condition**:
```python
if (marking_type == DEFAULT and
    schema_verdict_marking > 0 and
    schema_verdict == "incorrect" and
    not section_key.startswith("BONUS_")):
    logger.warning(
        f"Found positive marks({schema_verdict_marking}) for incorrect answer "
        f"in schema '{section_key}'. For Bonus sections, prefer adding "
        f"a prefix 'BONUS_' to the scheme name."
    )
```

**Constraints**:
- Only warns for DEFAULT marking type (not streaks)
- Only warns if section name doesn't start with `"BONUS_"`
- Doesn't prevent execution (warning only)
- Purpose: Catch accidental positive marks for wrong answers

**Examples**:
```python
# Triggers warning
section_key = "SECTION_A"
marking = {"correct": 3, "incorrect": 2, "unmarked": 0}
→ Warning: "Found positive marks(2) for incorrect answer..."

# No warning (BONUS_ prefix)
section_key = "BONUS_SECTION"
marking = {"correct": 3, "incorrect": 2, "unmarked": 0}
→ No warning

# No warning (zero/negative incorrect)
section_key = "SECTION_A"
marking = {"correct": 3, "incorrect": -1, "unmarked": 0}
→ No warning
```

---

## Performance Constraints

### Time Complexity

**Initialization**:
```python
__init__(): O(Q + V)
where:
  Q = number of questions in section
  V = number of verdicts (always 3)

Breakdown:
- parse_fields(): O(Q)
- parse_verdict_marking_from_scheme(): O(V) = O(1)
- validate_marking_scheme(): O(V) = O(1)
→ Total: O(Q)
```

**Per-Question Operations**:
```python
get_delta_and_update_streak(): O(1)
get_delta_for_verdict(): O(1)
reset_all_streaks(): O(V) = O(1)

→ Constant time per question
```

**Full Evaluation**:
```python
Evaluate N sections with Q questions each:
→ O(N * Q)
```

**Typical Performance**:
```
10 sections × 10 questions = 100 total evaluations
→ ~0.1ms total (negligible)
```

### Space Complexity

**Per Section Scheme**:
```python
# Fixed overhead
section_key: ~50 bytes (string)
set_name: ~50 bytes (string)
marking_type: ~20 bytes (string)
empty_value: ~10 bytes (string)

# Variable size
questions: Q × ~50 bytes (average)
marking: {
    3 verdicts × (
        8 bytes (number) OR
        S × 8 bytes (streak array)
    )
}

# Streak state
streaks: 3 × 8 bytes = 24 bytes (verdict level)
OR
section_level_streak: 8 bytes + previous_verdict: 8 bytes = 16 bytes

→ Total: ~200 + (Q × 50) + (S × 24) bytes
```

**Examples**:
```
Small section (5 questions, no streak):
→ ~450 bytes

Medium section (20 questions, no streak):
→ ~1.2 KB

Large section (50 questions, 10-level streak):
→ ~2.9 KB
```

### Browser Memory Constraints

**Typical Evaluation Config**:
```
5 sections × 20 questions = 100 total questions
→ ~6 KB total for all section schemes

Large config:
20 sections × 50 questions = 1000 total questions
→ ~60 KB total

Memory Impact: Negligible (<< 1MB)
```

**Recommendation**: No memory optimization needed for typical use cases.

---

## Concurrency Constraints

### Thread Safety

**Status**: NOT thread-safe (mutable streak state)

**Mutable State**:
```python
# VERDICT_LEVEL_STREAK
self.streaks = {"correct": 0, "incorrect": 0, "unmarked": 0}
# Modified by: get_delta_and_update_streak()

# SECTION_LEVEL_STREAK
self.section_level_streak = 0
self.previous_streak_verdict = None
# Modified by: get_delta_and_update_streak()
```

**Constraint**:
- Single section scheme instance CANNOT be shared across threads
- Each evaluation must have its own section scheme instance
- OR: Reset streaks before each evaluation (done via `reset_evaluation()`)

**Safe Usage Pattern**:
```python
# Python (current implementation)
# Each file gets its own evaluation config instance
evaluation_config = EvaluationConfigForSet(...)
# Streaks reset before each file
evaluation_config.reset_evaluation()

# Browser (recommended)
// Create new instance per evaluation OR reset streaks
const sectionScheme = new SectionMarkingScheme(...);
sectionScheme.resetAllStreaks();  // Before each sheet
```

**Browser Implication**:
- Safe in single-threaded JavaScript
- If using Web Workers for parallel processing, each worker needs separate instances
- SharedArrayBuffer: Don't share section scheme objects across workers

---

## Error Handling Constraints

### No Exceptions in Normal Flow

**Guarantee**: Normal operations never throw exceptions

**Operations**:
```python
get_delta_and_update_streak()     # Never throws
get_delta_for_verdict()           # Never throws
reset_all_streaks()               # Never throws
validate_marking_scheme()         # Warns, never throws
get_bonus_type()                  # Never throws
```

**Edge Cases Handled**:
```python
# Streak array too short → Use last value
marking_array = [3, 5, 7]
current_streak = 10
→ Returns: 7 (last value, no exception)

# Empty questions for custom section → Validation catches
# But in __init__, parse_fields handles empty gracefully

# Invalid verdict → Handled upstream in AnswerMatcher
```

### Exceptions During Initialization

**Possible Exceptions**:
```python
# From parse_fields()
FieldDefinitionError: Invalid field pattern
# Example: "q1..a" (non-numeric range)

# From AnswerMatcher.parse_verdict_marking()
ValueError: Invalid fraction denominator
# Example: "1/0"

# From EvaluationConfigForSet.validate_marking_schemes()
FieldDefinitionError: Overlapping questions
EvaluationError: Missing answer key
```

**Constraint**: Exceptions only during config loading, never during evaluation.

---

## Browser Migration Constraints

### JavaScript Number Precision

**Constraint**: IEEE 754 double precision
- Range: ±1.7e308
- Precision: ~15 decimal digits

**Impact on Marking Scores**:
```javascript
// Safe (typical marking scores)
3.0, -1.0, 0.5, 2.5  // No precision issues

// Edge case (unlikely in practice)
1/3  // 0.3333333333333333 (15 digits)
2/7  // 0.2857142857142857 (15 digits)

// Total score accumulation
1000 questions × 3.33 marks = 3330.0
→ No precision loss for typical cases
```

**Recommendation**: Use integers or simple fractions (1/2, 1/4, 3/4) to avoid precision issues.

### Array Mutation

**Constraint**: JavaScript arrays are mutable by reference

**Safe Practice**:
```javascript
// Python: sorted() creates new list
sorted_values = sorted(bubble_mean_values)

// JavaScript: sort() mutates in-place
const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);
// Use spread operator to create copy

// Section marking arrays
class SectionMarkingScheme {
  constructor(sectionScheme) {
    // Deep copy marking arrays to prevent external mutation
    this.marking = {
      [Verdict.ANSWER_MATCH]: Array.isArray(sectionScheme.correct)
        ? [...sectionScheme.correct]
        : sectionScheme.correct,
      // ... etc
    };
  }
}
```

### Streak State Persistence

**Constraint**: Streak state must be reset between evaluations

**Browser Pattern**:
```javascript
class EvaluationConfigForSet {
  evaluateSheet(omrResponse) {
    // CRITICAL: Reset before each sheet
    this.sectionMarkingSchemes.forEach(scheme => {
      scheme.resetAllStreaks();
    });

    // Evaluate questions
    for (const [question, answer] of Object.entries(omrResponse)) {
      const delta = this.evaluateQuestion(question, answer);
      score += delta;
    }

    return score;
  }
}
```

**Pitfall**: Forgetting to reset → streaks carry over from previous sheet.

---

## Validation Best Practices

### Configuration Validation

**Recommended Checks**:
```typescript
interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

function validateSectionScheme(
  sectionKey: string,
  scheme: SectionScheme,
  questions: string[]
): ValidationResult {
  const result: ValidationResult = { valid: true, errors: [], warnings: [] };

  // Check 1: Required fields
  if (!scheme.marking) {
    result.errors.push('Missing marking object');
    result.valid = false;
  }

  // Check 2: Required verdicts
  for (const verdict of ['correct', 'incorrect', 'unmarked']) {
    if (!(verdict in scheme.marking)) {
      result.errors.push(`Missing ${verdict} in marking`);
      result.valid = false;
    }
  }

  // Check 3: Streak array lengths
  if (scheme.markingType === 'verdict_level_streak' ||
      scheme.markingType === 'section_level_streak') {
    for (const [verdict, marking] of Object.entries(scheme.marking)) {
      if (Array.isArray(marking) && marking.length < questions.length) {
        result.warnings.push(
          `${verdict} has ${marking.length} levels but ` +
          `${questions.length} questions`
        );
      }
    }
  }

  // Check 4: Positive incorrect warning
  if (scheme.marking.incorrect > 0 && !sectionKey.startsWith('BONUS_')) {
    result.warnings.push(
      'Positive marks for incorrect answer in non-BONUS section'
    );
  }

  return result;
}
```

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| section_key "DEFAULT" | Required | Must exist in every evaluation config |
| Marking verdicts | 3 required (correct/incorrect/unmarked) | Missing verdict → error |
| Questions overlap | Disallowed | Same question in multiple sections → error |
| Streak array length | ≥ num_questions recommended | Too short → reuses last value with warning |
| Thread safety | NOT thread-safe | Need separate instances per thread |
| Performance | O(1) per question | Fast evaluation |
| Memory | ~1KB per section | Negligible overhead |
| Bonus type detection | 3 types (None, BONUS_FOR_ALL, BONUS_ON_ATTEMPT) | Based on marking values |
| Streak reset | Required before each evaluation | Prevents state leakage |

---

## Related Constraints

- **Answer Matcher**: `../answer-matching/constraints.md`
- **Evaluation Config**: `../constraints.md`
- **Config For Set**: `../config-for-set/constraints.md`
