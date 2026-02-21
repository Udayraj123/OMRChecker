# Answer Matcher - Flows

## Overview

The **AnswerMatcher** class is responsible for matching student responses against correct answers and determining scores. It supports three answer types: standard single answers, multiple correct answers, and weighted partial credit. The matcher integrates with the marking scheme system to calculate deltas (score changes) and track answer streaks.

**Python Reference**: `src/processors/evaluation/answer_matcher.py`

---

## Core Concepts

### Answer Types

The system supports three answer types defined in `AnswerType`:

1. **STANDARD** (`"standard"`)
   - Single correct answer (can be multi-marked like "AB")
   - Example: `"A"`, `"B"`, `"AB"`, `"01"`

2. **MULTIPLE_CORRECT** (`"multiple-correct"`)
   - Multiple acceptable answers with equal scoring
   - Example: `["A", "B", "AB"]` - any of these gets full marks

3. **MULTIPLE_CORRECT_WEIGHTED** (`"multiple-correct-weighted"`)
   - Multiple answers with different partial credit scores
   - Example: `[["A", 2], ["B", 0.5], ["AB", 2.5]]`

### Verdict System

**Question Verdicts** (internal matching result):
- `ANSWER_MATCH` - Student answer matches an allowed answer
- `NO_ANSWER_MATCH` - Student answer doesn't match any allowed answer
- `UNMARKED` - Student left question blank (matches `empty_value`)

**Schema Verdicts** (displayed to user):
- `CORRECT` - Maps from `ANSWER_MATCH`
- `INCORRECT` - Maps from `NO_ANSWER_MATCH`
- `UNMARKED` - Maps from `UNMARKED` verdict
- **Special case**: Negative custom weights in `MULTIPLE_CORRECT_WEIGHTED` force `INCORRECT` schema verdict

---

## Initialization Flow

```
┌─────────────────────────────────────────────────────────────┐
│ AnswerMatcher.__init__(answer_item, section_marking_scheme) │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─► get_answer_type(answer_item)
                     │   └─► Detect: STANDARD | MULTIPLE_CORRECT | MULTIPLE_CORRECT_WEIGHTED
                     │
                     ├─► parse_and_set_answer_item(answer_item)
                     │   └─► Parse fractions/floats for weighted scores
                     │
                     └─► set_local_marking_defaults(section_marking_scheme)
                         ├─► Copy section marking scheme locally
                         ├─► For MULTIPLE_CORRECT: Create verdict per answer
                         │   Example: "answer-match-AB" → section_marking["answer-match"]
                         └─► For MULTIPLE_CORRECT_WEIGHTED: Set custom scores
                             Example: "answer-match-A" → parsed_answer_score
```

### Answer Type Detection

**Algorithm** (`get_answer_type`):

```python
if isinstance(answer_item, str) and len(answer_item) >= 1:
    return STANDARD

if isinstance(answer_item, list):
    # Check for MULTIPLE_CORRECT: ['A', 'B', 'AB']
    if len >= 2 and all(is_standard_answer(item)):
        return MULTIPLE_CORRECT

    # Check for MULTIPLE_CORRECT_WEIGHTED: [['A', 1], ['B', 0.5]]
    if len >= 1 and all(is_two_tuple(item) and valid_score(item[1])):
        return MULTIPLE_CORRECT_WEIGHTED

raise EvaluationError("Unable to determine answer type")
```

**Examples**:
- `"A"` → `STANDARD`
- `["A", "B", "AB"]` → `MULTIPLE_CORRECT`
- `[["A", 2], ["B", 0.5]]` → `MULTIPLE_CORRECT_WEIGHTED`

---

## Answer Matching Flow

### Main Entry Point: `get_verdict_marking()`

```
┌──────────────────────────────────────────────────────┐
│ get_verdict_marking(marked_answer, allow_streak)     │
└────────────────┬─────────────────────────────────────┘
                 │
                 ├─► Determine question_verdict based on answer_type:
                 │   ├─► STANDARD: get_standard_verdict()
                 │   ├─► MULTIPLE_CORRECT: get_multiple_correct_verdict()
                 │   └─► MULTIPLE_CORRECT_WEIGHTED: get_multiple_correct_weighted_verdict()
                 │
                 ├─► section_marking_scheme.get_delta_and_update_streak()
                 │   ├─► Calculate delta (score change)
                 │   ├─► Track current_streak
                 │   └─► Update streak if allowed
                 │
                 └─► Return (question_verdict, delta, current_streak, updated_streak)
```

**Returns**:
- `question_verdict`: String verdict (e.g., "answer-match", "answer-match-AB")
- `delta`: Score change for this question
- `current_streak`: Streak count before this question
- `updated_streak`: Streak count after this question

---

## Answer Type Specific Matching

### 1. Standard Answer Matching

**Flow** (`get_standard_verdict`):

```
marked_answer = student_response
allowed_answer = self.answer_item

┌─────────────────────────────────┐
│ Is marked_answer == empty_value?│ ──Yes──► Return "unmarked"
└────────────┬────────────────────┘
             │ No
             ▼
┌─────────────────────────────────┐
│ Is marked_answer == allowed?    │ ──Yes──► Return "answer-match"
└────────────┬────────────────────┘
             │ No
             ▼
      Return "no-answer-match"
```

**Example**:
```javascript
// Configuration
answer_item: "A"
empty_value: ""

// Scenarios
marked_answer: "A"  → "answer-match"
marked_answer: "B"  → "no-answer-match"
marked_answer: ""   → "unmarked"
marked_answer: "AB" → "no-answer-match" (exact match required)
```

---

### 2. Multiple Correct Answer Matching

**Flow** (`get_multiple_correct_verdict`):

```
allowed_answers = ["A", "B", "AB"]

┌─────────────────────────────────┐
│ Is marked_answer == empty_value?│ ──Yes──► Return "unmarked"
└────────────┬────────────────────┘
             │ No
             ▼
┌─────────────────────────────────┐
│ Is marked_answer in allowed?    │ ──Yes──► Return "answer-match-{answer}"
└────────────┬────────────────────┘           Example: "answer-match-AB"
             │ No
             ▼
      Return "no-answer-match"
```

**Example**:
```javascript
// Configuration
answer_item: ["A", "B", "AB"]
marking: {
  "answer-match-A": 3,   // Created during initialization
  "answer-match-B": 3,   // Created during initialization
  "answer-match-AB": 3,  // Created during initialization
  "no-answer-match": -1,
  "unmarked": 0
}

// Scenarios
marked_answer: "A"  → verdict: "answer-match-A",  delta: 3
marked_answer: "B"  → verdict: "answer-match-B",  delta: 3
marked_answer: "AB" → verdict: "answer-match-AB", delta: 3
marked_answer: "C"  → verdict: "no-answer-match", delta: -1
marked_answer: ""   → verdict: "unmarked",        delta: 0
```

**Note**: All allowed answers receive the same score (from section marking scheme's `correct` value).

---

### 3. Weighted Answer Matching (Partial Credit)

**Flow** (`get_multiple_correct_weighted_verdict`):

```
allowed_answers = extract_answers([["A", 2], ["B", 0.5], ["AB", 2.5]])

┌─────────────────────────────────┐
│ Is marked_answer == empty_value?│ ──Yes──► Return "unmarked"
└────────────┬────────────────────┘
             │ No
             ▼
┌─────────────────────────────────┐
│ Is marked_answer in allowed?    │ ──Yes──► Return "answer-match-{answer}"
└────────────┬────────────────────┘           Use custom weight as delta
             │ No
             ▼
      Return "no-answer-match"
```

**Example**:
```javascript
// Configuration
answer_item: [["A", 2], ["B", 0.5], ["AB", 2.5], ["C", -1]]
marking: {
  "answer-match-A": 2,      // Custom weight
  "answer-match-B": 0.5,    // Custom weight
  "answer-match-AB": 2.5,   // Custom weight (full credit)
  "answer-match-C": -1,     // Negative weight (penalty)
  "no-answer-match": 0,     // From section scheme
  "unmarked": 0
}

// Scenarios
marked_answer: "A"  → verdict: "answer-match-A",  delta: 2    (partial credit)
marked_answer: "B"  → verdict: "answer-match-B",  delta: 0.5  (partial credit)
marked_answer: "AB" → verdict: "answer-match-AB", delta: 2.5  (full credit)
marked_answer: "C"  → verdict: "answer-match-C",  delta: -1   (penalty answer)
marked_answer: "D"  → verdict: "no-answer-match", delta: 0
marked_answer: ""   → verdict: "unmarked",        delta: 0
```

**Special Behavior**:
- Negative custom weights (`-1` for answer "C") force `schema_verdict = INCORRECT` even though it's technically an "answer-match"
- This allows penalizing specific wrong answers more than others

---

## Partial Credit Examples

### Use Case 1: Ambiguous Question (Multiple Fully Correct Answers)

**Scenario**: Question had a typo, both A and C are acceptable.

```json
{
  "answer_item": ["A", "C"],
  "type": "MULTIPLE_CORRECT"
}
```

**Result**: Both A and C get full marks (3 points each if section default is 3).

---

### Use Case 2: Partial Understanding (Graded Partial Credit)

**Scenario**: Complex multi-step problem where different answers show different levels of understanding.

```json
{
  "answer_item": [
    ["A", 4],    // Perfect answer (full understanding)
    ["B", 2],    // Partial solution (50% credit)
    ["C", 1],    // Basic attempt (25% credit)
    ["D", -1]    // Common misconception (penalty)
  ],
  "type": "MULTIPLE_CORRECT_WEIGHTED"
}
```

**Result**:
- Student marks A: +4 points (full credit)
- Student marks B: +2 points (partial credit)
- Student marks C: +1 point (minimal credit)
- Student marks D: -1 point (penalty for common mistake)
- Student marks E: 0 points (wrong, but no penalty beyond 0)

---

### Use Case 3: Bonus Questions

**Scenario**: Bonus question where any attempt gets points.

```json
{
  "marking_schemes": {
    "BONUS_ON_ATTEMPT": {
      "marking": {
        "correct": 5,
        "incorrect": 5,   // Same as correct!
        "unmarked": 0
      },
      "questions": ["q10"]
    }
  }
}
```

**Result**: Any non-empty answer on q10 gets 5 points.

---

## Helper Methods

### `is_part_of_some_answer(question_meta, answer_string)`

**Purpose**: Check if a bubble/character is part of any allowed answer.

**Used by**: Bubble interpretation drawing to highlight which bubbles are part of correct answer groups.

**Algorithm**:
```
if question has bonus_type:
    return True  // All answers are "correct" for bonus questions

matched_groups = get_matched_answer_groups(question_meta, answer_string)
return len(matched_groups) > 0
```

**Example**:
```javascript
// Question config
answer_item: ["A", "AB", "ABC"]

// Checks
is_part_of_some_answer("A")   → True  (in "A", "AB", "ABC")
is_part_of_some_answer("B")   → True  (in "AB", "ABC")
is_part_of_some_answer("C")   → True  (in "ABC")
is_part_of_some_answer("D")   → False
is_part_of_some_answer("AB")  → True  (exact match in allowed answers)
```

---

### `get_matched_answer_groups(question_meta, answer_string)`

**Purpose**: Find which answer groups contain the given answer string.

**Returns**: List of answer indices that match.

**Algorithm**:
```python
matched_groups = []

if answer_type == STANDARD:
    if answer_string in str(answer_item):  # "A" in "AB"
        matched_groups.append(0)

elif answer_type == MULTIPLE_CORRECT:
    for index, allowed_answer in enumerate(answer_item):
        if answer_string in allowed_answer:  # "A" in "AB"
            matched_groups.append(index)

elif answer_type == MULTIPLE_CORRECT_WEIGHTED:
    for index, (allowed_answer, score) in enumerate(answer_item):
        if answer_string in allowed_answer and score > 0:
            matched_groups.append(index)

return matched_groups
```

**Example**:
```javascript
// Question config
answer_item: ["A", "AB", "B"]

// Results
get_matched_answer_groups("A")  → [0, 1]  // Matches index 0 ("A") and index 1 ("AB")
get_matched_answer_groups("B")  → [1, 2]  // Matches index 1 ("AB") and index 2 ("B")
get_matched_answer_groups("AB") → [1]     // Matches index 1 ("AB")
get_matched_answer_groups("C")  → []      // No matches
```

**Note**: Weighted answers with `score <= 0` are excluded from matched groups (they're penalty answers).

---

## Schema Verdict Determination

### `get_schema_verdict(answer_type, question_verdict, delta=None)`

**Purpose**: Convert internal question verdict to user-facing schema verdict.

**Algorithm**:
```python
# Special case: Negative weights in MULTIPLE_CORRECT_WEIGHTED
if delta < 0 and answer_type == MULTIPLE_CORRECT_WEIGHTED:
    return SchemaVerdict.INCORRECT

# Standard mapping based on verdict prefix
for verdict in ["answer-match", "no-answer-match", "unmarked"]:
    if question_verdict.startswith(verdict):
        return VERDICT_TO_SCHEMA_VERDICT[verdict]
        # Maps: answer-match → correct
        #       no-answer-match → incorrect
        #       unmarked → unmarked
```

**Examples**:
```javascript
// Normal cases
get_schema_verdict(STANDARD, "answer-match", 3)
  → SchemaVerdict.CORRECT

get_schema_verdict(MULTIPLE_CORRECT, "answer-match-AB", 3)
  → SchemaVerdict.CORRECT  // Strips "-AB" suffix

get_schema_verdict(STANDARD, "no-answer-match", -1)
  → SchemaVerdict.INCORRECT

get_schema_verdict(STANDARD, "unmarked", 0)
  → SchemaVerdict.UNMARKED

// Special case: Negative weight on technically correct answer
get_schema_verdict(MULTIPLE_CORRECT_WEIGHTED, "answer-match-C", -1)
  → SchemaVerdict.INCORRECT  // Delta < 0 forces incorrect
```

---

## Marking Score Parsing

### `parse_verdict_marking(marking)`

**Purpose**: Parse marking scores that may be strings (fractions) or numbers.

**Algorithm**:
```python
if isinstance(marking, list):
    # Streak marking: [1, 2, 3, 5, 8]
    return [parse_float_or_fraction(m) for m in marking]

return parse_float_or_fraction(marking)
```

**Supported Formats**:
```javascript
// Integers
"3"      → 3
"-1"     → -1

// Decimals
"0.5"    → 0.5
"2.5"    → 2.5

// Fractions
"1/2"    → 0.5
"3/4"    → 0.75
"1/3"    → 0.333...

// Streak arrays
["1", "2", "3"]       → [1, 2, 3]
["0.5", "1", "1.5"]   → [0.5, 1.0, 1.5]
```

---

## Integration with Marking Scheme

### Local Marking Overrides

The AnswerMatcher creates local marking overrides for `MULTIPLE_CORRECT` and `MULTIPLE_CORRECT_WEIGHTED`:

**Example - Multiple Correct**:
```javascript
// Section marking scheme
section_marking = {
  "answer-match": 3,
  "no-answer-match": -1,
  "unmarked": 0
}

// After AnswerMatcher initialization with answer_item: ["A", "B", "AB"]
local_marking = {
  "answer-match": 3,           // Original (unused for this type)
  "answer-match-A": 3,         // Created (copied from answer-match)
  "answer-match-B": 3,         // Created (copied from answer-match)
  "answer-match-AB": 3,        // Created (copied from answer-match)
  "no-answer-match": -1,       // Original
  "unmarked": 0                // Original
}
```

**Example - Weighted**:
```javascript
// Section marking scheme
section_marking = {
  "answer-match": 3,
  "no-answer-match": 0,
  "unmarked": 0
}

// After AnswerMatcher initialization with answer_item: [["A", 2], ["B", 0.5]]
local_marking = {
  "answer-match": 3,           // Original (unused for this type)
  "answer-match-A": 2,         // Created (from custom weight)
  "answer-match-B": 0.5,       // Created (from custom weight)
  "no-answer-match": 0,        // Original
  "unmarked": 0                // Original
}
```

---

## Complete Example Walkthrough

### Scenario: Mixed Answer Types in Evaluation

**Evaluation Configuration**:
```json
{
  "options": {
    "questionsInOrder": ["q1", "q2", "q3", "q4"],
    "answersInOrder": [
      "A",                              // q1: Standard
      ["A", "B"],                       // q2: Multiple correct
      [["A", 3], ["B", 1.5], ["C", 0]], // q3: Weighted
      "C"                               // q4: Standard
    ]
  },
  "markingSchemes": {
    "DEFAULT": {
      "correct": 3,
      "incorrect": -1,
      "unmarked": 0
    }
  }
}
```

**Student Response**: `{"q1": "A", "q2": "B", "q3": "B", "q4": ""}`

**Evaluation Process**:

```
Question q1:
  AnswerMatcher: type=STANDARD, answer_item="A"
  Marked: "A" → verdict="answer-match", delta=3
  Score: 0 + 3 = 3

Question q2:
  AnswerMatcher: type=MULTIPLE_CORRECT, answer_item=["A", "B"]
  Marked: "B" → verdict="answer-match-B", delta=3
  Score: 3 + 3 = 6

Question q3:
  AnswerMatcher: type=MULTIPLE_CORRECT_WEIGHTED
  answer_item=[["A", 3], ["B", 1.5], ["C", 0]]
  Marked: "B" → verdict="answer-match-B", delta=1.5
  Score: 6 + 1.5 = 7.5

Question q4:
  AnswerMatcher: type=STANDARD, answer_item="C"
  Marked: "" → verdict="unmarked", delta=0
  Score: 7.5 + 0 = 7.5

Final Score: 7.5
Verdict Counts: {correct: 3, incorrect: 0, unmarked: 1}
```

---

## String Representation

### `__str__()`

Returns the answer item for debugging:
```python
str(answer_matcher) → str(answer_item)
```

**Examples**:
- Standard: `"A"`
- Multiple Correct: `"['A', 'B', 'AB']"`
- Weighted: `"[['A', 2], ['B', 0.5]]"`

---

## Method Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `__init__()` | Initialize matcher with answer config | None |
| `get_answer_type()` | Detect answer type from structure | AnswerType enum |
| `parse_and_set_answer_item()` | Parse scores in weighted answers | None |
| `set_local_marking_defaults()` | Create local marking overrides | None |
| `get_verdict_marking()` | Main matching entry point | (verdict, delta, streak, updated_streak) |
| `get_standard_verdict()` | Match for STANDARD type | Verdict string |
| `get_multiple_correct_verdict()` | Match for MULTIPLE_CORRECT type | Verdict string |
| `get_multiple_correct_weighted_verdict()` | Match for WEIGHTED type | Verdict string |
| `is_part_of_some_answer()` | Check if bubble is in answer groups | Boolean |
| `get_matched_answer_groups()` | Find which answers contain string | List[int] |
| `get_schema_verdict()` | Convert to user-facing verdict | SchemaVerdict enum |
| `parse_verdict_marking()` | Parse score strings/fractions | float or List[float] |
| `get_section_explanation()` | Get marking explanation | String |
| `get_matched_set_name()` | Get question set name | String |

---

## Next Steps

- See `constraints.md` for edge cases, performance considerations, and browser migration details
- See `../section-marking/` for marking scheme integration and streak bonuses
- See `../evaluation-meta/` for how question metadata is generated
