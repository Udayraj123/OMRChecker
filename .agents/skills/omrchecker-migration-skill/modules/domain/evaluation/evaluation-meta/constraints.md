# Evaluation Meta Constraints

**Module**: Domain - Evaluation - Meta
**Python Reference**: `src/processors/evaluation/evaluation_meta.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### concatenated_response

**Type**: `dict[str, str]`
**Format**: Question ID → Marked answer mapping

**Valid Inputs**:
```python
{}  # Empty - valid if no questions
{"q1": "A", "q2": "B"}  # Standard
{"q1": "AB", "q2": ""}  # Multi-marked and unmarked
{"roll_1": "01", "roll_2": "23"}  # Numeric roll numbers
```

**Constraints**:
- Keys: Question identifiers (strings, no format restrictions)
- Values: Marked answers (strings, can be empty for unmarked)
- Must contain ALL questions from `evaluation_config_for_response.questions_in_order`
- Extra keys allowed but ignored
- Order doesn't matter (processed in questions_in_order)

**Validation**:
```python
# Performed by prepare_and_validate_omr_response()
assert all(q in concatenated_response for q in questions_in_order)
```

### evaluation_config_for_response

**Type**: `EvaluationConfigForSet` instance

**Required Methods**:
- `prepare_and_validate_omr_response(response, allow_streak=True)`
- `match_answer_for_question(current_score, question, marked_answer)`
- `get_marking_scheme_for_question(question)`
- `conditionally_print_explanation()`
- `get_formatted_answers_summary()`

**Required Properties**:
- `questions_in_order`: List of question IDs

**Constraint**: Must be a valid, initialized EvaluationConfigForSet instance

---

## QuestionMeta Constraints

### Constructor Parameters

**question**:
- Type: `str`
- Constraint: Non-empty string
- Example: "q1", "roll_1", "section_a_q5"

**question_verdict**:
- Type: `str`
- Possible values:
  - `"answer-match"` - Standard correct answer
  - `"answer-match-A"` - Multiple correct, matched option A
  - `"answer-match-BC"` - Multiple correct, matched option BC
  - `"no-answer-match"` - Incorrect answer
  - `"unmarked"` - Empty/no response
- Constraint: Must be one of the Verdict constants or variant with suffix
- Pattern: `^(answer-match|no-answer-match|unmarked)(-[A-Z0-9]+)?$`

**marked_answer**:
- Type: `str`
- Range: Any string including empty
- Examples: `"A"`, `"BC"`, `"01"`, `""` (unmarked)
- Constraint: None (any string allowed)

**delta**:
- Type: `float`
- Range: Typically -1.0 to +2.0, but no hard limit
- Examples:
  - `+1.0` - Standard correct
  - `-0.25` - Negative marking
  - `0.0` - Unmarked or no penalty
  - `+0.5` - Partial credit (weighted)
  - `+2.0` - Bonus points
- Constraint: Must be finite (no NaN, no Inf)

**current_score**:
- Type: `float`
- Range: Any finite value (can be negative if penalties exceed points)
- Constraint: Must be finite
- Invariant: `current_score = previous_score + delta`

**answer_matcher**:
- Type: `AnswerMatcher` instance
- Required fields:
  - `answer_item`: Correct answer(s)
  - `answer_type`: Answer type constant
- Constraint: Must be valid AnswerMatcher instance

**bonus_type**:
- Type: `str | None`
- Possible values:
  - `None` - Regular question
  - `"BONUS_SECTION_1"` - Bonus section 1
  - `"BONUS_SECTION_2"` - Bonus section 2
  - `"BONUS_TOTAL"` - Bonus based on total
  - Any string starting with "BONUS"
- Constraint: None or string starting with BONUS_SECTION_PREFIX

**question_schema_verdict**:
- Type: `str`
- Possible values (from SchemaVerdict):
  - `"correct"` - Correct answer (includes partial credit)
  - `"incorrect"` - Wrong answer
  - `"unmarked"` - Not answered
- Constraint: Must be one of SchemaVerdict constants
- Pattern: `^(correct|incorrect|unmarked)$`

### QuestionMeta.to_dict() Output

**Return Type**: `dict`

**Keys** (all required):
```python
{
    "question_verdict": str,
    "marked_answer": str,
    "delta": float,
    "current_score": float,
    "answer_item": str | list[str] | list[list[str | float]],
    "answer_type": str,
    "bonus_type": str | None,
    "question_schema_verdict": str,
}
```

**Constraints**:
- All 8 keys always present
- No extra keys
- `question` field NOT included (used as key in parent dict)
- All values non-null except bonus_type

**answer_item Types by answer_type**:
```python
# answer_type: "standard"
answer_item: str
# Example: "A", "01", "BC"

# answer_type: "multiple-correct"
answer_item: list[str]
# Example: ["A", "B", "AB"], ["01", "1"]

# answer_type: "multiple-correct-weighted"
answer_item: list[list[str | float]]
# Example: [["A", 1.0], ["B", 0.5], ["C", 0.0]]
```

---

## EvaluationMeta Constraints

### Constructor

**No Parameters**: Default initialization

**Initial State**:
```python
score: 0.0  # Always starts at zero
questions_meta: {}  # Empty dict
```

**Constraint**: Always initialized with these exact values

### score Field

**Type**: `float`
**Range**: Unbounded (can be negative, can exceed 100)

**Update Rule**:
```python
# Only updated by adding delta
evaluation_meta.score += delta
```

**Examples**:
```python
# Standard scoring
0.0 → 1.0 → 2.0 → 3.0  # All correct

# With negative marking
0.0 → 1.0 → 0.75 → 1.75  # One incorrect (-0.25)

# Can go negative
0.0 → -0.25 → -0.5  # Multiple incorrect, no correct

# Can exceed 100
98.0 → 99.0 → 101.0  # Bonus questions push over 100
```

**Constraint**: Must remain finite throughout evaluation

### questions_meta Field

**Type**: `dict[str, dict]`
**Structure**: Question ID → QuestionMeta.to_dict()

**Constraints**:
- Keys: All questions from questions_in_order (after full evaluation)
- Values: QuestionMeta.to_dict() output (8 keys each)
- Order: Matches insertion order (questions_in_order)
- Size: len(questions_meta) == len(questions_in_order) after completion

**Example**:
```python
{
    "q1": {
        "question_verdict": "answer-match",
        "marked_answer": "A",
        "delta": 1.0,
        "current_score": 1.0,
        "answer_item": "A",
        "answer_type": "standard",
        "bonus_type": None,
        "question_schema_verdict": "correct"
    },
    "q2": {...},
    ...
}
```

### add_question_meta() Method

**Parameters**:
- `question`: str (question ID)
- `question_meta`: QuestionMeta instance

**Behavior**:
```python
self.questions_meta[question] = question_meta.to_dict()
```

**Constraints**:
- Overwrites if question already exists (idempotent)
- Always stores .to_dict() result, not QuestionMeta instance
- No validation of question uniqueness

### to_dict() Method

**Parameters**:
- `formatted_answers_summary`: str (summary string)

**Return Type**: `dict`

**Structure**:
```python
{
    "score": float,
    "questions_meta": dict[str, dict],
    "formatted_answers_summary": str
}
```

**Constraints**:
- All 3 keys always present
- `score`: Current total score
- `questions_meta`: Direct reference (not copy)
- `formatted_answers_summary`: Pass-through from parameter

---

## Output Constraints

### evaluate_concatenated_response() Return

**Return Type**: `tuple[float, dict]`

**Structure**:
```python
(
    score,  # float
    {
        "score": float,
        "questions_meta": dict[str, dict],
        "formatted_answers_summary": str
    }
)
```

**Constraints**:
- Tuple length: Always 2
- First element: Final score (float)
- Second element: Full metadata dict
- score == meta_dict["score"] (redundant but intentional)

**Example**:
```python
score, meta = evaluate_concatenated_response(...)

# score = 45.0
# meta = {
#     "score": 45.0,
#     "questions_meta": {...},
#     "formatted_answers_summary": "Correct: 45 Incorrect: 3 Unmarked: 2"
# }

assert score == meta["score"]  # Always true
```

### formatted_answers_summary

**Type**: `str`
**Source**: `evaluation_config_for_response.get_formatted_answers_summary()`

**Format**: Varies by config, typical examples:
```python
"Correct: 45 Incorrect: 3 Unmarked: 2"
"Score: 85/100"
"Correct: 40 Incorrect: 5 Unmarked: 5 Bonus: 2"
```

**Constraints**:
- Non-empty string (even if no questions)
- Format controlled by evaluation config, not by EvaluationMeta
- Used for display purposes only (not parsed)

---

## Performance Constraints

### Time Complexity

**evaluate_concatenated_response()**:
```python
O(n) where n = number of questions

Breakdown:
- For loop: O(n)
  - match_answer_for_question: O(1) per call
  - get_marking_scheme_for_question: O(1) lookup
  - QuestionMeta creation: O(1)
  - add_question_meta: O(1)
- get_formatted_answers_summary: O(n)
- Total: O(n)
```

**Space Complexity**:
```python
O(n) where n = number of questions

Memory:
- EvaluationMeta.score: 8 bytes (float)
- EvaluationMeta.questions_meta: ~200 bytes per question
  - question key: ~20 bytes
  - QuestionMeta dict: ~180 bytes (8 keys + values)
- Total: ~200n + 100 bytes

Examples:
- 10 questions: ~2 KB
- 50 questions: ~10 KB
- 100 questions: ~20 KB
```

### Typical Performance

```
Questions | Time (estimated)
----------|------------------
10        | < 0.5 ms
50        | < 2 ms
100       | < 5 ms
500       | < 20 ms
```

**Constraint**: Must complete in < 100ms for 100 questions on browser

---

## Browser-Specific Constraints

### JSON Serialization

**Requirement**: All metadata must be JSON-serializable

**Valid Types**:
```typescript
// ✓ Valid
string
number (finite)
boolean
null
object (plain)
array

// ✗ Invalid
undefined
NaN
Infinity
function
Symbol
```

**Constraint**: No NaN or Infinity in delta/current_score/score fields

### Memory Constraints

**Browser Heap**: Typically 1-4 GB per tab

**Memory Budget for Evaluation**:
```
Small batch (10 sheets × 50 questions): ~100 KB
Medium batch (100 sheets × 50 questions): ~1 MB
Large batch (1000 sheets × 50 questions): ~10 MB
```

**Constraint**: Must fit in < 100 MB for large batches

**Recommendation**: Don't keep all metadata in memory for 1000+ sheets. Stream to IndexedDB or download as generated.

### TypeScript Type Safety

**Interface Definitions**:
```typescript
interface QuestionMetaDict {
  question_verdict: string;
  marked_answer: string;
  delta: number;
  current_score: number;
  answer_item: string | string[] | [string, number][];
  answer_type: 'standard' | 'multiple-correct' | 'multiple-correct-weighted';
  bonus_type: string | null;
  question_schema_verdict: 'correct' | 'incorrect' | 'unmarked';
}

interface EvaluationMetaDict {
  score: number;
  questions_meta: Record<string, QuestionMetaDict>;
  formatted_answers_summary: string;
}
```

**Constraint**: All fields must match these exact types

### Immutability Considerations

**Python Approach**: Mutable objects
```python
evaluation_meta.score += delta  # Mutates score
evaluation_meta.add_question_meta(...)  # Mutates questions_meta
```

**Browser Alternative**: Immutable updates (optional)
```typescript
// Immutable approach (if using Redux/Zustand)
const newMeta = {
  ...evaluationMeta,
  score: evaluationMeta.score + delta,
  questions_meta: {
    ...evaluationMeta.questions_meta,
    [question]: questionMeta.toDict()
  }
};
```

**Recommendation**: Use mutable approach for performance unless state management library requires immutability.

---

## Validation Constraints

### Input Validation

**Required Checks**:
```python
# Check concatenated_response coverage
missing_questions = [
    q for q in questions_in_order
    if q not in concatenated_response
]
if missing_questions:
    raise ValidationError(f"Missing questions: {missing_questions}")

# Check finite values (after matching)
if not math.isfinite(delta):
    raise ValidationError(f"Non-finite delta: {delta}")

if not math.isfinite(current_score):
    raise ValidationError(f"Non-finite score: {current_score}")
```

**Current Implementation**: Validation in `prepare_and_validate_omr_response()`, finite checks implicit

**Recommendation for Browser**: Add explicit isFinite() checks

### Output Validation

**Invariants to Check**:
```python
# Score consistency
assert score == meta_dict["score"]

# Question coverage
assert len(meta_dict["questions_meta"]) == len(questions_in_order)

# All questions present
assert all(
    q in meta_dict["questions_meta"]
    for q in questions_in_order
)

# All dicts have 8 keys
assert all(
    len(qmeta) == 8
    for qmeta in meta_dict["questions_meta"].values()
)
```

---

## Error Handling Constraints

### No Exceptions in Normal Flow

**Guarantee**: `evaluate_concatenated_response()` doesn't raise exceptions for valid input

**Exceptions Only For**:
- Invalid evaluation_config_for_response
- Missing questions in concatenated_response
- Invalid answer_matcher

**Constraint**: All evaluation logic errors caught by validation step

### Graceful Degradation

**Missing answer_item** (shouldn't happen):
```python
# If answer_matcher.answer_item is None/missing
# QuestionMeta still created with None
# Browser should handle gracefully in display
```

**Invalid bonus_type** (shouldn't happen):
```python
# If bonus_type is invalid
# Still stored in metadata
# Display logic should handle
```

**Recommendation**: Add try-catch in browser for defensive programming

---

## Integration Constraints

### With Answer Matcher

**Dependency**: QuestionMeta requires AnswerMatcher instance

**Required Fields**:
```python
answer_matcher.answer_item  # Must exist
answer_matcher.answer_type  # Must exist
```

**Constraint**: Must call `match_answer_for_question()` before creating QuestionMeta

### With Evaluation Config

**Dependency**: Requires EvaluationConfigForSet instance

**Method Call Order**:
```python
# 1. Validate first
evaluation_config.prepare_and_validate_omr_response(...)

# 2. Then evaluate (calls match_answer_for_question internally)
for question in evaluation_config.questions_in_order:
    evaluation_config.match_answer_for_question(...)
```

**Constraint**: Must call prepare_and_validate before evaluation

### With CSV Export

**Usage**: Metadata written to CSV columns

**Constraint**: All values must be CSV-safe (no newlines, properly escaped)

**Format**:
```csv
file,q1,q2,q3,score,summary
sheet1.jpg,A,B,C,3.0,"Correct: 3 Incorrect: 0 Unmarked: 0"
```

**Recommendation**: Escape formatted_answers_summary if it contains commas

### With JSON Export

**Usage**: Full metadata exported as JSON file

**Format**:
```json
{
  "file": "sheet1.jpg",
  "score": 3.0,
  "questions_meta": {
    "q1": {...},
    "q2": {...},
    "q3": {...}
  },
  "formatted_answers_summary": "Correct: 3 Incorrect: 0 Unmarked: 0"
}
```

**Constraint**: Must be valid JSON (no NaN, no Inf)

---

## Concurrency Constraints

### Thread Safety

**Status**: Thread-safe for different instances, NOT thread-safe for shared instance

**Reasoning**:
```python
# Each evaluation creates new EvaluationMeta
evaluation_meta = EvaluationMeta()  # New instance

# Then mutates it
evaluation_meta.score += delta  # Mutation

# If multiple threads share same instance → race condition
```

**Constraint**: One EvaluationMeta per evaluation (per file)

**Browser Implication**:
- Main thread: Process one file at a time
- Web Workers: Each worker gets its own evaluation context
- No shared EvaluationMeta across workers

---

## Determinism Constraints

### Fully Deterministic

**Guarantee**: Same input always produces same output

**No Randomness**:
- No random number generation
- No time-based logic
- No external state

**Question Order Matters**:
```python
# Different question orders → different current_score per question
# (because current_score accumulates)

# Order 1: [q1, q2, q3]
# q1: current_score = 1.0
# q2: current_score = 2.0
# q3: current_score = 3.0

# Order 2: [q3, q2, q1]
# q3: current_score = 1.0
# q2: current_score = 2.0
# q1: current_score = 3.0

# But final score same: 3.0
```

**Constraint**: `questions_in_order` must be deterministic for reproducible metadata

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| Input validation | All questions must be present | Raises error if missing |
| score initialization | 0.0 | Always starts from zero |
| score update | += delta only | Accumulates linearly |
| current_score invariant | previous + delta | Must hold for all questions |
| questions_meta size | len(questions_in_order) | All questions included |
| QuestionMeta dict keys | Exactly 8 keys | Consistent structure |
| JSON serialization | No NaN/Inf | Browser-safe |
| Time complexity | O(n) | Linear in question count |
| Space complexity | O(n) | ~200 bytes per question |
| Thread safety | Per-instance only | New instance per evaluation |
| Determinism | Yes | Same input → same output |

---

## Related Constraints

- **Answer Matcher Constraints**: `../answer-matching/constraints.md`
- **Evaluation Config Constraints**: `../constraints.md`
- **Section Marking Constraints**: `../section-marking/constraints.md`
