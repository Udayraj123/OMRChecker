# Evaluation Config - Constraints

**Module**: Domain / Evaluation
**Python Reference**: `src/processors/evaluation/`
**Last Updated**: 2026-02-21

---

## Overview

This document details constraints, limitations, and edge cases in the evaluation system.

---

## Configuration Constraints

### 1. Required Fields

**evaluation.json Must Have**:
- `sourceType`: Must be "local", "csv", or "image_and_csv"
- `options`: Configuration object (contents depend on sourceType)
- `markingSchemes`: Must contain "DEFAULT" scheme
- `markingSchemes.DEFAULT`: Must have "correct", "incorrect", "unmarked" keys

**Validation**: JSON Schema validation at load time

**Error**: ConfigError if missing

---

### 2. Source Type Constraints

**For sourceType = "local"**:
- `options.questionsInOrder`: Required (array of field strings)
- `options.answersInOrder`: Required (array of answer items)
- Must have equal lengths
- Error if mismatch: FieldDefinitionError

**For sourceType = "csv" or "image_and_csv"**:
- `options.answerKeyCsvPath`: Required (string)
- `options.answerKeyImagePath`: Optional (string, for CSV generation)
- `options.questionsInOrder`: Required if answerKeyImagePath provided
- Error if CSV missing and no image: InputFileNotFoundError

---

### 3. Conditional Set Constraints

**Unique Set Names**:
- No duplicate names in conditional_sets array
- Error: ConfigError with duplicate name

**Questions/Answers Consistency**:
- If `answersInOrder` provided, `questionsInOrder` must exist
- If `questionsInOrder` provided, `answersInOrder` must exist
- Error: ConfigError with missing field

**Matcher Format**:
- `formatString`: Python format string (e.g., "{Roll}")
- `matchRegex`: Valid regex pattern
- Error: Silent failure (returns None) if format/regex fails

---

### 4. Marking Scheme Constraints

**DEFAULT Scheme Required**:
- Every evaluation.json must have "DEFAULT" in markingSchemes
- Error: JSON Schema validation failure

**Section Questions Disjoint**:
- Custom sections must not overlap
- Error: FieldDefinitionError

**All Section Questions Must Have Answers**:
- Every question in section must exist in questions_in_order
- Error: EvaluationError with missing questions list

**Marking Score Format**:
- Number (int/float): `3`, `-1`, `0.5`
- Fraction string: `"3/2"`, `"-1/2"`
- Streak array: `[3, 4, 5, 6]` (for streak marking)
- Error: Validation error if invalid format

---

### 5. Answer Type Constraints

**Standard Answer**:
- Must be string with len >= 1
- Examples: `"A"`, `"01"`, `"AB"`

**Multiple Correct**:
- Must be array of strings
- Min length: 2
- Examples: `["A", "B"]`, `["01", "1"]`

**Multiple Correct Weighted**:
- Must be array of [string, score] tuples
- Min length: 1
- Score can be number or fraction string
- Examples: `[["A", 3], ["B", 1.5]]`, `[["A", "3/2"]]`

**Validation**: Type checking during initialization
**Error**: EvaluationError if invalid format

---

### 6. Streak Marking Constraints

**Array Length Warning**:
- If marking score is array and section has questions
- If len(array) < len(questions): Log warning
- Behavior: Use last value for overflows

**Marking Type Constraint**:
- `markingType`: Must be "DEFAULT", "VERDICT_LEVEL_STREAK", or "SECTION_LEVEL_STREAK"
- DEFAULT: No streak arrays allowed in marking scores
- VERDICT_LEVEL_STREAK: Allows streak arrays
- SECTION_LEVEL_STREAK: Allows streak arrays

---

## Runtime Constraints

### 7. Question Validation

**All Questions Must Exist in Response**:
```python
omr_response_keys = set(concatenated_omr_response.keys())
all_questions = set(questions_in_order)
missing_questions = all_questions.difference(omr_response_keys)

if len(missing_questions) > 0:
    raise EvaluationError("Missing question keys")
```

**Warning for Extra Questions**:
- If response has fields starting with "q" not in questions_in_order
- Log warning (not error)
- Suggests renaming field in evaluation.json

---

### 8. Multi-Marked Answer Key Constraint

**If tuning_config.outputs.filter_out_multimarked_files = True**:
- Answer key cannot contain multi-marked answers (e.g., "AB")
- Validation during initialization
- Error: ConfigError

**Rationale**: Inconsistent behavior (filter student multi-marks but not answer key)

**Affected Answer Types**:
- Standard: Check if len(answer) > 1
- Multiple Correct: Check if any answer has len > 1
- Weighted: Check if any answer string has len > 1

---

### 9. CSV Answer Key Constraints

**CSV Format**:
- Two columns: question, answer
- No header row
- Question: Trimmed string
- Answer: Parsed via parse_answer_column

**Answer Parsing**:
- Whitespace removed before parsing
- `[...]`: Parsed as JSON (ast.literal_eval)
- Contains `,`: Split on comma
- Otherwise: String

**Errors**:
- CSV not found: InputFileNotFoundError (unless image provided)
- Parse error: ast.literal_eval exception
- Invalid JSON: SyntaxError

---

### 10. Image-Based Answer Key Constraints

**Image Must Exist**:
- Error: ImageReadError if file not found

**Image Must Be Readable**:
- Error: ImageReadError if ImageUtils.read_image_util fails

**Template Processing Must Succeed**:
- Uses full template pipeline (preprocessing, alignment, detection)
- Inherits all pipeline constraints

**Empty Answers Not Allowed** (if questionsInOrder provided):
- All questions in questionsInOrder must have non-empty responses
- Error: EvaluationError with list of empty questions

**Warning**: If questionsInOrder not provided, uses all non-empty fields (less validation)

---

## Browser Migration Constraints

### 11. File System Constraints

**Python**: Direct file system access via pathlib
**Browser**: File API (user-selected files only)

**Impact**:
- Cannot auto-discover answer key CSV relative to evaluation.json
- User must provide evaluation.json + CSV as separate file inputs
- Answer key image must be uploaded separately

**Mitigation**:
```typescript
interface EvaluationFiles {
    evaluationJson: File;
    answerKeyCsv?: File;
    answerKeyImage?: File;
}

async function loadEvaluation(files: EvaluationFiles): Promise<EvaluationConfig> {
    const json = JSON.parse(await files.evaluationJson.text());

    if (json.sourceType === 'csv' || json.sourceType === 'image_and_csv') {
        if (!files.answerKeyCsv && !files.answerKeyImage) {
            throw new Error('CSV or image required for this source type');
        }
    }

    // Process files...
}
```

---

### 12. Regex Constraints

**Python**: `re.search(pattern, string)`
**Browser**: `new RegExp(pattern).test(string)`

**Differences**:
- Python: Full re module (lookaheads, lookbehinds, named groups)
- JavaScript: Native RegExp (mostly compatible, but some advanced features differ)

**Common Patterns** (Both Compatible):
- `"A"`: Exact match
- `".*-SET[ABC]"`: Suffix match
- `"^[0-9]{5}$"`: Exact 5 digits
- `"(A|B|C)"`: Alternatives

**Incompatible Patterns**:
- `(?P<name>...)`: Named groups (Python only)
- Conditional patterns (limited in JavaScript)

**Mitigation**: Document regex limitations in browser version, test common patterns

---

### 13. Performance Constraints

**Large Answer Keys**:
- 1000+ questions: Python handles easily
- Browser: May need optimization (Web Workers)

**Streak Arrays**:
- Long arrays (1000+ values): Rare, but possible
- Memory: Minimal impact (small arrays)

**Conditional Sets**:
- Many sets (100+): Rare
- Performance: O(n) iteration (first match wins)
- Mitigation: Recommend putting common sets first

---

### 14. Memory Constraints

**Python**: Unrestricted (OS memory)
**Browser**: Heap limits (varies by browser)

**Evaluation Metadata**:
- Per-question metadata (QuestionMeta): ~200 bytes
- 1000 questions: ~200 KB (negligible)

**Explanation Table**:
- Rich table in memory
- Export as CSV: Minimal memory (streaming write)

**Browser Impact**: Minimal (evaluation metadata is small)

---

### 15. Format String Constraints

**Python**: `"{correct}".format(correct=8)` → `"8"`
**Browser**: Template literals or manual replacement

**Supported Variables**:
- Answers Summary: `{correct}`, `{incorrect}`, `{unmarked}`, `{bonus}`
- Score: `{score}`

**Browser Implementation**:
```typescript
function formatAnswersSummary(template: string, counts: Record<string, number>): string {
    return template.replace(/\{(\w+)\}/g, (_, key) => String(counts[key] || 0));
}
```

**Edge Cases**:
- Unknown variable: Return "0" or empty string
- Malformed template: Return original template

---

### 16. Validation Timing Constraints

**Python**:
- JSON Schema validation at load
- Config validation during initialization
- Runtime validation during evaluation

**Browser**:
- Zod validation at load (synchronous)
- Config validation during initialization (synchronous)
- Runtime validation during evaluation (synchronous)

**No Change**: Same validation timing in browser

---

## Edge Cases

### 17. Empty Answer Key

**Scenario**: questions_in_order = [], answers_in_order = []

**Behavior**:
- Valid configuration (no questions to evaluate)
- Score always 0
- No metadata

**Use Case**: Disable evaluation while keeping config structure

---

### 18. All Questions Unmarked

**Scenario**: Student doesn't mark any bubbles

**Behavior**:
- All verdicts: UNMARKED
- Score: sum of unmarked scores (typically 0)
- Warning: May want to flag for review

**Recommendation**: Add check for "all unmarked" and flag file

---

### 19. Negative Final Score

**Scenario**: Harsh negative marking

**Behavior**:
- Score can go negative
- No floor at 0 (by design)

**Use Case**: Penalty-heavy exams

**Alternative**: Apply floor in post-processing if needed

---

### 20. Fractional Scores

**Scenario**: Fraction string or decimal answers

**Behavior**:
- `"3/2"` → 1.5
- `-1.5` → -1.5
- Precision: Python float (64-bit)

**Browser**: JavaScript Number (64-bit float, same precision)

---

### 21. Conditional Set Match Failure

**Scenario**: No conditional set matches, formatString throws error

**Behavior**:
- Exception caught, returns None
- Falls back to default set

**Edge Cases**:
- Missing field in formatString: Exception → None → default
- Regex compile error: Exception → None → default

**Silent Failure**: Can be hard to debug

**Recommendation**: Log warning when falling back to default

---

### 22. Duplicate Questions in questions_in_order

**Scenario**: `["q1", "q2", "q1"]` (duplicate q1)

**Behavior**:
- Allowed (no validation for duplicates)
- Last answer for q1 used in question_to_answer_matcher
- Question evaluated multiple times in loop

**Impact**: Confusing, likely user error

**Recommendation**: Add validation to detect duplicates (future enhancement)

---

### 23. Parent-Child Question Override Edge Case

**Scenario**: Child overrides parent question with different answer type

**Example**:
- Parent: `"q1": "A"` (standard)
- Child: `"q1": ["A", "B"]` (multiple correct)

**Behavior**:
- Allowed (no type checking)
- Child answer type used

**Impact**: May confuse users

**Recommendation**: Document that answer type can change

---

### 24. Streak Overflow Warning

**Scenario**: Section has 100 questions, streak array has 5 values

**Behavior**:
- Warning logged during validation
- Last value used for streaks 5+

**Example**:
```json
{
  "correct": [3, 4, 5, 6, 7]
}
// Questions 1-5: Use array values
// Questions 6-100: All use 7
```

---

### 25. Bonus Section with Streak Marking

**Scenario**: Bonus section (incorrect > 0) with streak marking

**Behavior**:
- Bonus type detection ignores streak marking
- Both features work independently

**Example**:
```json
{
  "BONUS_STREAK": {
    "markingType": "VERDICT_LEVEL_STREAK",
    "questions": ["q1..10"],
    "marking": {
      "correct": [3, 4, 5],
      "incorrect": [3, 3, 3],  // Bonus even for incorrect
      "unmarked": "0"
    }
  }
}
```

**Bonus Type**: BONUS_ON_ATTEMPT (incorrect > 0, unmarked = 0)

---

## Performance Benchmarks (Python)

**Typical Exam** (100 questions, default marking):
- Initialization: <100ms
- Per-file evaluation: <10ms
- Memory: <1 MB

**Large Exam** (1000 questions, 10 sections):
- Initialization: <500ms
- Per-file evaluation: <50ms
- Memory: <5 MB

**Conditional Sets** (5 sets, 100 questions each):
- Initialization: <300ms
- Set matching: <1ms
- Per-file evaluation: <10ms

**Browser Expected** (similar or slightly slower):
- Initialization: <200ms (Zod validation)
- Per-file evaluation: <20ms (pure JS, no I/O)

---

## Summary

**Key Constraints**:
1. evaluation.json must have sourceType, options, markingSchemes (with DEFAULT)
2. Questions and answers must have equal lengths
3. Conditional sets must have unique names
4. Section questions must be disjoint
5. All section questions must have answers
6. Multi-marked answer keys not allowed if filter_out_multimarked_files enabled
7. CSV format: 2 columns, no header, auto-parsed
8. Image-based keys: No empty answers if questionsInOrder provided
9. Streak arrays: Last value used for overflow
10. All questions must exist in OMR response

**Browser Migration**:
- File API instead of file system (user must upload CSV/image separately)
- Native RegExp (mostly compatible, document limitations)
- Zod validation (similar to JSON Schema)
- Same performance characteristics (evaluation is pure computation)
- Template literals for format strings

**Edge Cases**:
- Empty answer keys (valid, score = 0)
- All unmarked (valid, may want to flag)
- Negative scores (allowed)
- Conditional set match failure (silent fallback to default)
- Duplicate questions (allowed, last answer wins)

**Recommendation**: Add validation for duplicate questions, log warnings for conditional set fallbacks
