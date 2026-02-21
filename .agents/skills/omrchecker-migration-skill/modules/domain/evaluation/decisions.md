# Evaluation Config - Decisions

**Module**: Domain / Evaluation
**Python Reference**: `src/processors/evaluation/`
**Last Updated**: 2026-02-21

---

## Overview

This document explains key design decisions in the evaluation system and their rationale.

---

## Decision 1: Two-Level Configuration Architecture

**Decision**: Split evaluation into EvaluationConfig (top-level) and EvaluationConfigForSet (per-set)

**Rationale**:
- **Separation of Concerns**: Routing logic (conditional sets) vs scoring logic (answer matching)
- **Code Reusability**: Each set is self-contained, can be tested independently
- **Flexibility**: Easy to add new sets without modifying scoring logic
- **Performance**: Lazy loading - only create set configs when needed

**Alternative Considered**: Single EvaluationConfig class with inline set logic
- Rejected: Would create god class with too many responsibilities

**Trade-off**:
- Pro: Clean separation, easier to test and maintain
- Con: Slightly more complex initialization, need to understand two classes

---

## Decision 2: Answer Types (Standard, Multiple-Correct, Weighted)

**Decision**: Support three distinct answer types with different matching logic

**Rationale**:
- **Real-World Needs**:
  - Standard: Most common (single correct answer)
  - Multiple-Correct: Ambiguous questions, bonus questions (any of A, B, AB correct)
  - Weighted: Partial credit scenarios (A gives +3, B gives +1.5)
- **Type Safety**: Explicit answer types prevent configuration errors
- **Flexibility**: Cover 99% of use cases without custom logic

**Alternative Considered**: Single answer type with complex matching rules
- Rejected: Would make validation and error messages confusing

**Trade-off**:
- Pro: Clear semantics, easy to validate, good error messages
- Con: Need to parse answer format to determine type

---

## Decision 3: Conditional Sets with Regex Matching

**Decision**: Use format strings + regex for conditional set routing

**Configuration Example**:
```json
{
  "matcher": {
    "formatString": "{q21}",
    "matchRegex": "A"
  }
}
```

**Rationale**:
- **Flexibility**: Support any field (questions, barcodes, roll numbers) for routing
- **Power**: Regex allows complex patterns (e.g., `".*-SET[ABC]"`)
- **Simplicity**: No need for custom DSL or programming language
- **Browser Compatible**: Native RegExp in JavaScript

**Alternative Considered**: Python lambdas or custom DSL
- Rejected: Not portable to browser, security risk (eval)

**Trade-off**:
- Pro: Flexible, no eval needed, browser-friendly
- Con: Format string errors fail silently (return None), regex can be complex

**Why First Match Wins**:
- Predictable behavior (deterministic)
- Simple to understand and debug
- Order matters (document this in user guide)

---

## Decision 4: First Match Wins for Conditional Sets

**Decision**: Return first matching set in conditional_sets array

**Rationale**:
- **Predictability**: Deterministic behavior, easy to debug
- **Simplicity**: No need for priority/conflict resolution
- **Performance**: Early exit, no need to check all sets
- **User Control**: User controls priority via array order

**Alternative Considered**: Best match (longest/most specific regex)
- Rejected: Complexity, ambiguity, harder to debug

**Trade-off**:
- Pro: Simple, fast, predictable
- Con: Order matters (must be documented)

**Recommendation**: Document in user guide that order matters

---

## Decision 5: Merge Semantics for Conditional Sets

**Decision**: Child sets inherit parent questions/answers and can override

**Merge Rules**:
1. Parent questions come first (preserve order)
2. Child can override parent answers for same question
3. Child can add new questions at end
4. Child gets partial defaults (only outputs_configuration)

**Rationale**:
- **DRY Principle**: Avoid repeating common questions across sets
- **Flexibility**: Override only what's different (e.g., different marking scheme for Set A)
- **Consistency**: All sets use same output configuration by default

**Alternative Considered**: Full isolation (no inheritance)
- Rejected: Too much duplication for large exams

**Example**:
```json
// Parent (default)
"questionsInOrder": ["q1..20"],
"answersInOrder": ["A", "B", ..., "T"]

// Child (Set A) - only override q1 and add q21
"questionsInOrder": ["q1", "q21"],
"answersInOrder": ["X", "U"]

// Merged: ["q1", "q2", ..., "q20", "q21"]
//         ["X", "B", ..., "T",   "U"]  (q1 overridden, q21 added)
```

---

## Decision 6: Marking Scheme Section Isolation

**Decision**: Marking scheme sections must have disjoint question sets

**Enforcement**:
```python
def validate_marking_schemes():
    section_questions = set()
    for section_scheme in section_marking_schemes.values():
        current_set = set(section_scheme.questions)
        if not section_questions.isdisjoint(current_set):
            raise FieldDefinitionError("Overlapping questions")
        section_questions = section_questions.union(current_set)
```

**Rationale**:
- **Ambiguity Prevention**: Clear which scheme applies to each question
- **Simplicity**: No conflict resolution needed
- **Performance**: O(1) lookup in question_to_scheme map

**Alternative Considered**: Allow overlap with priority rules
- Rejected: Confusing, error-prone, harder to debug

**Trade-off**:
- Pro: Clear, unambiguous, fast lookup
- Con: User must be careful to avoid overlaps (validation helps)

---

## Decision 7: Streak Marking Types

**Decision**: Support three marking types - DEFAULT, VERDICT_LEVEL_STREAK, SECTION_LEVEL_STREAK

**Verdict-Level Streak**:
- Independent streaks for correct/incorrect/unmarked
- Reset all streaks when verdict changes
- Use case: Bonus for consistent performance per verdict

**Section-Level Streak**:
- Single streak for entire section
- Reset only when verdict changes
- Use case: Bonus for overall consistency

**Rationale**:
- **Verdict-Level**: Common in competitive exams (bonus for answering all correctly in a row)
- **Section-Level**: Common in adaptive tests (bonus for consistent performance)
- **DEFAULT**: Most common (no streaks)

**Why Both**:
- Different educational/testing philosophies
- Cover most real-world use cases
- Can be combined (different sections use different types)

**Alternative Considered**: Only verdict-level streaks
- Rejected: Doesn't cover section-level consistency use case

---

## Decision 8: Streak Array Length Handling

**Decision**: If streak exceeds array length, use last value

**Configuration Example**:
```json
{
  "correct": [3, 4, 5, 6, 7]  // Streaks 0-4
}
// Streak 5+ all get score 7
```

**Rationale**:
- **User Expectation**: Last value is "max bonus"
- **Convenience**: Don't need to specify array to max questions
- **Safety**: No index out of bounds errors

**Alternative Considered**: Throw error on overflow
- Rejected: Too strict, inconvenient for users

**Validation Warning**: Log warning if array length < section questions count

---

## Decision 9: Bonus Marking Detection

**Decision**: Auto-detect bonus type from marking scores

**Detection Logic**:
```python
def get_bonus_type():
    if marking[NO_ANSWER_MATCH] <= 0:
        return None  # Not bonus
    if marking[UNMARKED] > 0:
        return "BONUS_FOR_ALL"  # Everyone gets bonus
    if marking[UNMARKED] == 0:
        return "BONUS_ON_ATTEMPT"  # Only if attempted
    return None
```

**Rationale**:
- **User Convenience**: No need to specify bonus type explicitly
- **Clarity**: Bonus intent clear from scores
- **Consistency**: Bonus type affects visual rendering

**Alternative Considered**: Explicit bonus type field
- Rejected: Redundant, user must remember to set both

**Trade-off**:
- Pro: Automatic, less configuration
- Con: Magic behavior (not obvious to new users)

**Recommendation**: Document bonus detection rules clearly

---

## Decision 10: CSV Answer Key Format

**Decision**: Support both direct CSV loading and image-based CSV generation

**Direct CSV**:
```csv
q1,A
q2,B
q3,"[""A"",""B""]"  # Multiple correct
```

**Image-based**:
- Process answer key through template
- Extract OMR response
- Use as answer key

**Rationale**:
- **Convenience**: Don't need to manually type answer keys
- **Accuracy**: Reduce human error in transcription
- **Flexibility**: Support both workflows (pre-made CSV or scan answer sheet)

**Why Both**: Different user workflows
- Pre-made CSV: Faster, pre-existing data
- Image-based: More accurate, visual verification

**Trade-off**:
- Pro: Two convenient workflows
- Con: More code to maintain, more edge cases

---

## Decision 11: Answer Parsing from CSV

**Decision**: Auto-detect answer type from CSV column format

**Parsing Rules**:
1. Starts with `[`: Parse as JSON (ast.literal_eval)
2. Contains `,`: Split as array (multiple-correct)
3. Otherwise: Treat as string (standard)

**Rationale**:
- **User Convenience**: Natural format (strings, comma-separated, JSON)
- **Robustness**: Handles common formats
- **Error Messages**: Easy to debug (clear parsing errors)

**Alternative Considered**: Separate columns for answer type
- Rejected: More complex CSV format, harder to edit manually

**Edge Cases**:
- `"AB"` vs `"A,B"`: First is single answer "AB", second is two answers "A" and "B"
- `"[""AB""]"` vs `"AB"`: First is array with one element, second is string

---

## Decision 12: Explanation Table Conditional Columns

**Decision**: Show columns only if relevant (custom marking, conditional sets, streaks)

**Column Rules**:
- "Marking Scheme": Only if has_custom_marking
- "Set Mapping": Only if has_conditional_sets
- "Streak": Only if has_streak_marking and allow_streak

**Rationale**:
- **Clarity**: Don't clutter table with irrelevant columns
- **Efficiency**: Less data to process and display
- **User Experience**: Cleaner output for simple cases

**Alternative Considered**: Always show all columns
- Rejected: Confusing for simple cases, wastes space

---

## Decision 13: Empty Value Handling

**Decision**: Configurable empty value (default: "")

**Rationale**:
- **Flexibility**: Different templates use different empty values
- **Clarity**: Explicit "unmarked" vs "not detected"
- **Consistency**: Same empty value used across template and evaluation

**Alternative Considered**: Fixed empty value ""
- Rejected: Some templates use "-" or "0" for empty

**Configuration**:
```json
// In template.json
{
  "emptyValue": ""
}

// Used in evaluation for unmarked detection
if marked_answer == empty_value:
    return Verdict.UNMARKED
```

---

## Decision 14: Schema Verdicts vs Question Verdicts

**Decision**: Two-level verdict system

**Schema Verdicts**: `correct`, `incorrect`, `unmarked` (for aggregation)
**Question Verdicts**: `ANSWER_MATCH`, `ANSWER_MATCH-A`, `NO_ANSWER_MATCH`, `UNMARKED` (for detail)

**Rationale**:
- **Aggregation**: Schema verdicts used for counts, summaries
- **Detail**: Question verdicts preserve specific answer matched
- **Flexibility**: Multiple question verdicts map to same schema verdict

**Example**:
- Question verdicts: `ANSWER_MATCH-A`, `ANSWER_MATCH-B`, `ANSWER_MATCH-AB`
- Schema verdict: All map to `correct`

**Why Both**:
- Schema verdicts: Simple, consistent (3 categories)
- Question verdicts: Rich, detailed (many categories)

---

## Decision 15: Weighted Answers Override Section Schemes

**Decision**: Weighted answers override section marking schemes

**Rationale**:
- **Question-Level Priority**: Question-specific weights should take precedence
- **User Expectation**: Explicit weights are more specific than section defaults
- **Flexibility**: Mix and match (some questions weighted, some use section scheme)

**Warning**: Log warning if question with weighted answers is in custom section

**Example**:
```json
{
  "markingSchemes": {
    "SECTION_A": {
      "questions": ["q1..5"],
      "marking": { "correct": "3", "incorrect": "-1", "unmarked": "0" }
    }
  },
  "answersInOrder": [
    [["A", 5], ["B", 2]],  // q1: Weighted (overrides SECTION_A)
    "B",                    // q2: Uses SECTION_A scheme
    ...
  ]
}
```

**q1**: Uses weighted scores (5 for A, 2 for B), ignores SECTION_A
**q2**: Uses SECTION_A scheme (3 for correct)

---

## Decision 16: Validate Format Strings at Initialization

**Decision**: Validate format strings (score, answers summary) during config initialization

**Rationale**:
- **Fail Fast**: Catch errors before processing files
- **Better UX**: Clear error message at startup
- **Safety**: Prevent runtime errors during evaluation

**Validation**:
```python
# Test with dummy data
try:
    answers_summary_format_string.format(**schema_verdict_counts)
except Exception:
    raise ConfigError("Invalid format string")
```

**Alternative Considered**: Validate on first use
- Rejected: Error happens during processing (confusing)

---

## Decision 17: Exclude Answer Key Images from Processing

**Decision**: Track answer key images and exclude from file processing

**Rationale**:
- **Correctness**: Answer key is not a student sheet
- **Performance**: Skip unnecessary processing
- **Clarity**: Separate answer keys from student responses

**Implementation**:
```python
# During CSV loading
if answer_key_image_path:
    self.exclude_files.append(image_path)

# In pipeline
if file_path in evaluation_config.get_exclude_files():
    continue  # Skip this file
```

---

## Summary

**Key Decisions**:
1. Two-level config architecture (routing vs scoring)
2. Three answer types (standard, multiple-correct, weighted)
3. Conditional sets with regex matching (first match wins)
4. Parent-child merging for conditional sets
5. Disjoint marking scheme sections
6. Three marking types (default, verdict-streak, section-streak)
7. Streak overflow uses last value
8. Auto-detect bonus type from scores
9. CSV and image-based answer keys
10. Auto-detect answer type from CSV format
11. Conditional explanation table columns
12. Configurable empty value
13. Two-level verdicts (schema + question)
14. Weighted answers override section schemes
15. Validate format strings at initialization
16. Exclude answer key images from processing

**Browser Migration**: All decisions are browser-compatible (no Python-specific features). Maintain same semantics in JavaScript.
