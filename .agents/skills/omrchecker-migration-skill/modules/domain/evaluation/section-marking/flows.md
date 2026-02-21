# Section Marking Scheme Flows

**Module**: Domain - Evaluation - Section Marking
**Python Reference**: `src/processors/evaluation/section_marking_scheme.py`
**Last Updated**: 2026-02-21

---

## Overview

SectionMarkingScheme manages **per-section scoring rules** for evaluation. It supports:
- **Default schemes**: Apply to all questions unless overridden
- **Custom section schemes**: Apply different scoring to question subsets
- **Streak-based marking**: Bonus/penalty for consecutive correct/incorrect answers
- **Weighted scores**: Custom marks per verdict (correct/incorrect/unmarked)

**Use Case**: Flexible scoring for different question types (easy/hard sections, bonus questions, negative marking).

---

## Core Concepts

### Marking Types

```python
class MarkingSchemeType:
    DEFAULT = "default"                      # Standard per-question scoring
    VERDICT_LEVEL_STREAK = "verdict_level_streak"  # Track streaks per verdict type
    SECTION_LEVEL_STREAK = "section_level_streak"  # Track overall section streak
```

**Marking Type Comparison**:

| Type | Streak Tracking | Use Case |
|------|----------------|----------|
| DEFAULT | None | Standard scoring (e.g., +3 correct, -1 incorrect) |
| VERDICT_LEVEL_STREAK | Per verdict (correct/incorrect/unmarked) | Bonus for consecutive correct answers |
| SECTION_LEVEL_STREAK | Section-wide | Bonus for maintaining same verdict type |

### Verdicts and Schema Verdicts

```python
# Question verdicts (granular)
Verdict = {
    "ANSWER_MATCH": "answer-match",           # e.g., "answer-match-A"
    "NO_ANSWER_MATCH": "no-answer-match",     # Wrong answer
    "UNMARKED": "unmarked"                     # Empty/blank answer
}

# Schema verdicts (for scoring)
SchemaVerdict = {
    "CORRECT": "correct",       # Mapped from ANSWER_MATCH
    "INCORRECT": "incorrect",   # Mapped from NO_ANSWER_MATCH
    "UNMARKED": "unmarked"      # Direct mapping
}

# Mapping
VERDICT_TO_SCHEMA_VERDICT = {
    Verdict.ANSWER_MATCH: SchemaVerdict.CORRECT,
    Verdict.NO_ANSWER_MATCH: SchemaVerdict.INCORRECT,
    Verdict.UNMARKED: SchemaVerdict.UNMARKED,
}
```

---

## Initialization Flow

### Constructor Flow

```
START: SectionMarkingScheme.__init__(section_key, section_scheme, set_name, empty_value)
│
├─► STEP 1: Store Basic Properties
│   │
│   self.section_key = section_key           # e.g., "DEFAULT", "BONUS_SECTION", "HARD_QUESTIONS"
│   self.set_name = set_name                 # e.g., "DEFAULT_SET", "SET_A"
│   self.empty_value = empty_value           # e.g., "", "0"
│   self.marking_type = section_scheme.get("marking_type", "default")
│
├─► STEP 2: Initialize Streak State
│   │
│   self.reset_all_streaks()
│   │
│   ├─ If marking_type == VERDICT_LEVEL_STREAK:
│   │  │
│   │  self.streaks = {
│   │      "correct": 0,
│   │      "incorrect": 0,
│   │      "unmarked": 0
│   │  }
│   │
│   └─ Else (DEFAULT or SECTION_LEVEL_STREAK):
│      │
│      self.section_level_streak = 0
│      self.previous_streak_verdict = None
│
├─► STEP 3: Parse Section Questions
│   │
│   ├─ If section_key == "DEFAULT":
│   │  │
│   │  self.questions = None  # Applies to all questions not in other sections
│   │  self.marking = self.parse_verdict_marking_from_scheme(section_scheme)
│   │  # Parse directly from section_scheme (shorthand)
│   │
│   └─ Else (custom section):
│      │
│      self.questions = parse_fields(section_key, section_scheme["questions"])
│      # Example: "q1..10" → ["q1", "q2", ..., "q10"]
│      # Example: ["q5", "q7", "q9"] → ["q5", "q7", "q9"]
│      │
│      self.marking = self.parse_verdict_marking_from_scheme(section_scheme["marking"])
│      # Parse from section_scheme["marking"]
│
├─► STEP 4: Parse Verdict Marking
│   │
│   # For each verdict (answer-match, no-answer-match, unmarked):
│   For verdict in VERDICTS_IN_ORDER:
│   │   schema_verdict = VERDICT_TO_SCHEMA_VERDICT[verdict]
│   │   # e.g., "answer-match" → "correct"
│   │
│   │   schema_verdict_marking = AnswerMatcher.parse_verdict_marking(
│   │       section_scheme[schema_verdict]
│   │   )
│   │   # Parses: "3", "-1", "1/2", or [1, 2, 3] (streak array)
│   │   # Returns: 3.0, -1.0, 0.5, or [1.0, 2.0, 3.0]
│   │
│   │   # Validation: Warn if positive marks for incorrect in non-bonus section
│   │   If (marking_type == DEFAULT and
│   │       schema_verdict_marking > 0 and
│   │       schema_verdict == "incorrect" and
│   │       not section_key.startswith("BONUS_")):
│   │       logger.warning("Found positive marks for incorrect answer...")
│   │
│   │   parsed_marking[verdict] = schema_verdict_marking
│   │
│   self.marking = parsed_marking
│   # Result: {
│   #   "answer-match": 3.0,
│   #   "no-answer-match": -1.0,
│   #   "unmarked": 0.0
│   # }
│
└─► STEP 5: Validate Marking Scheme
    │
    self.validate_marking_scheme()
    # Check streak array lengths vs. number of questions

END
```

---

## Verdict Marking Calculation Flow

### Main Flow: get_delta_and_update_streak()

```
START: get_delta_and_update_streak(answer_matcher_marking, answer_type, question_verdict, allow_streak)
│
│ INPUT:
│ - answer_matcher_marking: Marking scores dict (e.g., {"answer-match": 3, ...})
│ - answer_type: "standard", "multiple-correct", or "multiple-correct-weighted"
│ - question_verdict: "answer-match", "no-answer-match", or "unmarked"
│ - allow_streak: bool (whether to update streak counter)
│
├─► STEP 1: Get Schema Verdict
│   │
│   schema_verdict = AnswerMatcher.get_schema_verdict(answer_type, question_verdict, 0)
│   # Maps question_verdict to schema_verdict
│   # "answer-match" → "correct"
│   # "no-answer-match" → "incorrect"
│   # "unmarked" → "unmarked"
│
├─► STEP 2: Branch by Marking Type
│   │
│   ┌─ BRANCH A: VERDICT_LEVEL_STREAK
│   │  │
│   │  ├─► Get Current Streak
│   │  │   current_streak = self.streaks[schema_verdict]
│   │  │   # e.g., streaks["correct"] = 2 (2 consecutive correct so far)
│   │  │
│   │  ├─► Reset All Streaks
│   │  │   self.reset_all_streaks()
│   │  │   # All verdicts back to 0: {"correct": 0, "incorrect": 0, "unmarked": 0}
│   │  │
│   │  ├─► Update Current Verdict Streak
│   │  │   If allow_streak and schema_verdict != "unmarked":
│   │  │       self.streaks[schema_verdict] = current_streak + 1
│   │  │       # Increase ONLY current verdict streak
│   │  │
│   │  ├─► Get Delta Score
│   │  │   delta = self.get_delta_for_verdict(
│   │  │       answer_matcher_marking,
│   │  │       question_verdict,
│   │  │       current_streak
│   │  │   )
│   │  │   # Uses current_streak (BEFORE increment)
│   │  │   # Example: If marking["correct"] = [3, 5, 7]
│   │  │   #          and current_streak = 2
│   │  │   #          then delta = 7 (third correct in a row gets 7 marks)
│   │  │
│   │  └─► Return updated_streak
│   │      updated_streak = self.streaks[schema_verdict]
│   │      # e.g., 3 (after incrementing from 2)
│   │
│   ┌─ BRANCH B: SECTION_LEVEL_STREAK
│   │  │
│   │  ├─► Get Current Streak
│   │  │   current_streak = self.section_level_streak
│   │  │   previous_verdict = self.previous_streak_verdict
│   │  │
│   │  ├─► Reset Streak State
│   │  │   self.reset_all_streaks()
│   │  │   # section_level_streak = 0, previous_streak_verdict = None
│   │  │
│   │  ├─► Update Section Streak
│   │  │   If (allow_streak and previous_verdict is None) or
│   │  │      (schema_verdict == previous_verdict):
│   │  │       self.section_level_streak = current_streak + 1
│   │  │       # Increment if first question OR same verdict as previous
│   │  │   Else:
│   │  │       # Verdict changed, streak resets to 0
│   │  │       pass
│   │  │
│   │  ├─► Get Delta Score
│   │  │   delta = self.get_delta_for_verdict(
│   │  │       answer_matcher_marking,
│   │  │       question_verdict,
│   │  │       current_streak
│   │  │   )
│   │  │
│   │  └─► Return updated_streak
│   │      updated_streak = self.section_level_streak
│   │
│   └─ BRANCH C: DEFAULT
│      │
│      ├─► No Streak Tracking
│      │   current_streak = 0
│      │   updated_streak = 0
│      │
│      └─► Get Delta Score
│          delta = self.get_delta_for_verdict(
│              answer_matcher_marking,
│              question_verdict,
│              0  # streak always 0
│          )
│
└─► RETURN (delta, current_streak, updated_streak)

END
```

### Helper Flow: get_delta_for_verdict()

```
START: get_delta_for_verdict(answer_matcher_marking, question_verdict, current_streak)
│
│ INPUT:
│ - answer_matcher_marking: {"answer-match": 3, "no-answer-match": -1, ...}
│   OR {"answer-match": [3, 5, 7], ...} (streak array)
│ - question_verdict: "answer-match", "no-answer-match", or "unmarked"
│ - current_streak: int (e.g., 0, 1, 2, ...)
│
├─► STEP 1: Get Marking Value
│   │
│   marking_value = answer_matcher_marking[question_verdict]
│   # Could be: 3.0 (number) OR [3.0, 5.0, 7.0] (list)
│
├─► STEP 2: Check if List (Streak Array)
│   │
│   If isinstance(marking_value, list):
│   │   # Streak-based marking
│   │   │
│   │   ├─► Bounds Check
│   │   │   If current_streak < len(marking_value):
│   │   │       RETURN marking_value[current_streak]
│   │   │       # Use streak index: marking_value[0], marking_value[1], ...
│   │   │   Else:
│   │   │       RETURN marking_value[-1]
│   │   │       # Use last value if streak exceeds array length
│   │   │       # (Not shown in code, but would happen via IndexError → use last)
│   │
│   Else:
│   │   # Non-streak marking (single value)
│   │   │
│   │   ├─► Check for Non-Zero Streak Warning
│   │   │   If current_streak > 0:
│   │   │       logger.warning(
│   │   │           f"Non zero streak({current_streak}) for verdict {question_verdict}. "
│   │   │           f"Using non-streak score for this verdict."
│   │   │       )
│   │   │       # Warn: Streak tracking enabled but marking is not an array
│   │   │
│   │   └─► RETURN marking_value
│   │       # Always return the same score regardless of streak
│
└─► END

Examples:
┌─────────────────────────────────────────────────────────────────
│ Example 1: Non-Streak Marking
│ marking_value = 3.0
│ current_streak = 0 → RETURN 3.0
│ current_streak = 2 → RETURN 3.0 (with warning)
│
│ Example 2: Streak Marking
│ marking_value = [3, 5, 7]
│ current_streak = 0 → RETURN 3
│ current_streak = 1 → RETURN 5
│ current_streak = 2 → RETURN 7
│ current_streak = 5 → RETURN 7 (last value, out of bounds)
└─────────────────────────────────────────────────────────────────
```

---

## Validation Flow

### validate_marking_scheme()

```
START: validate_marking_scheme()
│
├─► STEP 1: Check if Streak-Based
│   │
│   If marking_type NOT IN {VERDICT_LEVEL_STREAK, SECTION_LEVEL_STREAK}:
│       RETURN (no validation needed for DEFAULT)
│
├─► STEP 2: Find Streak Markings
│   │
│   streak_markings = []
│   For verdict, marking in self.marking.items():
│       If isinstance(marking, list):
│           streak_markings.append((verdict, len(marking)))
│   │
│   # Example: [("answer-match", 5), ("no-answer-match", 3)]
│
├─► STEP 3: Validate Streak Array Lengths
│   │
│   If self.questions exists AND len(self.questions) > 0 AND len(streak_markings) > 0:
│   │
│   │   max_possible_streak = len(self.questions)
│   │   # E.g., 10 questions → max streak = 10
│   │
│   │   For verdict, length in streak_markings:
│   │       If length < max_possible_streak:
│   │           logger.warning(
│   │               f"Marking scheme '{self.section_key}': "
│   │               f"Verdict '{verdict}' has {length} streak levels "
│   │               f"but section has {max_possible_streak} questions. "
│   │               f"Consider adding more streak levels or the last level "
│   │               f"will be used for all remaining streaks."
│   │           )
│
└─► END

Example Warning:
Section "HARD_QUESTIONS" has 10 questions
Marking: {"correct": [3, 5, 7]}  # Only 3 streak levels

Warning: Verdict 'answer-match' has 3 streak levels but section has 10 questions.
Result: Questions 1-3 get [3,5,7], Questions 4-10 all get 7 (last value reused)
```

---

## Streak Tracking Examples

### Example 1: VERDICT_LEVEL_STREAK

**Configuration**:
```json
{
  "markingType": "verdict_level_streak",
  "marking": {
    "correct": [3, 5, 7, 10],
    "incorrect": [-1, -2, -3],
    "unmarked": 0
  },
  "questions": ["q1", "q2", "q3", "q4", "q5"]
}
```

**Question Sequence**:
```
START: Initial state
streaks = {"correct": 0, "incorrect": 0, "unmarked": 0}

Question q1: Correct
├─ current_streak = streaks["correct"] = 0
├─ delta = marking["correct"][0] = 3
├─ Reset all streaks → {"correct": 0, "incorrect": 0, "unmarked": 0}
├─ Update current → streaks["correct"] = 1
└─ Score: +3 (first correct)

Question q2: Correct
├─ current_streak = streaks["correct"] = 1
├─ delta = marking["correct"][1] = 5
├─ Reset all streaks → {"correct": 0, "incorrect": 0, "unmarked": 0}
├─ Update current → streaks["correct"] = 2
└─ Score: +5 (second consecutive correct)

Question q3: Incorrect
├─ current_streak = streaks["incorrect"] = 0
├─ delta = marking["incorrect"][0] = -1
├─ Reset all streaks (correct streak LOST)
├─ Update current → streaks["incorrect"] = 1
└─ Score: -1 (first incorrect, correct streak broken)

Question q4: Correct
├─ current_streak = streaks["correct"] = 0 (reset by q3)
├─ delta = marking["correct"][0] = 3
├─ Reset all streaks
├─ Update current → streaks["correct"] = 1
└─ Score: +3 (correct streak restarted)

Question q5: Correct
├─ current_streak = streaks["correct"] = 1
├─ delta = marking["correct"][1] = 5
├─ Reset all streaks
├─ Update current → streaks["correct"] = 2
└─ Score: +5

Total Score: 3 + 5 - 1 + 3 + 5 = 15
```

### Example 2: SECTION_LEVEL_STREAK

**Configuration**:
```json
{
  "markingType": "section_level_streak",
  "marking": {
    "correct": [2, 3, 4, 5],
    "incorrect": [-1],
    "unmarked": 0
  },
  "questions": ["q1", "q2", "q3", "q4"]
}
```

**Question Sequence**:
```
START: Initial state
section_level_streak = 0
previous_streak_verdict = None

Question q1: Correct
├─ current_streak = 0
├─ previous_verdict = None
├─ delta = marking["correct"][0] = 2
├─ Condition: allow_streak and previous is None → True
├─ section_level_streak = 1
├─ previous_streak_verdict = "correct"
└─ Score: +2 (streak = 0 → 1)

Question q2: Correct
├─ current_streak = 1
├─ previous_verdict = "correct"
├─ delta = marking["correct"][1] = 3
├─ Condition: schema_verdict == previous_verdict → True
├─ section_level_streak = 2
└─ Score: +3 (streak = 1 → 2)

Question q3: Incorrect
├─ current_streak = 2
├─ previous_verdict = "correct"
├─ delta = marking["incorrect"][2] = ? (out of bounds, uses [-1]? or last?)
├─ Condition: "incorrect" != "correct" → False
├─ section_level_streak = 0 (reset)
├─ previous_streak_verdict = "incorrect"
└─ Score: -1 (streak broken, reset to 0)

Question q4: Incorrect
├─ current_streak = 0
├─ previous_verdict = "incorrect"
├─ delta = marking["incorrect"][0] = -1
├─ Condition: "incorrect" == "incorrect" → True
├─ section_level_streak = 1
└─ Score: -1 (new streak of incorrect)

Total Score: 2 + 3 - 1 - 1 = 3
```

### Example 3: DEFAULT (No Streak)

**Configuration**:
```json
{
  "marking": {
    "correct": 4,
    "incorrect": -1,
    "unmarked": 0
  }
}
```

**Question Sequence**:
```
Question q1: Correct → +4
Question q2: Correct → +4 (no bonus)
Question q3: Incorrect → -1
Question q4: Correct → +4 (no bonus)

Total Score: 4 + 4 - 1 + 4 = 11
```

---

## Bonus Type Detection Flow

### get_bonus_type()

```
START: get_bonus_type()
│
├─► STEP 1: Check if Streak Marking
│   │
│   If marking_type == VERDICT_LEVEL_STREAK:
│       RETURN None  # Streak marking doesn't have bonus type
│
├─► STEP 2: Check No-Answer-Match Score
│   │
│   If self.marking[Verdict.NO_ANSWER_MATCH] <= 0:
│       RETURN None  # Negative/zero incorrect score → not bonus
│
├─► STEP 3: Determine Bonus Type
│   │
│   ├─ If self.marking[Verdict.UNMARKED] > 0:
│   │   RETURN "BONUS_FOR_ALL"
│   │   # Unmarked answers also get marks
│   │   # Example: {"correct": 3, "incorrect": 3, "unmarked": 3}
│   │
│   ├─ Elif self.marking[Verdict.UNMARKED] == 0:
│   │   RETURN "BONUS_ON_ATTEMPT"
│   │   # Only attempted answers get marks
│   │   # Example: {"correct": 3, "incorrect": 3, "unmarked": 0}
│   │
│   └─ Else:
│       RETURN None
│       # Unmarked has negative score → not bonus

END

Examples:
┌─────────────────────────────────────────────────────────────────
│ Example 1: BONUS_FOR_ALL
│ marking = {"correct": 3, "incorrect": 3, "unmarked": 3}
│ → Everyone gets marks regardless of answer
│
│ Example 2: BONUS_ON_ATTEMPT
│ marking = {"correct": 3, "incorrect": 3, "unmarked": 0}
│ → Marks for correct OR incorrect, but not blank
│
│ Example 3: Standard (Not Bonus)
│ marking = {"correct": 3, "incorrect": -1, "unmarked": 0}
│ → incorrect <= 0 → Not bonus
└─────────────────────────────────────────────────────────────────
```

---

## Question Management Flow

### update_questions()

```
START: update_questions(questions)
│
├─► Update Questions List
│   self.questions = questions
│   # E.g., ["q1", "q2", "q3", "q4", "q5"]
│
└─► Revalidate Marking Scheme
    self.validate_marking_scheme()
    # Check streak array lengths against new question count

END
```

### deepcopy_with_questions()

```
START: deepcopy_with_questions(questions)
│
├─► Deep Copy Self
│   clone = deepcopy(self)
│   # Copy all properties including marking, streaks, etc.
│
├─► Update Questions in Clone
│   clone.update_questions(questions)
│   # Update and revalidate
│
└─► RETURN clone

END

Use Case: Create subset marking scheme for conditional sets
Example: Parent has q1..20, child only needs q5..10
```

---

## Integration with Answer Matching

### Flow: Question Evaluation with Section Scheme

```
START: Evaluate Question with Section Scheme
│
├─► Get Answer Matcher
│   answer_matcher = AnswerMatcher(answer_item, section_marking_scheme)
│   # Creates answer matcher with section's marking as base
│
├─► Match Answer
│   question_verdict, delta, current_streak, updated_streak =
│       answer_matcher.get_verdict_marking(marked_answer, allow_streak)
│   │
│   │ Inside get_verdict_marking():
│   │ ├─ Determine question_verdict ("answer-match", "no-answer-match", "unmarked")
│   │ │
│   │ └─ Call section_marking_scheme.get_delta_and_update_streak()
│   │    ├─ Get schema_verdict from question_verdict
│   │    ├─ Get current streak
│   │    ├─ Calculate delta using streak
│   │    ├─ Update streak counters
│   │    └─ Return (delta, current_streak, updated_streak)
│
└─► Update Score
    current_score += delta
    # Apply delta to running total

END
```

---

## Performance Characteristics

### Time Complexity

```
Initialization:
- parse_verdict_marking_from_scheme(): O(V) where V = number of verdicts (always 3)
- parse_fields(): O(Q) where Q = number of questions in section
- validate_marking_scheme(): O(V)
→ Total: O(Q)

Per Question Evaluation:
- get_delta_and_update_streak(): O(1)
- get_delta_for_verdict(): O(1)
- reset_all_streaks(): O(V) = O(1) (V always 3)
→ Total: O(1) per question

Full Section Evaluation (Q questions):
→ O(Q) * O(1) = O(Q)
```

### Space Complexity

```
Per Section Scheme:
- marking dict: 3 entries (correct/incorrect/unmarked)
  - Each entry: number OR array of numbers
  - Worst case (streak): 3 * max_questions * 8 bytes
- questions list: Q * ~50 bytes (average question name)
- streak counters: 3 * 8 bytes (or 2 * 8 bytes for section level)
→ Total: O(Q) + O(streak_levels)

Example:
- 10 questions, no streak: ~600 bytes
- 10 questions, streak with 10 levels: ~900 bytes
```

---

## Browser Migration

### TypeScript Implementation

```typescript
enum MarkingSchemeType {
  DEFAULT = 'default',
  VERDICT_LEVEL_STREAK = 'verdict_level_streak',
  SECTION_LEVEL_STREAK = 'section_level_streak'
}

enum Verdict {
  ANSWER_MATCH = 'answer-match',
  NO_ANSWER_MATCH = 'no-answer-match',
  UNMARKED = 'unmarked'
}

enum SchemaVerdict {
  CORRECT = 'correct',
  INCORRECT = 'incorrect',
  UNMARKED = 'unmarked'
}

type MarkingScore = number | number[];

interface MarkingScores {
  [Verdict.ANSWER_MATCH]: MarkingScore;
  [Verdict.NO_ANSWER_MATCH]: MarkingScore;
  [Verdict.UNMARKED]: MarkingScore;
}

interface SectionScheme {
  marking?: MarkingScores;  // For custom sections
  markingType?: MarkingSchemeType;
  questions?: string[];
  // For DEFAULT section, marking is at top level
  correct?: MarkingScore;
  incorrect?: MarkingScore;
  unmarked?: MarkingScore;
}

class SectionMarkingScheme {
  sectionKey: string;
  setName: string;
  markingType: MarkingSchemeType;
  marking: MarkingScores;
  questions: string[] | null;
  emptyValue: string;

  // Streak tracking
  private streaks: Record<SchemaVerdict, number>;
  private sectionLevelStreak: number;
  private previousStreakVerdict: SchemaVerdict | null;

  constructor(
    sectionKey: string,
    sectionScheme: SectionScheme,
    setName: string,
    emptyValue: string
  ) {
    this.sectionKey = sectionKey;
    this.setName = setName;
    this.emptyValue = emptyValue;
    this.markingType = sectionScheme.markingType || MarkingSchemeType.DEFAULT;

    this.resetAllStreaks();

    // Parse questions and marking
    if (sectionKey === 'DEFAULT') {
      this.questions = null;
      this.marking = this.parseVerdictMarkingFromScheme(sectionScheme);
    } else {
      this.questions = this.parseFields(sectionKey, sectionScheme.questions!);
      this.marking = this.parseVerdictMarkingFromScheme(sectionScheme.marking!);
    }

    this.validateMarkingScheme();
  }

  private resetAllStreaks(): void {
    if (this.markingType === MarkingSchemeType.VERDICT_LEVEL_STREAK) {
      this.streaks = {
        [SchemaVerdict.CORRECT]: 0,
        [SchemaVerdict.INCORRECT]: 0,
        [SchemaVerdict.UNMARKED]: 0
      };
    } else {
      this.sectionLevelStreak = 0;
      this.previousStreakVerdict = null;
    }
  }

  getDeltaAndUpdateStreak(
    answerMatcherMarking: MarkingScores,
    answerType: AnswerType,
    questionVerdict: Verdict,
    allowStreak: boolean
  ): { delta: number; currentStreak: number; updatedStreak: number } {
    const schemaVerdict = AnswerMatcher.getSchemaVerdict(
      answerType,
      questionVerdict,
      0
    );

    let currentStreak: number;
    let updatedStreak: number;
    let delta: number;

    if (this.markingType === MarkingSchemeType.VERDICT_LEVEL_STREAK) {
      currentStreak = this.streaks[schemaVerdict];
      this.resetAllStreaks();

      if (allowStreak && schemaVerdict !== SchemaVerdict.UNMARKED) {
        this.streaks[schemaVerdict] = currentStreak + 1;
      }

      delta = this.getDeltaForVerdict(
        answerMatcherMarking,
        questionVerdict,
        currentStreak
      );
      updatedStreak = this.streaks[schemaVerdict];

    } else if (this.markingType === MarkingSchemeType.SECTION_LEVEL_STREAK) {
      currentStreak = this.sectionLevelStreak;
      const previousVerdict = this.previousStreakVerdict;
      this.resetAllStreaks();

      if ((allowStreak && previousVerdict === null) ||
          schemaVerdict === previousVerdict) {
        this.sectionLevelStreak = currentStreak + 1;
      }

      delta = this.getDeltaForVerdict(
        answerMatcherMarking,
        questionVerdict,
        currentStreak
      );
      updatedStreak = this.sectionLevelStreak;

    } else {
      currentStreak = 0;
      updatedStreak = 0;
      delta = this.getDeltaForVerdict(
        answerMatcherMarking,
        questionVerdict,
        0
      );
    }

    return { delta, currentStreak, updatedStreak };
  }

  private getDeltaForVerdict(
    answerMatcherMarking: MarkingScores,
    questionVerdict: Verdict,
    currentStreak: number
  ): number {
    const marking = answerMatcherMarking[questionVerdict];

    if (Array.isArray(marking)) {
      // Streak array
      if (currentStreak < marking.length) {
        return marking[currentStreak];
      } else {
        // Use last value if streak exceeds array length
        return marking[marking.length - 1];
      }
    } else {
      // Single value
      if (currentStreak > 0) {
        console.warn(
          `Non zero streak(${currentStreak}) for verdict ${questionVerdict}. ` +
          `Using non-streak score for this verdict.`
        );
      }
      return marking;
    }
  }

  private validateMarkingScheme(): void {
    if (
      this.markingType !== MarkingSchemeType.VERDICT_LEVEL_STREAK &&
      this.markingType !== MarkingSchemeType.SECTION_LEVEL_STREAK
    ) {
      return;
    }

    const streakMarkings: Array<[Verdict, number]> = [];
    for (const [verdict, marking] of Object.entries(this.marking)) {
      if (Array.isArray(marking)) {
        streakMarkings.push([verdict as Verdict, marking.length]);
      }
    }

    if (this.questions && this.questions.length > 0 && streakMarkings.length > 0) {
      const maxPossibleStreak = this.questions.length;

      for (const [verdict, length] of streakMarkings) {
        if (length < maxPossibleStreak) {
          console.warn(
            `Marking scheme '${this.sectionKey}': Verdict '${verdict}' has ` +
            `${length} streak levels but section has ${maxPossibleStreak} questions. ` +
            `Consider adding more streak levels or the last level will be used ` +
            `for all remaining streaks.`
          );
        }
      }
    }
  }

  getBonusType(): string | null {
    if (
      this.markingType === MarkingSchemeType.VERDICT_LEVEL_STREAK ||
      this.marking[Verdict.NO_ANSWER_MATCH] <= 0
    ) {
      return null;
    }

    if (this.marking[Verdict.UNMARKED] > 0) {
      return 'BONUS_FOR_ALL';
    }

    if (this.marking[Verdict.UNMARKED] === 0) {
      return 'BONUS_ON_ATTEMPT';
    }

    return null;
  }

  // Helper methods
  private parseFields(sectionKey: string, fields: string[]): string[] {
    // Parse field patterns like "q1..10" into ["q1", "q2", ..., "q10"]
    // Implementation details omitted for brevity
    return fields;
  }

  private parseVerdictMarkingFromScheme(scheme: any): MarkingScores {
    // Parse marking scores (numbers, fractions, or arrays)
    // Implementation details omitted for brevity
    return {} as MarkingScores;
  }
}
```

### Zod Validation

```typescript
import { z } from 'zod';

const MarkingScoreSchema = z.union([
  z.number(),
  z.string().regex(/^-?(\d+)(\/(\d+))?$/),  // Fraction support
  z.array(z.union([z.number(), z.string().regex(/^-?(\d+)(\/(\d+))?$/)]))
]);

const SectionMarkingWithoutStreakSchema = z.object({
  correct: MarkingScoreSchema,
  incorrect: MarkingScoreSchema,
  unmarked: MarkingScoreSchema
});

const CustomSectionMarkingSchema = z.object({
  markingType: z.enum(['default', 'verdict_level_streak', 'section_level_streak']).optional(),
  questions: z.union([z.string(), z.array(z.string())]),
  marking: SectionMarkingWithoutStreakSchema
});

const MarkingSchemesSchema = z.record(
  z.union([
    SectionMarkingWithoutStreakSchema,  // DEFAULT section
    CustomSectionMarkingSchema          // Custom sections
  ])
).refine(
  (schemes) => 'DEFAULT' in schemes,
  { message: 'DEFAULT marking scheme is required' }
);
```

---

## Related Documentation

- **Answer Matcher**: `../answer-matching/flows.md`
- **Evaluation Config**: `../concept.md`
- **Config For Set**: `../config-for-set/flows.md`
- **Evaluation Meta**: `../evaluation-meta/flows.md`

---

## Summary

SectionMarkingScheme provides:

1. **Flexible Scoring**: Per-section custom marking rules
2. **Streak Support**: Bonus/penalty for consecutive answers
3. **Bonus Types**: BONUS_FOR_ALL, BONUS_ON_ATTEMPT
4. **Validation**: Ensures streak arrays match question counts
5. **Efficient**: O(1) per-question evaluation with O(Q) initialization

**Best For**: Complex evaluation scenarios with different scoring for different question types.
**Limitations**: Streak arrays must be pre-defined for all possible streak lengths.
**Typical Usage**: Standard exams (DEFAULT), bonus sections, hard/easy question grouping.
