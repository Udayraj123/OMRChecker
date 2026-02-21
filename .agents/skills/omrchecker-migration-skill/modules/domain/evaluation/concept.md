# Evaluation Config - Concept

**Module**: Domain / Evaluation
**Python Reference**: `src/processors/evaluation/evaluation_config.py`, `src/processors/evaluation/evaluation_config_for_set.py`
**Last Updated**: 2026-02-21

---

## Overview

The Evaluation Config is the central system for scoring OMR responses against answer keys. It manages answer keys, marking schemes, conditional sets, scoring rules, and output configuration for visual feedback.

**Key Responsibilities**:
1. **Answer Key Management**: Load answer keys from local config, CSV files, or images
2. **Marking Schemes**: Define scoring rules (correct/incorrect/unmarked) with support for custom sections
3. **Conditional Sets**: Support multiple answer keys based on response patterns (e.g., different sets)
4. **Score Calculation**: Match student responses against answer keys and calculate scores
5. **Evaluation Metadata**: Generate detailed scoring explanations and visualizations

---

## Evaluation Architecture

### Two-Level Configuration

**EvaluationConfig**: Top-level coordinator
- Manages conditional sets (multiple answer keys)
- Routes responses to appropriate answer key
- Handles set matching based on regex patterns

**EvaluationConfigForSet**: Per-set configuration
- Single answer key and marking scheme
- Question-to-answer matching
- Score calculation and metadata generation

```
EvaluationConfig
├── default_evaluation_config (EvaluationConfigForSet)
├── set_mapping: Dict[str, EvaluationConfigForSet]
└── conditional_sets: List[Tuple[name, matcher]]

EvaluationConfigForSet
├── questions_in_order: List[str]
├── answers_in_order: List[AnswerItem]
├── section_marking_schemes: Dict[str, SectionMarkingScheme]
├── question_to_answer_matcher: Dict[str, AnswerMatcher]
└── outputs_configuration: OutputsConfiguration
```

---

## Core Classes

### EvaluationConfig

**Code Reference**: `src/processors/evaluation/evaluation_config.py:10-122`

**Purpose**: Top-level evaluation orchestrator supporting conditional sets

**Attributes**:
- `path` (Path): Path to evaluation.json file
- `conditional_sets` (list): List of [name, matcher] tuples for set selection
- `default_evaluation_config` (EvaluationConfigForSet): Default answer key
- `set_mapping` (dict): Map from set name to EvaluationConfigForSet
- `exclude_files` (list): Files to exclude from processing (answer key images)

**Key Methods**:
- `get_evaluation_config_for_response(concatenated_response, file_path)`: Route response to appropriate set
- `get_matching_set(concatenated_response, file_path)`: Determine which conditional set matches
- `validate_conditional_sets()`: Ensure no duplicate set names

**Initialization Flow**:
1. Load evaluation.json
2. Extract conditional_sets (if any)
3. Create default_evaluation_config from base JSON
4. For each conditional set:
   - Validate questions/answers consistency
   - Merge with partial defaults (outputs_configuration)
   - Create EvaluationConfigForSet
   - Add to set_mapping

---

### EvaluationConfigForSet

**Code Reference**: `src/processors/evaluation/evaluation_config_for_set.py:31-809`

**Purpose**: Single answer key configuration with scoring logic

**Attributes**:
- `set_name` (str): Name of this set (DEFAULT or conditional set name)
- `questions_in_order` (list[str]): Ordered list of question field names
- `answers_in_order` (list): Ordered list of answer items (types vary)
- `section_marking_schemes` (dict): Custom marking schemes by section
- `default_marking_scheme` (SectionMarkingScheme): Default marking for questions
- `question_to_answer_matcher` (dict): Map from question to AnswerMatcher
- `question_to_scheme` (dict): Map from question to SectionMarkingScheme
- `schema_verdict_counts` (dict): Count of correct/incorrect/unmarked verdicts
- `explanation_table` (Table): Rich table for scoring explanation
- `has_custom_marking` (bool): Whether custom marking schemes exist
- `has_streak_marking` (bool): Whether streak bonuses exist
- `has_conditional_sets` (bool): Whether this is part of conditional sets

**Configuration Fields** (from outputs_configuration):
- `draw_score`: Draw final score on image
- `draw_answers_summary`: Draw verdict summary (correct/incorrect/unmarked counts)
- `draw_question_verdicts`: Draw per-question verdict symbols
- `draw_detected_bubble_texts`: Draw detected bubble values
- `should_explain_scoring`: Print explanation table to console
- `should_export_explanation_csv`: Export explanation as CSV

**Key Methods**:
- `prepare_and_validate_omr_response()`: Validate response has all questions
- `match_answer_for_question()`: Match a single question and return verdict/delta
- `reset_evaluation()`: Reset verdict counts and explanation table
- `get_formatted_answers_summary()`: Format summary string (e.g., "Correct: 8, Incorrect: 2")
- `get_formatted_score()`: Format score string (e.g., "Score: 24/30")

---

## Answer Key Sources

### 1. Local Source (sourceType: "local")

**Code Reference**: `evaluation_config_for_set.py:132-137`

**Configuration**:
```json
{
  "sourceType": "local",
  "options": {
    "questionsInOrder": ["q1..10"],
    "answersInOrder": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"]
  }
}
```

**Features**:
- Questions and answers defined directly in evaluation.json
- Most common and straightforward approach
- Supports all answer types (standard, multiple-correct, weighted)

---

### 2. CSV Source (sourceType: "csv" or "image_and_csv")

**Code Reference**: `evaluation_config_for_set.py:139-232`

**Configuration**:
```json
{
  "sourceType": "csv",
  "options": {
    "answerKeyCsvPath": "answer_key.csv",
    "answerKeyImagePath": "answer_key.jpg",  // Optional: generate CSV from image
    "questionsInOrder": ["q1..10"]  // Required if using image
  }
}
```

**CSV Format**:
```csv
q1,A
q2,B
q3,C
q4,"[""A"",""B""]"  // Multiple correct
q5,"[[""A"",3],[""B"",1.5]]"  // Weighted answers
```

**Flow**:
1. If CSV exists: Load and parse CSV
2. If CSV missing and image provided:
   - Process answer key image through template
   - Extract OMR response
   - Validate no empty answers
   - Use as answer key (optionally save CSV)

**Answer Parsing** (`parse_answer_column`):
- Removes whitespaces
- Detects answer type:
  - Starts with `[`: Evaluate as literal (ast.literal_eval)
  - Contains `,`: Split as multiple-correct
  - Otherwise: Single correct answer

---

## Answer Types

### 1. Standard Answer (AnswerType.STANDARD)

**Format**: String (e.g., "A", "01", "ABC")

**Example**:
```json
"answersInOrder": ["A", "B", "C"]
```

**Matching Logic**:
- Exact match required
- Multi-marked bubbles (e.g., "AB") treated as single answer

---

### 2. Multiple Correct (AnswerType.MULTIPLE_CORRECT)

**Format**: Array of strings (e.g., ["A", "B", "AB"])

**Example**:
```json
"answersInOrder": [
  ["A", "B"],           // q1: Accept either A or B
  ["01", "1"],          // q2: Accept 01 or 1
  ["A", "B", "AB"]      // q3: Accept A, B, or AB
]
```

**Matching Logic**:
- Any answer in array counts as correct
- Each correct answer gets same score from marking scheme
- Useful for ambiguous questions or bonus questions

---

### 3. Multiple Correct Weighted (AnswerType.MULTIPLE_CORRECT_WEIGHTED)

**Format**: Array of [answer, score] tuples

**Example**:
```json
"answersInOrder": [
  [["A", 3], ["C", -2]],           // q1: A gives +3, C gives -2
  [["B", 2], ["C", "3/2"]],        // q2: B gives +2, C gives +1.5
  [["AB", 2.5], ["A", 1], ["B", 1]] // q3: Partial credit for partial answers
]
```

**Matching Logic**:
- Custom score for each answer
- Overrides section marking scheme
- Supports fractions and negative scores
- Useful for partial credit

---

## Marking Schemes

### Default Marking Scheme

**Code Reference**: `evaluation_config_for_set.py:362-373`

**Format**:
```json
{
  "markingSchemes": {
    "DEFAULT": {
      "correct": "3",      // +3 for correct
      "incorrect": "-1",   // -1 for incorrect
      "unmarked": "0"      // 0 for unmarked
    }
  }
}
```

**Applied To**: All questions not in custom sections

---

### Custom Section Marking Schemes

**Code Reference**: `section_marking_scheme.py:18-177`

**Format**:
```json
{
  "markingSchemes": {
    "DEFAULT": { "correct": "3", "incorrect": "-1", "unmarked": "0" },
    "SECTION_A": {
      "questions": ["q1..5"],
      "marking": {
        "correct": "4",     // +4 for section A
        "incorrect": "-2",
        "unmarked": "0"
      }
    },
    "BONUS_SECTION": {
      "questions": ["q10"],
      "marking": {
        "correct": "3",
        "incorrect": "3",   // Bonus: incorrect gives +3
        "unmarked": "0"     // Only on attempt
      }
    }
  }
}
```

**Features**:
- Per-section scoring rules
- Questions must be disjoint across sections
- Supports bonus marking patterns

---

### Bonus Marking Types

**Code Reference**: `section_marking_scheme.py:166-177`

**1. BONUS_ON_ATTEMPT**:
- `correct > 0`, `incorrect > 0`, `unmarked = 0`
- Student gets bonus if they attempt (regardless of correctness)

**2. BONUS_FOR_ALL**:
- `correct > 0`, `incorrect > 0`, `unmarked > 0`
- Everyone gets bonus (even if not attempted)

**Visual Feedback**:
- Bonus questions highlighted in special color
- Symbols: `*` for bonus, `+` for positive, `-` for negative, `o` for neutral

---

### Streak Marking

**Code Reference**: `section_marking_scheme.py:38-105`

**Marking Types**:
1. `DEFAULT`: Static scores (most common)
2. `VERDICT_LEVEL_STREAK`: Bonus for consecutive correct/incorrect/unmarked
3. `SECTION_LEVEL_STREAK`: Bonus for consecutive same verdicts in section

**Verdict-Level Streak Example**:
```json
{
  "STREAK_SECTION": {
    "markingType": "VERDICT_LEVEL_STREAK",
    "questions": ["q1..10"],
    "marking": {
      "correct": [3, 4, 5, 6, 7],  // Increasing bonus for streak
      "incorrect": "-1",
      "unmarked": "0"
    }
  }
}
```

**Logic**:
- Each verdict (correct/incorrect/unmarked) has independent streak counter
- On new verdict, all streaks reset
- Array index = current streak length
- If streak exceeds array length, use last value

---

## Conditional Sets (Multi-Set Support)

**Code Reference**: `evaluation_config.py:17-77`

**Purpose**: Support multiple answer keys based on response patterns

**Use Cases**:
- Multiple question papers (Set A, Set B, Set C)
- Different answer keys based on class/section
- Dynamic routing based on barcode/roll number

**Configuration**:
```json
{
  "sourceType": "local",
  "options": {
    "questionsInOrder": ["q1..20"],
    "answersInOrder": ["B", "D", "C", ...]  // Default answers
  },
  "markingSchemes": { "DEFAULT": { ... } },
  "conditionalSets": [
    {
      "name": "Set A",
      "matcher": {
        "formatString": "{q21}",  // Use q21 field
        "matchRegex": "A"         // Match if q21 == "A"
      },
      "evaluation": {
        "sourceType": "local",
        "options": {
          "questionsInOrder": [],  // Empty: use parent questions
          "answersInOrder": []     // Empty: use parent answers
        },
        "markingSchemes": {
          "DEFAULT": {
            "correct": "4",  // Override: +4 instead of +3
            "incorrect": "0",
            "unmarked": "0"
          }
        }
      }
    }
  ]
}
```

**Matcher**:
- `formatString`: Python format string with field names (e.g., `{Roll}`, `{barcode}`, `{file_name}`)
- `matchRegex`: Regex to match formatted string
- Available fields: All OMR response fields + `file_path`, `file_name`

**Merging Logic**:
1. Start with partial defaults (only outputs_configuration)
2. Merge conditional set evaluation JSON
3. For questions/answers:
   - Merge with parent questions (parent questions come first)
   - Override parent answers if same question exists
   - Append new questions at end
4. For marking schemes:
   - Inherit parent custom schemes for questions not in child
   - Prefix inherited schemes with "parent-"

**Validation**:
- No duplicate set names
- If `answersInOrder` provided, `questionsInOrder` must also be provided
- No overlapping questions across sections

---

## Question-to-Answer Matching

**Code Reference**: `answer_matcher.py:15-229`

### AnswerMatcher Class

**Purpose**: Match student response against answer key for a single question

**Attributes**:
- `answer_item`: The correct answer(s)
- `answer_type`: AnswerType (STANDARD, MULTIPLE_CORRECT, MULTIPLE_CORRECT_WEIGHTED)
- `section_marking_scheme`: Associated marking scheme
- `marking`: Local marking scores (may override section scheme)
- `empty_value`: Empty value for unmarked bubbles

**Key Method**: `get_verdict_marking(marked_answer, allow_streak=False)`

**Returns**: `(question_verdict, delta, current_streak, updated_streak)`

**Verdicts**:
- `Verdict.UNMARKED`: Student didn't mark anything
- `Verdict.ANSWER_MATCH`: Correct answer (or `ANSWER_MATCH-A` for multi-correct)
- `Verdict.NO_ANSWER_MATCH`: Incorrect answer

**Delta Calculation**:
1. Determine question verdict based on answer type
2. Get schema verdict (correct/incorrect/unmarked)
3. Consult marking scheme for delta
4. Handle streak bonuses (if enabled)
5. Return delta and updated streak

---

## Evaluation Flow

### High-Level Flow

**Code Reference**: `evaluation_meta.py:57-96`

```
1. Get evaluation config for response (conditional set matching)
2. Prepare and validate OMR response (check all questions present)
3. For each question in order:
   a. Get marked answer from response
   b. Match answer and get verdict/delta
   c. Update score
   d. Add question metadata
   e. Update explanation table (if enabled)
4. Print explanation table (if enabled)
5. Export explanation CSV (if enabled)
6. Return score and evaluation metadata
```

**Code Example**:
```python
def evaluate_concatenated_response(concatenated_response, evaluation_config_for_response):
    evaluation_config_for_response.prepare_and_validate_omr_response(
        concatenated_response, allow_streak=True
    )
    evaluation_meta = EvaluationMeta()
    for question in evaluation_config_for_response.questions_in_order:
        marked_answer = concatenated_response[question]
        (delta, question_verdict, answer_matcher, question_schema_verdict) =
            evaluation_config_for_response.match_answer_for_question(
                evaluation_meta.score, question, marked_answer
            )
        marking_scheme = evaluation_config_for_response.get_marking_scheme_for_question(question)
        bonus_type = marking_scheme.get_bonus_type()
        evaluation_meta.score += delta
        question_meta = QuestionMeta(...)
        evaluation_meta.add_question_meta(question, question_meta)

    evaluation_config_for_response.conditionally_print_explanation()
    formatted_answers_summary = evaluation_config_for_response.get_formatted_answers_summary()
    return evaluation_meta.score, evaluation_meta.to_dict(formatted_answers_summary)
```

---

## Evaluation Metadata

### QuestionMeta

**Code Reference**: `evaluation_meta.py:4-37`

**Per-Question Metadata**:
- `question` (str): Question field name
- `question_verdict` (str): Verdict (UNMARKED, ANSWER_MATCH, NO_ANSWER_MATCH)
- `marked_answer` (str): Student's answer
- `delta` (float): Score change for this question
- `current_score` (float): Cumulative score after this question
- `answer_item`: Correct answer(s)
- `answer_type` (AnswerType): Answer type
- `bonus_type` (str): Bonus type (if any)
- `question_schema_verdict` (SchemaVerdict): correct/incorrect/unmarked

---

### EvaluationMeta

**Code Reference**: `evaluation_meta.py:40-53`

**Overall Evaluation Metadata**:
- `score` (float): Final score
- `questions_meta` (dict): Map from question to QuestionMeta dict
- `formatted_answers_summary` (str): Summary string (e.g., "Correct: 8, Incorrect: 2")

**Output Format**:
```json
{
  "score": 24.0,
  "questions_meta": {
    "q1": {
      "question_verdict": "ANSWER_MATCH",
      "marked_answer": "A",
      "delta": 3.0,
      "current_score": 3.0,
      "answer_item": "A",
      "answer_type": "STANDARD",
      "bonus_type": null,
      "question_schema_verdict": "correct"
    },
    "q2": { ... }
  },
  "formatted_answers_summary": "Correct: 8, Incorrect: 2, Unmarked: 0"
}
```

---

## Visual Outputs

### 1. Draw Score

**Configuration**:
```json
{
  "drawScore": {
    "enabled": true,
    "position": [650, 750],
    "scoreFormatString": "Score: {score}",
    "size": 1.5
  }
}
```

**Output**: Score text on image at specified position

---

### 2. Draw Answers Summary

**Configuration**:
```json
{
  "drawAnswersSummary": {
    "enabled": true,
    "position": [650, 820],
    "answersSummaryFormatString": "Correct: {correct}, Incorrect: {incorrect}, Unmarked: {unmarked}",
    "size": 0.85
  }
}
```

**Variables**: `{correct}`, `{incorrect}`, `{unmarked}`, `{bonus}`

---

### 3. Draw Question Verdicts

**Code Reference**: `evaluation_config_for_set.py:714-809`

**Configuration**:
```json
{
  "drawQuestionVerdicts": {
    "enabled": true,
    "verdictColors": {
      "correct": "#00FF00",
      "neutral": null,
      "incorrect": "#FF0000",
      "bonus": "#00DDDD"
    },
    "verdictSymbolColors": {
      "positive": "#000000",
      "neutral": "#000000",
      "negative": "#000000",
      "bonus": "#000000"
    },
    "drawAnswerGroups": {
      "enabled": true,
      "colorSequence": ["#8DFBC4", "#F7FB8D", "#8D9EFB", "#EA666F"]
    }
  }
}
```

**Verdict Symbols**:
- `+`: Positive delta (correct)
- `-`: Negative delta (incorrect)
- `o`: Neutral delta (zero)
- `*`: Bonus question
- ` `: Unmarked bubble

**Colors**:
- Field box colored by verdict (correct/incorrect/neutral/bonus)
- Symbol colored by delta sign
- Answer groups highlighted with color sequence

---

### 4. Explanation Table

**Code Reference**: `evaluation_config_for_set.py:694-713`

**Configuration**:
```json
{
  "shouldExplainScoring": true
}
```

**Output**: Rich table printed to console

**Columns**:
- Marking Scheme (if custom marking)
- Question
- Marked (student answer)
- Answer(s) (correct answer)
- Verdict (schema verdict + question verdict)
- Delta (score change)
- Score (cumulative score)
- Set Mapping (if conditional sets)
- Streak (if streak marking)

**Example**:
```
┌─────────────┬────────┬────────┬─────────┬───────────┬───────┬───────┐
│ Question    │ Marked │ Answer │ Verdict │ Delta     │ Score │       │
├─────────────┼────────┼────────┼─────────┼───────────┼───────┼───────┤
│ q1          │ A      │ A      │ Correct │ 3.0       │ 3.0   │       │
│ q2          │ C      │ B      │ Incorrect│ -1.0     │ 2.0   │       │
│ q3          │        │ C      │ Unmarked│ 0.0       │ 2.0   │       │
└─────────────┴────────┴────────┴─────────┴───────────┴───────┴───────┘
```

---

## Browser Migration Notes

### TypeScript Interfaces

```typescript
interface EvaluationConfig {
    path: string;
    conditionalSets: Array<[string, Matcher]>;
    defaultEvaluationConfig: EvaluationConfigForSet;
    setMapping: Map<string, EvaluationConfigForSet>;
    excludeFiles: string[];

    getEvaluationConfigForResponse(
        response: OMRResponse,
        filePath: string
    ): EvaluationConfigForSet;
}

interface EvaluationConfigForSet {
    setName: string;
    questionsInOrder: string[];
    answersInOrder: AnswerItem[];
    sectionMarkingSchemes: Map<string, SectionMarkingScheme>;
    defaultMarkingScheme: SectionMarkingScheme;
    questionToAnswerMatcher: Map<string, AnswerMatcher>;
    schemaVerdictCounts: Record<SchemaVerdict, number>;

    matchAnswerForQuestion(
        currentScore: number,
        question: string,
        markedAnswer: string
    ): { delta: number; verdict: string; matcher: AnswerMatcher; schemaVerdict: string };
}

type AnswerItem =
    | string  // Standard
    | string[]  // Multiple correct
    | Array<[string, number]>;  // Multiple correct weighted
```

---

### Zod Validation

```typescript
import { z } from 'zod';

const MarkingScoreSchema = z.union([
    z.string().regex(/-?\d+(\/\d+)?/),  // Fraction string
    z.number(),  // Numeric
    z.array(z.union([z.string(), z.number()]))  // Streak array
]);

const MarkingSchemeSchema = z.object({
    correct: MarkingScoreSchema,
    incorrect: MarkingScoreSchema,
    unmarked: MarkingScoreSchema
});

const AnswerItemSchema = z.union([
    z.string(),  // Standard
    z.array(z.string()).min(2),  // Multiple correct
    z.array(z.tuple([z.string(), MarkingScoreSchema])).min(1)  // Weighted
]);

const EvaluationSchema = z.object({
    sourceType: z.enum(['local', 'csv', 'image_and_csv']),
    options: z.object({
        questionsInOrder: z.array(z.string()).optional(),
        answersInOrder: z.array(AnswerItemSchema).optional(),
        answerKeyCsvPath: z.string().optional(),
        answerKeyImagePath: z.string().optional()
    }),
    markingSchemes: z.record(z.union([
        MarkingSchemeSchema,
        z.object({
            questions: z.array(z.string()),
            marking: MarkingSchemeSchema,
            markingType: z.enum(['DEFAULT', 'VERDICT_LEVEL_STREAK', 'SECTION_LEVEL_STREAK']).optional()
        })
    ])),
    conditionalSets: z.array(z.object({
        name: z.string(),
        matcher: z.object({
            formatString: z.string(),
            matchRegex: z.string()
        }),
        evaluation: z.lazy(() => EvaluationSchema)  // Recursive
    })).optional()
});
```

---

### CSV Parsing

```typescript
// Browser: Use papaparse or custom CSV parser
import Papa from 'papaparse';

async function loadAnswerKeyFromCSV(csvPath: string): Promise<Map<string, AnswerItem>> {
    const response = await fetch(csvPath);
    const csvText = await response.text();

    const parsed = Papa.parse<[string, string]>(csvText, {
        header: false,
        skipEmptyLines: true
    });

    const answerKey = new Map<string, AnswerItem>();
    for (const [question, answerStr] of parsed.data) {
        const answer = parseAnswerColumn(answerStr.trim());
        answerKey.set(question.trim(), answer);
    }

    return answerKey;
}

function parseAnswerColumn(answerStr: string): AnswerItem {
    // Remove whitespaces
    answerStr = answerStr.replace(/\s/g, '');

    if (answerStr.startsWith('[')) {
        // Parse as JSON
        return JSON.parse(answerStr);
    } else if (answerStr.includes(',')) {
        // Multiple correct
        return answerStr.split(',');
    } else {
        // Standard
        return answerStr;
    }
}
```

---

### Regex Matching

```typescript
// Browser: Native RegExp support
function matchConditionalSet(
    response: OMRResponse,
    filePath: string,
    conditionalSets: Array<[string, Matcher]>
): string | null {
    const formattingFields = {
        ...response,
        file_path: filePath,
        file_name: filePath.split('/').pop() || ''
    };

    for (const [name, matcher] of conditionalSets) {
        const { formatString, matchRegex } = matcher;
        try {
            // Simple template string replacement
            const formatted = formatString.replace(
                /\{(\w+)\}/g,
                (_, key) => formattingFields[key] || ''
            );
            const regex = new RegExp(matchRegex);
            if (regex.test(formatted)) {
                return name;
            }
        } catch (e) {
            console.error('Regex matching error:', e);
        }
    }

    return null;
}
```

---

### Explanation Export

```typescript
// Browser: Export as downloadable CSV
function exportExplanationCSV(
    explanationData: Array<Record<string, any>>,
    fileName: string
): void {
    const csv = Papa.unparse(explanationData, {
        quotes: true,
        header: true
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${fileName}.csv`;
    link.click();
    URL.revokeObjectURL(url);
}
```

---

## Summary

**Evaluation Config**: Two-level scoring system with conditional sets
**Answer Sources**: Local (JSON), CSV, or image-based answer keys
**Answer Types**: Standard, multiple-correct, multiple-correct-weighted
**Marking Schemes**: Default + custom sections + bonus + streak bonuses
**Conditional Sets**: Multi-set support with regex-based routing
**Metadata**: Detailed per-question scoring with explanation tables
**Visual Outputs**: Score, summary, verdict symbols on image

**Browser Migration**:
- Use Zod for validation
- Native RegExp for conditional set matching
- papaparse for CSV parsing
- Blob/download for explanation export
- Maintain same scoring logic and metadata structure
- Consider React/Vue table components for explanation display

**Key Takeaway**: Evaluation Config is a flexible scoring engine supporting diverse use cases (single/multi-set, standard/weighted, bonus/streak). Browser version should maintain same flexibility with web-native implementations for file I/O and validation.
