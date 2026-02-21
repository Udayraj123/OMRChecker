# Evaluation Meta Flows

**Module**: Domain - Evaluation - Meta
**Python Reference**: `src/processors/evaluation/evaluation_meta.py`
**Last Updated**: 2026-02-21

---

## Overview

Evaluation Meta generates comprehensive metadata about the evaluation process, tracking per-question scoring details and explanations. It captures the verdict, delta, running score, answer type, and bonus information for every question.

**Use Case**: Provide detailed scoring breakdowns for students, instructors, and debugging. Enables understanding of why each question was scored the way it was.

---

## Core Entities

### QuestionMeta Class

```python
class QuestionMeta:
    def __init__(
        question,
        question_verdict,
        marked_answer,
        delta,
        current_score,
        answer_matcher,
        bonus_type,
        question_schema_verdict,
    )
```

**Purpose**: Encapsulates all metadata for a single question

**Fields**:
- `question`: Question identifier (e.g., "q1", "roll_1")
- `question_verdict`: Detailed verdict (e.g., "answer-match", "no-answer-match", "answer-match-A")
- `marked_answer`: Student's response (e.g., "A", "AB", "")
- `delta`: Score change for this question (e.g., +1.0, -0.25, 0.0)
- `current_score`: Running total after this question
- `answer_item`: Correct answer(s) from answer_matcher
- `answer_type`: One of "standard", "multiple-correct", "multiple-correct-weighted"
- `bonus_type`: Bonus section identifier or None
- `question_schema_verdict`: Simplified verdict ("correct", "incorrect", "unmarked")

### EvaluationMeta Class

```python
class EvaluationMeta:
    def __init__(self)
        self.score = 0.0
        self.questions_meta = {}
```

**Purpose**: Aggregate container for all question metadata

**Fields**:
- `score`: Final total score
- `questions_meta`: Dict mapping question → QuestionMeta.to_dict()

---

## Main Flow

### evaluate_concatenated_response()

**Entry Point**: Main utility function to evaluate and generate metadata

```
START: evaluate_concatenated_response(concatenated_response, evaluation_config_for_response)
│
├─► STEP 1: Validate and Prepare Response
│   │
│   evaluation_config_for_response.prepare_and_validate_omr_response(
│       concatenated_response,
│       allow_streak=True
│   )
│   │
│   │ Purpose: Validate response format and prepare for evaluation
│   │ - Check all required questions are present
│   │ - Validate answer format
│   │ - Enable streak bonuses if applicable
│
├─► STEP 2: Initialize Evaluation Meta
│   │
│   evaluation_meta = EvaluationMeta()
│   │
│   │ Initial state:
│   │ - score: 0.0
│   │ - questions_meta: {}
│
├─► STEP 3: Process Each Question in Order
│   │
│   For question in evaluation_config_for_response.questions_in_order:
│   │
│   ├─► 3.1: Extract Marked Answer
│   │   │
│   │   marked_answer = concatenated_response[question]
│   │   │
│   │   │ Example:
│   │   │ question = "q1"
│   │   │ concatenated_response = {"q1": "A", "q2": "B", ...}
│   │   │ marked_answer = "A"
│   │
│   ├─► 3.2: Match Answer for Question
│   │   │
│   │   (
│   │       delta,
│   │       question_verdict,
│   │       answer_matcher,
│   │       question_schema_verdict,
│   │   ) = evaluation_config_for_response.match_answer_for_question(
│   │       evaluation_meta.score,  # Current score (for streak bonuses)
│   │       question,
│   │       marked_answer
│   │   )
│   │   │
│   │   │ Returns:
│   │   │ - delta: Score change (e.g., +1.0, -0.25, 0.0)
│   │   │ - question_verdict: Detailed verdict string
│   │   │ - answer_matcher: AnswerMatcher instance with answer_item/answer_type
│   │   │ - question_schema_verdict: "correct", "incorrect", or "unmarked"
│   │   │
│   │   │ Example for correct answer:
│   │   │ delta = +1.0
│   │   │ question_verdict = "answer-match"
│   │   │ answer_matcher.answer_item = "A"
│   │   │ answer_matcher.answer_type = "standard"
│   │   │ question_schema_verdict = "correct"
│   │   │
│   │   │ Example for multiple correct:
│   │   │ delta = +1.0
│   │   │ question_verdict = "answer-match-BC"
│   │   │ answer_matcher.answer_item = ["A", "BC", "C"]
│   │   │ answer_matcher.answer_type = "multiple-correct"
│   │   │ question_schema_verdict = "correct"
│   │   │
│   │   │ Example for weighted answer:
│   │   │ delta = +0.5
│   │   │ question_verdict = "answer-match-B"
│   │   │ answer_matcher.answer_item = [["A", 1], ["B", 0.5]]
│   │   │ answer_matcher.answer_type = "multiple-correct-weighted"
│   │   │ question_schema_verdict = "correct"
│   │
│   ├─► 3.3: Get Bonus Type
│   │   │
│   │   marking_scheme = evaluation_config_for_response.get_marking_scheme_for_question(
│   │       question
│   │   )
│   │   bonus_type = marking_scheme.get_bonus_type()
│   │   │
│   │   │ Returns:
│   │   │ - None: Regular question
│   │   │ - "BONUS_SECTION_1": Question in bonus section 1
│   │   │ - "BONUS_TOTAL": Bonus based on total score
│   │
│   ├─► 3.4: Update Running Score
│   │   │
│   │   evaluation_meta.score += delta
│   │   │
│   │   │ Example progression:
│   │   │ Question 1: 0.0 + 1.0 = 1.0
│   │   │ Question 2: 1.0 + 1.0 = 2.0
│   │   │ Question 3: 2.0 + (-0.25) = 1.75
│   │   │ Question 4: 1.75 + 0.0 = 1.75 (unmarked)
│   │
│   ├─► 3.5: Create Question Meta
│   │   │
│   │   question_meta = QuestionMeta(
│   │       question,
│   │       question_verdict,
│   │       marked_answer,
│   │       delta,
│   │       evaluation_meta.score,  # Current score AFTER this question
│   │       answer_matcher,
│   │       bonus_type,
│   │       question_schema_verdict,
│   │   )
│   │
│   └─► 3.6: Add to Evaluation Meta
│       │
│       evaluation_meta.add_question_meta(question, question_meta)
│       │
│       │ Stores question_meta.to_dict() in evaluation_meta.questions_meta
│
├─► STEP 4: Print Explanation (Optional)
│   │
│   evaluation_config_for_response.conditionally_print_explanation()
│   │
│   │ If enabled in config:
│   │ - Prints detailed scoring explanation to console
│   │ - Shows per-question breakdown
│   │ - Useful for debugging
│
├─► STEP 5: Get Formatted Summary
│   │
│   (
│       formatted_answers_summary,
│       *_,
│   ) = evaluation_config_for_response.get_formatted_answers_summary()
│   │
│   │ Returns summary like:
│   │ "Correct: 45 Incorrect: 3 Unmarked: 2"
│
└─► STEP 6: Return Results
    │
    RETURN (
        evaluation_meta.score,
        evaluation_meta.to_dict(formatted_answers_summary)
    )
    │
    │ Returns tuple:
    │ - score: float (final score)
    │ - meta_dict: dict with structure:
    │   {
    │       "score": 45.0,
    │       "questions_meta": {
    │           "q1": {
    │               "question_verdict": "answer-match",
    │               "marked_answer": "A",
    │               "delta": 1.0,
    │               "current_score": 1.0,
    │               "answer_item": "A",
    │               "answer_type": "standard",
    │               "bonus_type": None,
    │               "question_schema_verdict": "correct"
    │           },
    │           "q2": {...},
    │           ...
    │       },
    │       "formatted_answers_summary": "Correct: 45 Incorrect: 3 Unmarked: 2"
    │   }

END
```

---

## QuestionMeta.to_dict() Flow

```
START: QuestionMeta.to_dict()
│
└─► Return Dictionary with All Fields (except question itself)
    │
    RETURN {
        "question_verdict": self.question_verdict,
        "marked_answer": self.marked_answer,
        "delta": self.delta,
        "current_score": self.current_score,
        "answer_item": self.answer_item,      # From answer_matcher
        "answer_type": self.answer_type,      # From answer_matcher
        "bonus_type": self.bonus_type,
        "question_schema_verdict": self.question_schema_verdict,
    }

END

Note: 'question' field is NOT included in dict (used as key in parent questions_meta dict)
```

---

## EvaluationMeta.to_dict() Flow

```
START: EvaluationMeta.to_dict(formatted_answers_summary)
│
└─► Combine Score, Question Metadata, and Summary
    │
    RETURN {
        "score": self.score,
        "questions_meta": self.questions_meta,  # Already dict of dicts
        "formatted_answers_summary": formatted_answers_summary
    }

END
```

---

## Detailed Examples

### Example 1: Standard Answer Type

```python
# Setup
concatenated_response = {
    "q1": "A",
    "q2": "C",
    "q3": "",  # Unmarked
}

# Evaluation config has answers: {"q1": "A", "q2": "B", "q3": "C"}
# Marking: correct=+1, incorrect=-0.25, unmarked=0

# Flow for q1:
marked_answer = "A"
delta, question_verdict, answer_matcher, schema_verdict = match_answer_for_question(0.0, "q1", "A")
# Returns: (1.0, "answer-match", AnswerMatcher(...), "correct")

bonus_type = marking_scheme.get_bonus_type()
# Returns: None

evaluation_meta.score += 1.0  # 0.0 → 1.0

question_meta = QuestionMeta(
    question="q1",
    question_verdict="answer-match",
    marked_answer="A",
    delta=1.0,
    current_score=1.0,
    answer_matcher=answer_matcher,
    bonus_type=None,
    question_schema_verdict="correct"
)

evaluation_meta.add_question_meta("q1", question_meta)

# Flow for q2:
marked_answer = "C"
delta, question_verdict, answer_matcher, schema_verdict = match_answer_for_question(1.0, "q2", "C")
# Returns: (-0.25, "no-answer-match", AnswerMatcher(...), "incorrect")

evaluation_meta.score += -0.25  # 1.0 → 0.75

question_meta = QuestionMeta(
    question="q2",
    question_verdict="no-answer-match",
    marked_answer="C",
    delta=-0.25,
    current_score=0.75,
    answer_matcher=answer_matcher,
    bonus_type=None,
    question_schema_verdict="incorrect"
)

# Flow for q3:
marked_answer = ""
delta, question_verdict, answer_matcher, schema_verdict = match_answer_for_question(0.75, "q3", "")
# Returns: (0.0, "unmarked", AnswerMatcher(...), "unmarked")

evaluation_meta.score += 0.0  # 0.75 → 0.75

# Final result
score, meta = evaluate_concatenated_response(...)
# score = 0.75
# meta = {
#     "score": 0.75,
#     "questions_meta": {
#         "q1": {
#             "question_verdict": "answer-match",
#             "marked_answer": "A",
#             "delta": 1.0,
#             "current_score": 1.0,
#             "answer_item": "A",
#             "answer_type": "standard",
#             "bonus_type": None,
#             "question_schema_verdict": "correct"
#         },
#         "q2": {
#             "question_verdict": "no-answer-match",
#             "marked_answer": "C",
#             "delta": -0.25,
#             "current_score": 0.75,
#             "answer_item": "B",
#             "answer_type": "standard",
#             "bonus_type": None,
#             "question_schema_verdict": "incorrect"
#         },
#         "q3": {
#             "question_verdict": "unmarked",
#             "marked_answer": "",
#             "delta": 0.0,
#             "current_score": 0.75,
#             "answer_item": "C",
#             "answer_type": "standard",
#             "bonus_type": None,
#             "question_schema_verdict": "unmarked"
#         }
#     },
#     "formatted_answers_summary": "Correct: 1 Incorrect: 1 Unmarked: 1"
# }
```

### Example 2: Multiple Correct Answer Type

```python
# Setup
concatenated_response = {"q1": "BC"}

# Evaluation config: answer_item = ["A", "BC", "C"]
# All three answers are equally correct (bonus question)

# Flow:
marked_answer = "BC"
delta, question_verdict, answer_matcher, schema_verdict = match_answer_for_question(0.0, "q1", "BC")
# Returns: (1.0, "answer-match-BC", AnswerMatcher(...), "correct")

# answer_matcher fields:
# answer_matcher.answer_item = ["A", "BC", "C"]
# answer_matcher.answer_type = "multiple-correct"

question_meta = QuestionMeta(
    question="q1",
    question_verdict="answer-match-BC",  # Includes which answer matched
    marked_answer="BC",
    delta=1.0,
    current_score=1.0,
    answer_matcher=answer_matcher,
    bonus_type=None,
    question_schema_verdict="correct"
)

# Stored metadata shows WHICH correct answer was chosen:
# {
#     "question_verdict": "answer-match-BC",  # Not just "answer-match"
#     "marked_answer": "BC",
#     "answer_item": ["A", "BC", "C"],        # All accepted answers
#     "answer_type": "multiple-correct"
# }
```

### Example 3: Weighted Answer Type

```python
# Setup
concatenated_response = {"q1": "B"}

# Evaluation config: answer_item = [["A", 1.0], ["B", 0.5], ["C", 0.0]]
# A is fully correct, B is partially correct, C gives no credit

# Flow:
marked_answer = "B"
delta, question_verdict, answer_matcher, schema_verdict = match_answer_for_question(0.0, "q1", "B")
# Returns: (0.5, "answer-match-B", AnswerMatcher(...), "correct")

# answer_matcher fields:
# answer_matcher.answer_item = [["A", 1.0], ["B", 0.5], ["C", 0.0]]
# answer_matcher.answer_type = "multiple-correct-weighted"

question_meta = QuestionMeta(
    question="q1",
    question_verdict="answer-match-B",
    marked_answer="B",
    delta=0.5,  # Partial credit!
    current_score=0.5,
    answer_matcher=answer_matcher,
    bonus_type=None,
    question_schema_verdict="correct"  # Still "correct" even though partial
)

# Metadata shows partial credit:
# {
#     "delta": 0.5,  # Not full 1.0
#     "answer_item": [["A", 1.0], ["B", 0.5], ["C", 0.0]],
#     "answer_type": "multiple-correct-weighted"
# }
```

### Example 4: Bonus Section

```python
# Setup
concatenated_response = {"bonus1": "A"}

# Question "bonus1" is in a BONUS section

# Flow:
marked_answer = "A"
delta, question_verdict, answer_matcher, schema_verdict = match_answer_for_question(10.0, "bonus1", "A")
# Returns: (1.0, "answer-match", AnswerMatcher(...), "correct")

marking_scheme = evaluation_config_for_response.get_marking_scheme_for_question("bonus1")
bonus_type = marking_scheme.get_bonus_type()
# Returns: "BONUS_SECTION_1"  (or similar identifier)

question_meta = QuestionMeta(
    question="bonus1",
    question_verdict="answer-match",
    marked_answer="A",
    delta=1.0,
    current_score=11.0,
    answer_matcher=answer_matcher,
    bonus_type="BONUS_SECTION_1",  # Marked as bonus!
    question_schema_verdict="correct"
)

# Metadata shows bonus:
# {
#     "bonus_type": "BONUS_SECTION_1",  # Not None
#     "delta": 1.0,
#     "question_verdict": "answer-match"
# }
```

---

## Integration Points

### Input: concatenated_response

**Format**: Dictionary mapping question → marked answer

```python
{
    "q1": "A",
    "q2": "BC",
    "roll_1": "01",
    "roll_2": "23",
    ...
}
```

**Source**: Detection phase output (from ReadOMR processor)

### Input: evaluation_config_for_response

**Type**: `EvaluationConfigForSet` instance

**Provides**:
- `questions_in_order`: List of question IDs in evaluation order
- `match_answer_for_question()`: Core matching logic
- `get_marking_scheme_for_question()`: Marking scheme lookup
- `conditionally_print_explanation()`: Debug output
- `get_formatted_answers_summary()`: Summary string generation

### Output: (score, meta_dict)

**score**: `float` - Final total score

**meta_dict**: `dict` - Detailed metadata structure:
```python
{
    "score": 45.0,
    "questions_meta": {
        "q1": QuestionMeta.to_dict(),
        "q2": QuestionMeta.to_dict(),
        ...
    },
    "formatted_answers_summary": "Correct: 45 Incorrect: 3 Unmarked: 2"
}
```

**Usage**: Written to CSV, exported to JSON, displayed in UI

---

## Browser Migration

### TypeScript Implementation

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

class QuestionMeta {
  constructor(
    public question: string,
    public question_verdict: string,
    public marked_answer: string,
    public delta: number,
    public current_score: number,
    public answer_item: string | string[] | [string, number][],
    public answer_type: 'standard' | 'multiple-correct' | 'multiple-correct-weighted',
    public bonus_type: string | null,
    public question_schema_verdict: 'correct' | 'incorrect' | 'unmarked'
  ) {}

  toDict(): QuestionMetaDict {
    return {
      question_verdict: this.question_verdict,
      marked_answer: this.marked_answer,
      delta: this.delta,
      current_score: this.current_score,
      answer_item: this.answer_item,
      answer_type: this.answer_type,
      bonus_type: this.bonus_type,
      question_schema_verdict: this.question_schema_verdict,
    };
  }
}

class EvaluationMeta {
  score: number = 0.0;
  questions_meta: Record<string, QuestionMetaDict> = {};

  addQuestionMeta(question: string, questionMeta: QuestionMeta): void {
    this.questions_meta[question] = questionMeta.toDict();
  }

  toDict(formattedAnswersSummary: string): EvaluationMetaDict {
    return {
      score: this.score,
      questions_meta: this.questions_meta,
      formatted_answers_summary: formattedAnswersSummary,
    };
  }
}

function evaluateConcatenatedResponse(
  concatenatedResponse: Record<string, string>,
  evaluationConfigForResponse: EvaluationConfigForSet
): [number, EvaluationMetaDict] {
  // Step 1: Validate
  evaluationConfigForResponse.prepareAndValidateOmrResponse(
    concatenatedResponse,
    true // allow_streak
  );

  // Step 2: Initialize
  const evaluationMeta = new EvaluationMeta();

  // Step 3: Process each question
  for (const question of evaluationConfigForResponse.questionsInOrder) {
    const markedAnswer = concatenatedResponse[question];

    const [delta, questionVerdict, answerMatcher, questionSchemaVerdict] =
      evaluationConfigForResponse.matchAnswerForQuestion(
        evaluationMeta.score,
        question,
        markedAnswer
      );

    const markingScheme =
      evaluationConfigForResponse.getMarkingSchemeForQuestion(question);
    const bonusType = markingScheme.getBonusType();

    evaluationMeta.score += delta;

    const questionMeta = new QuestionMeta(
      question,
      questionVerdict,
      markedAnswer,
      delta,
      evaluationMeta.score,
      answerMatcher.answerItem,
      answerMatcher.answerType,
      bonusType,
      questionSchemaVerdict
    );

    evaluationMeta.addQuestionMeta(question, questionMeta);
  }

  // Step 4: Print explanation (conditional)
  evaluationConfigForResponse.conditionallyPrintExplanation();

  // Step 5: Get summary
  const [formattedAnswersSummary] =
    evaluationConfigForResponse.getFormattedAnswersSummary();

  // Step 6: Return
  return [evaluationMeta.score, evaluationMeta.toDict(formattedAnswersSummary)];
}
```

### Key Differences from Python

1. **Type Annotations**:
   ```typescript
   // Python: Duck typing
   def to_dict(self):
       return {...}

   // TypeScript: Explicit interfaces
   toDict(): QuestionMetaDict {
       return {...};
   }
   ```

2. **Dictionary Access**:
   ```typescript
   // Python: dict[key]
   marked_answer = concatenated_response[question]

   // TypeScript: Record type
   const markedAnswer: string = concatenatedResponse[question];
   ```

3. **Record vs Dict**:
   ```typescript
   // Python
   questions_meta = {}  # dict

   // TypeScript
   questions_meta: Record<string, QuestionMetaDict> = {};
   ```

4. **Tuple Return**:
   ```typescript
   // Python
   return score, meta_dict

   // TypeScript (use array)
   return [score, metaDict] as [number, EvaluationMetaDict];
   ```

---

## Related Documentation

- **Answer Matcher**: `../answer-matching/flows.md`
- **Evaluation Config**: `../concept.md`
- **Section Marking Scheme**: `../section-marking/flows.md`

---

## Summary

Evaluation Meta:

1. **Tracks per-question metadata** during evaluation
2. **Captures scoring details** (verdict, delta, running score)
3. **Stores answer information** (answer_item, answer_type)
4. **Identifies bonus questions** via bonus_type
5. **Provides formatted summary** for display/export

**Best For**: Detailed scoring breakdowns, debugging, student feedback
**Output Format**: JSON-serializable dict with nested question metadata
**Browser Migration**: Straightforward with TypeScript interfaces
