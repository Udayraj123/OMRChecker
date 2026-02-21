# Evaluation JSON Format

**Module**: modules/integration/evaluation-format/
**Created**: 2026-02-20
**Reference**: `src/schemas/evaluation_schema.py`

## Evaluation Structure

```json
{
  "source_type": "csv",
  "answer_key": "answers.csv",
  "marking_scheme": {
    "default": {
      "correct": 4,
      "incorrect": -1,
      "unmarked": 0
    }
  },
  "section_marking": {
    "Section_A": {
      "correct": 5,
      "incorrect": -2,
      "unmarked": 0
    }
  }
}
```

## Fields

**source_type**: Where to find answers (csv, json, inline)
**answer_key**: Path to answer key file
**marking_scheme**: Default scoring rules
**section_marking**: Section-specific scoring

## Browser Evaluation

```typescript
const EvaluationSchema = z.object({
  source_type: z.enum(['csv', 'json', 'inline']),
  answer_key: z.string().optional(),
  marking_scheme: z.object({
    default: z.object({
      correct: z.number(),
      incorrect: z.number(),
      unmarked: z.number()
    })
  }),
  section_marking: z.record(z.object({
    correct: z.number(),
    incorrect: z.number(),
    unmarked: z.number()
  })).optional()
});
```

## Scoring Logic

```javascript
function calculateScore(detectedAnswers, answerKey, markingScheme) {
  let totalScore = 0;

  for (const [question, detected] of Object.entries(detectedAnswers)) {
    const correct = answerKey[question];
    const scheme = markingScheme[getSectionForQuestion(question)] ||
                   markingScheme.default;

    if (!detected) {
      totalScore += scheme.unmarked;
    } else if (detected === correct) {
      totalScore += scheme.correct;
    } else {
      totalScore += scheme.incorrect;
    }
  }

  return totalScore;
}
```
