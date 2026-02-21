# Evaluation Config - Flows

**Module**: Domain / Evaluation
**Python Reference**: `src/processors/evaluation/`
**Last Updated**: 2026-02-21

---

## Overview

This document details the execution flows for evaluation config initialization, answer key loading, conditional set matching, and score calculation.

---

## Flow 1: Evaluation Config Initialization

**Code Reference**: `evaluation_config.py:11-78`

### Input
- `curr_dir` (Path): Directory containing evaluation.json
- `local_evaluation_path` (Path): Path to evaluation.json
- `template` (Template): Template instance
- `tuning_config` (TuningConfig): Global configuration

### Steps

```
1. Load evaluation.json
   └─> open_evaluation_with_defaults(local_evaluation_path)
       └─> Load JSON file
       └─> Deep merge with default evaluation config

2. Extract conditional_sets
   └─> Pop "conditional_sets" from JSON (if exists)
   └─> Store as list of [name, matcher] tuples

3. Validate conditional sets
   └─> Check for duplicate set names
   └─> Raise ConfigError if duplicates found

4. Create default evaluation config
   └─> EvaluationConfigForSet(
           DEFAULT_SET_NAME,
           curr_dir,
           default_evaluation_json,
           template,
           tuning_config
       )
   └─> Parse questions/answers from options
   └─> Parse marking schemes
   └─> Build question-to-answer matchers

5. For each conditional set:
   a. Extract name and evaluation JSON
   b. Validate questions/answers consistency
      └─> If answersInOrder provided, questionsInOrder must exist
      └─> If questionsInOrder provided, answersInOrder must exist
      └─> Raise ConfigError if inconsistent
   c. Merge with partial defaults
      └─> partial_default_evaluation_json = {
              "outputs_configuration": default_evaluation_json["outputs_configuration"]
          }
      └─> merged_evaluation_json = OVERRIDE_MERGER.merge(
              partial_default_evaluation_json,
              evaluation_json_for_set
          )
   d. Create EvaluationConfigForSet
      └─> Pass parent_evaluation_config=default_evaluation_config
      └─> Merge questions/answers with parent
      └─> Inherit parent marking schemes
   e. Add to set_mapping[set_name]
   f. Collect exclude_files (answer key images)

6. Store exclude_files for pipeline
```

### Output
- `EvaluationConfig` instance with default and conditional sets

### Edge Cases
1. **Duplicate set names**: ConfigError raised
2. **Missing questions or answers in conditional set**: ConfigError raised
3. **Inconsistent questions/answers**: ConfigError raised
4. **Missing answer key CSV**: Warning logged (unless image provided)

---

## Flow 2: EvaluationConfigForSet Initialization

**Code Reference**: `evaluation_config_for_set.py:34-127`

### Input
- `set_name` (str): Name of this set
- `curr_dir` (Path): Directory for relative paths
- `merged_evaluation_json` (dict): Merged JSON configuration
- `template` (Template): Template instance
- `tuning_config` (TuningConfig): Global configuration
- `parent_evaluation_config` (EvaluationConfigForSet | None): Parent config for merging

### Steps

```
1. Extract configuration sections
   └─> options = merged_evaluation_json["options"]
   └─> outputs_configuration = merged_evaluation_json["outputs_configuration"]
   └─> marking_schemes = merged_evaluation_json["marking_schemes"]
   └─> source_type = merged_evaluation_json["source_type"]

2. Parse outputs configuration
   └─> draw_answers_summary
   └─> draw_detected_bubble_texts
   └─> draw_question_verdicts (parse colors if enabled)
   └─> draw_score
   └─> should_explain_scoring
   └─> should_export_explanation_csv

3. Parse questions and answers (based on source_type)
   ├─> If source_type == "local":
   │   └─> parse_local_question_answers(options)
   │       └─> questions_in_order = parse_questions_in_order(options["questions_in_order"])
   │       └─> answers_in_order = options["answers_in_order"]
   │
   └─> If source_type in ["csv", "image_and_csv"]:
       └─> parse_csv_question_answers(curr_dir, options, tuning_config, template)
           └─> See Flow 3: CSV Answer Key Loading

4. Merge with parent (if conditional set)
   └─> merge_parsed_questions_and_schemes_with_parent(
           parent_evaluation_config,
           local_questions_in_order,
           local_answers_in_order
       )
       └─> See Flow 4: Parent-Child Merging

5. Validate questions
   └─> Check len(questions_in_order) == len(answers_in_order)
   └─> Raise FieldDefinitionError if mismatch

6. Parse marking schemes
   └─> set_parsed_marking_schemes(marking_schemes, parent_evaluation_config, template)
       └─> For each section:
           └─> Create SectionMarkingScheme
           └─> If DEFAULT: Store as default_marking_scheme
           └─> If custom: Add to section_marking_schemes and question_to_scheme
       └─> If parent exists: Inherit parent schemes for unmapped questions

7. Validate marking schemes
   └─> Check no overlapping questions across sections
   └─> Check all section questions have answers
   └─> Raise EvaluationError if missing answers

8. Parse answers and create matchers
   └─> parse_answers_and_map_questions()
       └─> For each question:
           └─> Get marking scheme
           └─> Create AnswerMatcher(answer_item, section_marking_scheme)
           └─> Store in question_to_answer_matcher

9. Validate answers
   └─> If filter_out_multimarked_files enabled:
       └─> Check answer key for multi-marked answers
       └─> Raise ConfigError if found

10. Validate format strings
    └─> Test answers_summary_format_string with schema_verdict_counts
    └─> Test score_format_string with score=0
    └─> Raise ConfigError if invalid

11. Reset evaluation state
    └─> Initialize schema_verdict_counts to 0
    └─> Prepare explanation_table (if enabled)
```

### Output
- `EvaluationConfigForSet` instance ready for scoring

---

## Flow 3: CSV Answer Key Loading

**Code Reference**: `evaluation_config_for_set.py:139-232`

### Input
- `curr_dir` (Path): Directory containing evaluation.json
- `options` (dict): Options section from evaluation JSON
- `tuning_config` (TuningConfig): Global configuration
- `template` (Template): Template instance

### Steps

```
1. Initialize variables
   └─> questions_in_order = None
   └─> csv_path = curr_dir / options["answer_key_csv_path"]
   └─> answer_key_image_path = options.get("answer_key_image_path", None)

2. Check if CSV exists
   └─> If csv_path.exists():
       └─> Go to Step 3: Load from CSV
   └─> Else if answer_key_image_path:
       └─> Go to Step 4: Generate from Image
   └─> Else:
       └─> Raise InputFileNotFoundError

3. Load from CSV
   └─> answer_key = pd.read_csv(
           csv_path,
           header=None,
           names=["question", "answer"],
           converters={
               "question": lambda q: q.strip(),
               "answer": parse_answer_column
           }
       )
   └─> questions_in_order = answer_key["question"].to_list()
   └─> answers_in_order = answer_key["answer"].to_list()
   └─> Return (questions_in_order, answers_in_order)

4. Generate from Image
   a. Validate image exists
      └─> image_path = curr_dir / answer_key_image_path
      └─> Raise ImageReadError if not exists
      └─> Add to exclude_files

   b. Read answer key image
      └─> gray_image, colored_image = ImageUtils.read_image_util(image_path, tuning_config)
      └─> Raise ImageReadError if failed

   c. Process through template
      └─> context = template.process_file(image_path, gray_image, colored_image)
      └─> concatenated_omr_response = context.omr_response

   d. Determine questions
      ├─> If "questions_in_order" in options:
      │   └─> questions_in_order = parse_questions_in_order(options["questions_in_order"])
      │   └─> Validate no empty answers
      │       └─> empty_answer_regex = rf"{re.escape(empty_value)}+" if empty_value != "" else r"^$"
      │       └─> empty_answered_questions = [q for q in questions_in_order if re.search(empty_answer_regex, response[q])]
      │       └─> If empty_answered_questions:
      │           └─> Log error
      │           └─> Raise EvaluationError
      │
      └─> Else:
          └─> Log warning: "questions_in_order not provided, using non-empty values"
          └─> questions_in_order = sorted([q for (q, ans) in response.items() if not re.search(empty_answer_regex, ans)])

   e. Extract answers
      └─> answers_in_order = [concatenated_omr_response[q] for q in questions_in_order]

   f. (TODO) Save CSV for future use

   g. Return (questions_in_order, answers_in_order)
```

### parse_answer_column Function

**Code Reference**: `evaluation_config_for_set.py:234-247`

```
Input: answer_column (str) - Raw CSV column value

Steps:
1. Remove all whitespaces
   └─> answer_column = answer_column.replace(" ", "")

2. Detect answer type
   ├─> If answer_column[0] == "[":
   │   └─> Multiple-correct-weighted or multiple-correct (JSON array)
   │   └─> parsed_answer = ast.literal_eval(answer_column)
   │
   ├─> Else if "," in answer_column:
   │   └─> Multiple-correct (comma-separated)
   │   └─> parsed_answer = answer_column.split(",")
   │
   └─> Else:
       └─> Single-correct (standard string)
       └─> parsed_answer = answer_column

Output: parsed_answer (AnswerItem)
```

**Examples**:
- `"A"` → `"A"`
- `"A,B"` → `["A", "B"]`
- `"[""A"",""B""]"` → `["A", "B"]`
- `"[[""A"",3],[""B"",1.5]]"` → `[["A", 3], ["B", 1.5]]`

---

## Flow 4: Parent-Child Question/Answer Merging

**Code Reference**: `evaluation_config_for_set.py:291-329`

### Input
- `parent_evaluation_config` (EvaluationConfigForSet | None): Parent config
- `local_questions_in_order` (list[str]): Child questions
- `local_answers_in_order` (list): Child answers

### Steps

```
1. If no parent (default set):
   └─> Return (local_questions_in_order, local_answers_in_order)

2. Create local question-to-answer mapping
   └─> local_question_to_answer_item = dict(zip(local_questions_in_order, local_answers_in_order))

3. Initialize merged lists
   └─> merged_questions_in_order = []
   └─> merged_answers_in_order = []

4. Iterate parent questions
   └─> For each (parent_question, parent_answer_item):
       a. Add parent_question to merged_questions_in_order
       b. If parent_question in local_question_to_answer_item:
          └─> Use child answer (override)
          └─> merged_answers_in_order.append(local_question_to_answer_item[parent_question])
       c. Else:
          └─> Use parent answer (inherit)
          └─> merged_answers_in_order.append(parent_answer_item)

5. Append new child questions
   └─> parent_questions_set = set(parent_questions_in_order)
   └─> For each (question, answer_item) in child:
       └─> If question not in parent_questions_set:
           └─> merged_questions_in_order.append(question)
           └─> merged_answers_in_order.append(answer_item)

6. Return (merged_questions_in_order, merged_answers_in_order)
```

### Example

**Parent**:
- questions: `["q1", "q2", "q3", "q4", "q5"]`
- answers: `["A", "B", "C", "D", "E"]`

**Child**:
- questions: `["q2", "q4", "q6"]`
- answers: `["X", "Y", "F"]`

**Merged**:
- questions: `["q1", "q2", "q3", "q4", "q5", "q6"]`
- answers: `["A", "X", "C", "Y", "E", "F"]`

---

## Flow 5: Conditional Set Matching

**Code Reference**: `evaluation_config.py:97-122`

### Input
- `concatenated_response` (dict): OMR response fields
- `file_path` (Path): Current file path

### Steps

```
1. Prepare formatting fields
   └─> formatting_fields = {
           **concatenated_response,
           "file_path": str(file_path),
           "file_name": str(file_path.name)
       }

2. Iterate conditional sets in order
   └─> For each (name, matcher):
       a. Extract format_string and match_regex
          └─> format_string = matcher["format_string"]
          └─> match_regex = matcher["match_regex"]

       b. Try to format and match
          └─> Try:
              ├─> formatted_string = format_string.format(**formatting_fields)
              ├─> if re.search(match_regex, formatted_string) is not None:
              │   └─> Return name (first match wins)
              └─> Except Exception:
                  └─> Return None (format failed)

3. If no match found:
   └─> Return None (use default set)
```

### Output
- Set name (str) or None

### Example

**Conditional Set Matcher**:
```json
{
  "formatString": "{q21}",
  "matchRegex": "A"
}
```

**Response**:
```json
{
  "q21": "A",
  "q1": "B",
  "q2": "C"
}
```

**Flow**:
1. `formatting_fields = {"q21": "A", "q1": "B", "q2": "C", "file_path": "...", "file_name": "..."}`
2. `formatted_string = "{q21}".format(**formatting_fields) = "A"`
3. `re.search("A", "A")` → Match!
4. Return `"Set A"`

---

## Flow 6: Score Calculation

**Code Reference**: `evaluation_meta.py:57-96`

### Input
- `concatenated_response` (dict): OMR response
- `evaluation_config_for_response` (EvaluationConfigForSet): Evaluation config

### Steps

```
1. Prepare and validate OMR response
   └─> evaluation_config_for_response.prepare_and_validate_omr_response(
           concatenated_response,
           allow_streak=True
       )
       └─> Check all questions exist in response
       └─> Check no missing questions
       └─> Raise EvaluationError if missing
       └─> Reset evaluation state (verdict counts, explanation table)

2. Initialize evaluation metadata
   └─> evaluation_meta = EvaluationMeta()
   └─> evaluation_meta.score = 0.0
   └─> evaluation_meta.questions_meta = {}

3. For each question in order:
   a. Get marked answer
      └─> marked_answer = concatenated_response[question]

   b. Match answer and get verdict
      └─> (delta, question_verdict, answer_matcher, question_schema_verdict) =
              evaluation_config_for_response.match_answer_for_question(
                  evaluation_meta.score,
                  question,
                  marked_answer
              )
          └─> See Flow 7: Answer Matching

   c. Get marking scheme and bonus type
      └─> marking_scheme = evaluation_config_for_response.get_marking_scheme_for_question(question)
      └─> bonus_type = marking_scheme.get_bonus_type()

   d. Update score
      └─> evaluation_meta.score += delta

   e. Create question metadata
      └─> question_meta = QuestionMeta(
              question,
              question_verdict,
              marked_answer,
              delta,
              evaluation_meta.score,
              answer_matcher,
              bonus_type,
              question_schema_verdict
          )

   f. Add to metadata
      └─> evaluation_meta.add_question_meta(question, question_meta)

4. Print explanation table (if enabled)
   └─> evaluation_config_for_response.conditionally_print_explanation()

5. Get formatted answers summary
   └─> formatted_answers_summary = evaluation_config_for_response.get_formatted_answers_summary()

6. Return score and metadata
   └─> return (evaluation_meta.score, evaluation_meta.to_dict(formatted_answers_summary))
```

### Output
- `score` (float): Final score
- `evaluation_meta` (dict): Evaluation metadata with per-question details

---

## Flow 7: Answer Matching (Single Question)

**Code Reference**: `evaluation_config_for_set.py:566-595`

### Input
- `current_score` (float): Score before this question
- `question` (str): Question field name
- `marked_answer` (str): Student's marked answer

### Steps

```
1. Get answer matcher for question
   └─> answer_matcher = question_to_answer_matcher[question]

2. Get verdict and delta
   └─> (question_verdict, delta, current_streak, updated_streak) =
           answer_matcher.get_verdict_marking(marked_answer, allow_streak=True)
       └─> See Flow 8: Verdict Determination

3. Get schema verdict
   └─> question_schema_verdict = AnswerMatcher.get_schema_verdict(
           answer_matcher.answer_type,
           question_verdict,
           delta
       )
       └─> Map verdict to schema verdict (correct/incorrect/unmarked)
       └─> Special case: Negative delta for weighted answers → "incorrect"

4. Update schema verdict counts
   └─> schema_verdict_counts[question_schema_verdict] += 1

5. Conditionally add explanation row
   └─> If should_explain_scoring:
       └─> Add row to explanation_table with:
           - Marking Scheme (if custom)
           - Question
           - Marked Answer
           - Correct Answer(s)
           - Verdict
           - Delta
           - Current Score + Delta
           - Set Mapping (if conditional sets)
           - Streak (if streak marking)

6. Return verdict data
   └─> return (delta, question_verdict, answer_matcher, question_schema_verdict)
```

---

## Flow 8: Verdict Determination (AnswerMatcher)

**Code Reference**: `answer_matcher.py:180-198`

### Input
- `marked_answer` (str): Student's answer
- `allow_streak` (bool): Whether to apply streak bonuses

### Steps

```
1. Determine question verdict (based on answer type)
   ├─> If AnswerType.STANDARD:
   │   └─> question_verdict = get_standard_verdict(marked_answer)
   │       ├─> If marked_answer == empty_value: Return Verdict.UNMARKED
   │       ├─> If marked_answer == allowed_answer: Return Verdict.ANSWER_MATCH
   │       └─> Else: Return Verdict.NO_ANSWER_MATCH
   │
   ├─> If AnswerType.MULTIPLE_CORRECT:
   │   └─> question_verdict = get_multiple_correct_verdict(marked_answer)
   │       ├─> If marked_answer == empty_value: Return Verdict.UNMARKED
   │       ├─> If marked_answer in allowed_answers: Return f"ANSWER_MATCH-{marked_answer}"
   │       └─> Else: Return Verdict.NO_ANSWER_MATCH
   │
   └─> If AnswerType.MULTIPLE_CORRECT_WEIGHTED:
       └─> question_verdict = get_multiple_correct_weighted_verdict(marked_answer)
           ├─> If marked_answer == empty_value: Return Verdict.UNMARKED
           ├─> If marked_answer in allowed_answers: Return f"ANSWER_MATCH-{marked_answer}"
           └─> Else: Return Verdict.NO_ANSWER_MATCH

2. Get delta and update streak
   └─> (delta, current_streak, updated_streak) =
           section_marking_scheme.get_delta_and_update_streak(
               marking,
               answer_type,
               question_verdict,
               allow_streak
           )
       └─> See Flow 9: Delta Calculation with Streak

3. Return verdict and delta
   └─> return (question_verdict, delta, current_streak, updated_streak)
```

---

## Flow 9: Delta Calculation with Streak

**Code Reference**: `section_marking_scheme.py:46-105`

### Input
- `answer_matcher_marking` (dict): Marking scores for all verdicts
- `answer_type` (AnswerType): Type of answer
- `question_verdict` (str): Verdict from answer matching
- `allow_streak` (bool): Whether to apply streak bonuses

### Steps

```
1. Get schema verdict
   └─> schema_verdict = AnswerMatcher.get_schema_verdict(answer_type, question_verdict, delta=0)

2. Calculate delta based on marking type
   ├─> If marking_type == VERDICT_LEVEL_STREAK:
   │   a. Get current streak for this schema verdict
   │      └─> current_streak = streaks[schema_verdict]
   │   b. Reset all streaks
   │      └─> streaks = {correct: 0, incorrect: 0, unmarked: 0}
   │   c. Increase only current verdict streak (if allowed)
   │      └─> If allow_streak and schema_verdict != UNMARKED:
   │          └─> streaks[schema_verdict] = current_streak + 1
   │   d. Get delta for current streak
   │      └─> delta = get_delta_for_verdict(answer_matcher_marking, question_verdict, current_streak)
   │          ├─> If marking[question_verdict] is list:
   │          │   └─> Return marking[question_verdict][current_streak]
   │          │       └─> If current_streak > len(list): Use last value
   │          └─> Else:
   │              └─> Return marking[question_verdict]
   │              └─> Warn if current_streak > 0
   │   e. Get updated streak
   │      └─> updated_streak = streaks[schema_verdict]
   │
   ├─> If marking_type == SECTION_LEVEL_STREAK:
   │   a. Get current section-level streak
   │      └─> current_streak = section_level_streak
   │   b. Get previous verdict
   │      └─> previous_verdict = previous_streak_verdict
   │   c. Reset streak state
   │      └─> section_level_streak = 0
   │      └─> previous_streak_verdict = None
   │   d. Increase streak if same verdict
   │      └─> If allow_streak and (previous_verdict is None or schema_verdict == previous_verdict):
   │          └─> section_level_streak = current_streak + 1
   │   e. Get delta for current streak
   │      └─> delta = get_delta_for_verdict(answer_matcher_marking, question_verdict, current_streak)
   │   f. Get updated streak
   │      └─> updated_streak = section_level_streak
   │
   └─> Else (DEFAULT marking):
       a. No streak bonuses
          └─> current_streak = 0
          └─> updated_streak = 0
       b. Get delta
          └─> delta = get_delta_for_verdict(answer_matcher_marking, question_verdict, 0)

3. Return delta and streak info
   └─> return (delta, current_streak, updated_streak)
```

### Streak Examples

**Verdict-Level Streak**:
```
Questions: q1, q2, q3, q4, q5
Answers:   correct, correct, correct, incorrect, correct

Streaks:
q1: correct (streak=0) → delta from marking[correct][0]
q2: correct (streak=1) → delta from marking[correct][1]
q3: correct (streak=2) → delta from marking[correct][2]
q4: incorrect (streak=0) → delta from marking[incorrect][0], reset correct streak
q5: correct (streak=0) → delta from marking[correct][0], reset incorrect streak
```

**Section-Level Streak**:
```
Questions: q1, q2, q3, q4, q5
Answers:   correct, correct, incorrect, correct, correct

Streaks:
q1: correct (streak=0) → delta from marking[correct][0]
q2: correct (streak=1, same as previous) → delta from marking[correct][1]
q3: incorrect (streak=0, different from previous) → delta from marking[incorrect][0]
q4: correct (streak=0, different from previous) → delta from marking[correct][0]
q5: correct (streak=1, same as previous) → delta from marking[correct][1]
```

---

## Flow 10: Explanation Table Export

**Code Reference**: `evaluation_config_for_set.py:649-658`

### Input
- `file_path` (Path): Original OMR image path

### Steps

```
1. Check if export enabled
   └─> If not should_export_explanation_csv:
       └─> Return False (early exit)

2. Get explanation table
   └─> explanation_table = get_explanation_table()
       └─> Rich Table object with rows

3. Convert table to DataFrame
   └─> explanation_df = table_to_df(explanation_table)
       └─> Extract columns and rows from Rich Table
       └─> Create pandas DataFrame

4. Export as CSV
   └─> csv_path = get_evaluations_dir() / f"{file_path.stem}.csv"
   └─> explanation_df.to_csv(
           csv_path,
           quoting=QUOTE_NONNUMERIC,
           index=False
       )

5. Return True (export completed)
```

### CSV Output Example

```csv
"Marking Scheme","Question","Marked","Answer(s)","Verdict","Delta","Score","Set Mapping"
"DEFAULT","q1","A","A","Correct (ANSWER_MATCH)","3.0","3.0","Set A"
"DEFAULT","q2","C","B","Incorrect (NO_ANSWER_MATCH)","-1.0","2.0","Set A"
"SECTION_A","q3","","C","Unmarked (UNMARKED)","0.0","2.0","Set A"
```

---

## Browser Migration Notes

### Async CSV Loading

```typescript
async function loadEvaluationConfig(
    evaluationPath: string,
    template: Template,
    tuningConfig: TuningConfig
): Promise<EvaluationConfig> {
    const response = await fetch(evaluationPath);
    const json = await response.json();

    // Validate with Zod
    const validated = EvaluationSchema.parse(json);

    // Create config
    const config = new EvaluationConfig(evaluationPath, json, template, tuningConfig);
    await config.initialize();
    return config;
}
```

### Regex Matching (Native)

```typescript
function matchConditionalSet(
    response: Record<string, string>,
    filePath: string,
    conditionalSets: Array<[string, Matcher]>
): string | null {
    const fields = { ...response, file_path: filePath, file_name: filePath.split('/').pop() };

    for (const [name, { formatString, matchRegex }] of conditionalSets) {
        try {
            const formatted = formatString.replace(/\{(\w+)\}/g, (_, key) => fields[key] || '');
            if (new RegExp(matchRegex).test(formatted)) {
                return name;
            }
        } catch {
            return null;
        }
    }

    return null;
}
```

### Score Calculation (Pure JS)

```typescript
function evaluateResponse(
    response: Record<string, string>,
    config: EvaluationConfigForSet
): { score: number; meta: EvaluationMeta } {
    config.prepareAndValidateResponse(response);
    const meta = new EvaluationMeta();

    for (const question of config.questionsInOrder) {
        const markedAnswer = response[question];
        const { delta, verdict, matcher, schemaVerdict } = config.matchAnswerForQuestion(
            meta.score,
            question,
            markedAnswer
        );

        const scheme = config.getMarkingSchemeForQuestion(question);
        const bonusType = scheme.getBonusType();

        meta.score += delta;
        meta.addQuestionMeta(question, {
            verdict,
            markedAnswer,
            delta,
            currentScore: meta.score,
            answerItem: matcher.answerItem,
            answerType: matcher.answerType,
            bonusType,
            schemaVerdict
        });
    }

    return { score: meta.score, meta };
}
```

---

## Summary

**Key Flows**:
1. Initialization: Load JSON → Parse questions/answers → Build matchers
2. CSV Loading: Load from CSV or generate from image
3. Conditional Set Matching: Regex-based routing
4. Score Calculation: Iterate questions → Match → Update score → Metadata
5. Answer Matching: Determine verdict → Get delta with streak
6. Explanation Export: Table → CSV download

**Browser Considerations**:
- Use native `RegExp` for conditional set matching
- Use `fetch` for loading evaluation.json and CSV files
- Use papaparse for CSV parsing
- Use Blob/download for explanation export
- Maintain same scoring logic (no changes needed)
- Consider React/Vue table for explanation display
