# Evaluation Config For Set Flows

**Module**: Domain - Evaluation - Config For Set
**Python Reference**: `src/processors/evaluation/evaluation_config_for_set.py`
**Last Updated**: 2026-02-21

---

## Overview

EvaluationConfigForSet manages the evaluation configuration for a specific question set (default or conditional). It handles answer key parsing from multiple sources (local, CSV, or image), merges configurations with parent sets, manages marking schemes per section, and validates all evaluation components.

**Use Cases**:
- Default answer key configuration for all OMR sheets
- Conditional set-specific answer keys (e.g., Set A, Set B, Set C based on student's marked set field)
- Per-section custom marking schemes (bonus sections, weighted answers, streak bonuses)
- Answer key generation from scanned answer key images

**Parent Relationship**: Each conditional set can inherit from and override the default set's configuration.

---

## Initialization Flow

### Main Initialization

```
START: EvaluationConfigForSet.__init__(set_name, curr_dir, merged_evaluation_json, template, tuning_config, parent_evaluation_config)
│
├─► STEP 1: Set Basic Properties
│   │
│   self.set_name = set_name  # e.g., "DEFAULT_SET", "Set A", "Set B"
│   │
│   Extract from merged_evaluation_json:
│   ├─ options                    # Answer key source configuration
│   ├─ outputs_configuration      # Visual output settings
│   ├─ marking_schemes           # Scoring rules
│   └─ source_type               # "local", "csv", or "image_and_csv"
│   │
│   self.has_conditional_sets = (parent_evaluation_config is not None)
│   │
│   self.has_custom_marking = False  # Will be set based on marking schemes
│   self.has_streak_marking = False  # Will be set if streak bonuses used
│   self.exclude_files = []          # Files to skip during processing
│
├─► STEP 2: Parse Output Configuration
│   │
│   Extract visualization settings:
│   ├─ draw_answers_summary          # Show verdict counts (Correct: X, Incorrect: Y, Unmarked: Z)
│   ├─ draw_detected_bubble_texts    # Show detected bubble values on image
│   ├─ draw_question_verdicts        # Color bubbles by verdict
│   ├─ draw_score                    # Show total score
│   ├─ should_explain_scoring        # Generate explanation table
│   └─ should_export_explanation_csv # Export scoring breakdown to CSV
│   │
│   If draw_question_verdicts enabled:
│   │   └─► parse_draw_question_verdicts()
│   │       ├─ Parse verdict colors (correct, incorrect, neutral, bonus)
│   │       ├─ Parse verdict symbol colors (positive, negative, neutral, bonus)
│   │       └─ Parse answer group drawing settings
│
├─► STEP 3: Parse Question-Answer Pairs (Based on source_type)
│   │
│   ┌─────────────────────────────────────────────────────────┐
│   │ BRANCH: source_type == "local"                         │
│   │                                                         │
│   │ parse_local_question_answers(options)                  │
│   │ ├─ questions_in_order = ["q1", "q2", ..., "q20"]      │
│   │ │   (parsed from field strings like "q1..20")         │
│   │ │                                                       │
│   │ └─ answers_in_order = ["A", "B", "C", ..., "D"]       │
│   │     (direct array from evaluation.json)                │
│   └─────────────────────────────────────────────────────────┘
│   │
│   ┌─────────────────────────────────────────────────────────┐
│   │ BRANCH: source_type in {"csv", "image_and_csv"}       │
│   │                                                         │
│   │ parse_csv_question_answers(curr_dir, options,          │
│   │                             tuning_config, template)   │
│   │ │                                                       │
│   │ ├─► SUB-FLOW: CSV Path Resolution                     │
│   │ │   csv_path = curr_dir / options["answer_key_csv_path"]
│   │ │   answer_key_image_path = options.get("answer_key_image_path")
│   │ │                                                       │
│   │ ├─► CASE 1: CSV File Exists                          │
│   │ │   │                                                   │
│   │ │   Read CSV with pandas:                             │
│   │ │   ├─ Column 1: question (e.g., "q1", "q2")          │
│   │ │   └─ Column 2: answer (parsed via parse_answer_column)
│   │ │       │                                               │
│   │ │       parse_answer_column() handles:                │
│   │ │       ├─ "[['A', 2], ['B', 1]]" → weighted answers  │
│   │ │       ├─ "A,B,C" → multiple correct ["A","B","C"]   │
│   │ │       └─ "A" → standard answer "A"                  │
│   │ │   │                                                   │
│   │ │   questions_in_order = csv["question"].to_list()    │
│   │ │   answers_in_order = csv["answer"].to_list()        │
│   │ │                                                       │
│   │ ├─► CASE 2: CSV Missing, Image Path Provided         │
│   │ │   │                                                   │
│   │ │   Read answer key from scanned image:               │
│   │ │   │                                                   │
│   │ │   ├─ image_path = curr_dir / answer_key_image_path  │
│   │ │   │                                                   │
│   │ │   ├─ gray_image, colored_image = ImageUtils.read_image_util()
│   │ │   │                                                   │
│   │ │   ├─ context = template.process_file()              │
│   │ │   │   # Run full pipeline on answer key image       │
│   │ │   │   # (preprocessing, alignment, detection)       │
│   │ │   │                                                   │
│   │ │   ├─ concatenated_omr_response = context.omr_response
│   │ │   │   # e.g., {"q1": "A", "q2": "B", "q3": "C"}     │
│   │ │   │                                                   │
│   │ │   ├─ Validate no empty answers:                     │
│   │ │   │   empty_regex = "^$" or f"{empty_value}+"       │
│   │ │   │   Check all questions_in_order don't match empty │
│   │ │   │                                                   │
│   │ │   ├─ If questions_in_order provided:                │
│   │ │   │   questions_in_order from options               │
│   │ │   │   answers_in_order from omr_response            │
│   │ │   │   ERROR if any answer is empty                  │
│   │ │   │                                                   │
│   │ │   └─ Else (no questions_in_order):                  │
│   │ │       questions_in_order = non-empty q-prefixed fields
│   │ │       answers_in_order from omr_response            │
│   │ │   │                                                   │
│   │ │   self.exclude_files.append(image_path)             │
│   │ │   # Prevent processing answer key as student sheet  │
│   │ │                                                       │
│   │ └─► CASE 3: Neither CSV nor Image                    │
│   │     │                                                   │
│   │     RAISE InputFileNotFoundError                       │
│   └─────────────────────────────────────────────────────────┘
│   │
│   local_questions_in_order = [parsed questions]
│   local_answers_in_order = [parsed answers]
│
├─► STEP 4: Merge with Parent Config (if conditional set)
│   │
│   (questions_in_order, answers_in_order) =
│       merge_parsed_questions_and_schemes_with_parent(
│           parent_evaluation_config,
│           local_questions_in_order,
│           local_answers_in_order
│       )
│   │
│   If parent_evaluation_config is None:
│   │   # This is the default set
│   │   return local questions/answers as-is
│   │
│   Else (conditional set inheriting from parent):
│   │   │
│   │   Create merged lists:
│   │   │
│   │   ├─► For each parent question:
│   │   │   │
│   │   │   ├─ Append parent question to merged_questions
│   │   │   │
│   │   │   └─ If child overrides this question:
│   │   │       │   Use child's answer
│   │   │       Else:
│   │   │           Use parent's answer
│   │   │
│   │   └─► For each child-only question (not in parent):
│   │       Append to merged lists at end
│   │   │
│   │   Example:
│   │   Parent: q1-q20 with answers [A, B, C, ..., T]
│   │   Child:  q1-q3 override + q21 new
│   │   Result: q1-q20 (first 3 from child, rest from parent) + q21
│
├─► STEP 5: Validate Questions
│   │
│   validate_questions()
│   │
│   ├─ Check len(questions_in_order) == len(answers_in_order)
│   │
│   └─ If mismatch:
│       RAISE FieldDefinitionError
│
├─► STEP 6: Parse and Set Marking Schemes
│   │
│   set_parsed_marking_schemes(marking_schemes, parent_evaluation_config, template)
│   │
│   Initialize:
│   ├─ self.section_marking_schemes = {}
│   ├─ self.question_to_scheme = {}       # Maps question → scheme
│   └─ self.default_marking_scheme        # Required DEFAULT scheme
│   │
│   For each section_key, section_scheme in marking_schemes:
│   │   │
│   │   section_marking_scheme = SectionMarkingScheme(
│   │       section_key, section_scheme, set_name, empty_value
│   │   )
│   │   │
│   │   If section_key == "DEFAULT":
│   │   │   self.default_marking_scheme = section_marking_scheme
│   │   │
│   │   Else (custom section):
│   │   │   self.section_marking_schemes[section_key] = section_marking_scheme
│   │   │   │
│   │   │   For each question in section.questions:
│   │   │   │   self.question_to_scheme[q] = section_marking_scheme
│   │   │   │
│   │   │   self.has_custom_marking = True
│   │   │   │
│   │   │   If marking_type == "verdict_level_streak":
│   │   │       self.has_streak_marking = True
│   │
│   If parent_evaluation_config exists:
│   │   └─► update_marking_schemes_from_parent()
│   │       │
│   │       For each parent section scheme:
│   │       │   │
│   │       │   questions_subset = [questions not already mapped locally]
│   │       │   │
│   │       │   If len(questions_subset) > 0:
│   │       │   │   │
│   │       │   │   Create deepcopy with updated questions
│   │       │   │   section_key = f"parent-{parent_section_key}"
│   │       │   │   │
│   │       │   │   self.section_marking_schemes[section_key] = subset_scheme
│   │       │   │   │
│   │       │   │   Map remaining questions to this scheme
│   │       │   │   self.has_custom_marking = True
│
├─► STEP 7: Validate Marking Schemes
│   │
│   validate_marking_schemes()
│   │
│   ├─ Check no overlapping questions across sections
│   │   If overlap found: RAISE FieldDefinitionError
│   │
│   └─ Check all scheme questions exist in answer key
│       If missing: RAISE EvaluationError
│
├─► STEP 8: Parse Answers and Map Questions
│   │
│   question_to_answer_matcher = parse_answers_and_map_questions()
│   │
│   For each (question, answer_item) in zip(questions, answers):
│   │   │
│   │   section_marking_scheme = get_marking_scheme_for_question(question)
│   │   # Uses question_to_scheme map or default scheme
│   │   │
│   │   answer_matcher = AnswerMatcher(answer_item, section_marking_scheme)
│   │   │   ├─ Determines answer_type:
│   │   │   │   ├─ STANDARD: "A"
│   │   │   │   ├─ MULTIPLE_CORRECT: ["A", "B", "AB"]
│   │   │   │   └─ MULTIPLE_CORRECT_WEIGHTED: [["A", 2], ["B", 1]]
│   │   │   │
│   │   │   └─ Sets local marking defaults
│   │   │
│   │   question_to_answer_matcher[question] = answer_matcher
│   │   │
│   │   If answer_type == MULTIPLE_CORRECT_WEIGHTED:
│   │       self.has_custom_marking = True
│
├─► STEP 9: Validate Answers (Multi-mark Check)
│   │
│   validate_answers(tuning_config)
│   │
│   If config.filter_out_multimarked_files is True:
│   │   │
│   │   Check answer key for multi-marked answers:
│   │   ├─ STANDARD: len(answer) > 1 (e.g., "AB")
│   │   ├─ MULTIPLE_CORRECT: any len(answer) > 1 (e.g., ["AB", "C"])
│   │   └─ MULTIPLE_CORRECT_WEIGHTED: any len(answer) > 1
│   │   │
│   │   If multi-marked found:
│   │       RAISE ConfigError (incompatible with filter setting)
│
├─► STEP 10: Reset Evaluation State
│   │
│   reset_evaluation()
│   │   ├─ explanation_table = None
│   │   ├─ Reset all section streaks
│   │   ├─ schema_verdict_counts = {correct: 0, incorrect: 0, unmarked: 0}
│   │   └─ prepare_explanation_table() if should_explain_scoring
│
└─► STEP 11: Validate Format Strings
    │
    validate_format_strings()
    │
    ├─ Test answers_summary_format_string:
    │   answers_summary_format_string.format(**schema_verdict_counts)
    │   # e.g., "Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}"
    │   If invalid: RAISE ConfigError
    │
    └─ Test score_format_string:
        score_format_string.format(score=0)
        # e.g., "Score: {score}"
        If invalid: RAISE ConfigError

END: EvaluationConfigForSet instance ready for use
```

---

## Answer Key Parsing Flows

### Local Source Type

```
parse_local_question_answers(options)
│
├─ questions_in_order = parse_questions_in_order(options["questions_in_order"])
│   │
│   parse_fields("questions_in_order", ["q1..20", "q25..30"])
│   │   └─► Returns: ["q1", "q2", ..., "q20", "q25", "q26", ..., "q30"]
│
└─ answers_in_order = options["answers_in_order"]
    # Direct array: ["A", "B", "C", ...]

Return: (questions_in_order, answers_in_order)
```

### CSV Source Type

```
parse_csv_question_answers(curr_dir, options, tuning_config, template)
│
├─ csv_path = curr_dir / options["answer_key_csv_path"]
│   # e.g., "answer-key.csv"
│
└─ answer_key_image_path = options.get("answer_key_image_path", None)
    # Optional fallback image

CSV Format:
┌──────────┬────────────────────────┐
│ question │ answer                 │
├──────────┼────────────────────────┤
│ q1       │ A                      │
│ q2       │ B                      │
│ q3       │ A,B                    │  ← Multiple correct
│ q4       │ [['A', 2], ['B', 1]]   │  ← Weighted
│ q5       │ C                      │
└──────────┴────────────────────────┘

Read and parse:
│
├─ answer_key = pd.read_csv(
│       csv_path,
│       header=None,
│       names=["question", "answer"],
│       converters={
│           "question": lambda q: q.strip(),
│           "answer": parse_answer_column
│       }
│   )
│
├─ parse_answer_column(answer_str):
│   │
│   ├─ Remove whitespace
│   │
│   ├─ If starts with "[":
│   │   return ast.literal_eval(answer_str)
│   │   # e.g., "[['A', 2], ['B', 1]]" → [["A", 2], ["B", 1]]
│   │
│   ├─ Elif contains ",":
│   │   return answer_str.split(",")
│   │   # e.g., "A,B,C" → ["A", "B", "C"]
│   │
│   └─ Else:
│       return answer_str
│       # e.g., "A" → "A"
│
├─ questions_in_order = answer_key["question"].to_list()
│
└─ answers_in_order = answer_key["answer"].to_list()

Return: (questions_in_order, answers_in_order)
```

### Image Source Type (Answer Key Generation)

```
parse_csv_question_answers() with answer_key_image_path
│
├─ CSV not found, but image provided
│
├─ image_path = curr_dir / answer_key_image_path
│
├─ gray_image, colored_image = ImageUtils.read_image_util(image_path, tuning_config)
│
├─ context = template.process_file(image_path, gray_image, colored_image)
│   # Run full OMR pipeline on answer key image
│   │
│   ├─ Preprocessing (crop, rotate, enhance)
│   ├─ Alignment (SIFT/template matching)
│   └─ Detection (bubble threshold/OCR/barcode)
│   │
│   Returns: ProcessingContext with omr_response
│
├─ concatenated_omr_response = context.omr_response
│   # e.g., {"q1": "A", "q2": "B", "q3": "", "q4": "C", ...}
│
├─ Determine empty answer pattern:
│   empty_value = template.global_empty_val  # e.g., "" or "X"
│   │
│   If empty_value == "":
│   │   empty_answer_regex = r"^$"
│   Else:
│       empty_answer_regex = rf"{re.escape(empty_value)}+"
│
├─► CASE 1: questions_in_order provided in options
│   │
│   questions_in_order = parse_questions_in_order(options["questions_in_order"])
│   │
│   Validate no empty answers:
│   │
│   empty_answered_questions = [
│       q for q in questions_in_order
│       if re.search(empty_answer_regex, omr_response[q])
│   ]
│   │
│   If len(empty_answered_questions) > 0:
│   │   logger.error(f"Found empty answers for: {empty_answered_questions}")
│   │   RAISE EvaluationError
│   │       "Found empty answers in file. Check template in --setLayout mode."
│   │
│   answers_in_order = [omr_response[q] for q in questions_in_order]
│
└─► CASE 2: questions_in_order NOT provided
    │
    logger.warning("questions_in_order not provided, using non-empty values")
    │
    questions_in_order = sorted([
        question
        for (question, answer) in omr_response.items()
        if not re.search(empty_answer_regex, answer)
        and question.startswith("q")
    ])
    │
    answers_in_order = [omr_response[q] for q in questions_in_order]

│
└─ self.exclude_files.append(image_path)
    # Don't process answer key as a student sheet

Return: (questions_in_order, answers_in_order)
```

---

## Parent-Child Set Merging Flow

```
merge_parsed_questions_and_schemes_with_parent(parent_config, local_questions, local_answers)
│
└─► If parent_evaluation_config is None:
    │   # This is the default set (no parent)
    │   return local_questions, local_answers
    │
    Else:
        │
        ├─ parent_questions = parent_config.questions_in_order
        │  parent_answers = parent_config.answers_in_order
        │  # e.g., ["q1", ..., "q20"] with 20 answers
        │
        ├─ local_question_to_answer_item = dict(zip(local_questions, local_answers))
        │  # e.g., {"q1": "B", "q2": "D", "q3": "A"}
        │
        ├─ Initialize merged lists
        │
        ├─► Phase 1: Merge parent questions with child overrides
        │   │
        │   For each (parent_question, parent_answer) in parent:
        │   │   │
        │   │   merged_questions.append(parent_question)
        │   │   │
        │   │   If parent_question in local_question_to_answer_item:
        │   │   │   # Child overrides this question's answer
        │   │   │   merged_answers.append(local_question_to_answer_item[parent_question])
        │   │   Else:
        │   │       # Use parent's answer
        │   │       merged_answers.append(parent_answer)
        │
        └─► Phase 2: Add new child-only questions
            │
            parent_questions_set = set(parent_questions)
            │
            For each (question, answer) in local:
            │   │
            │   If question not in parent_questions_set:
            │       # New question only in child set
            │       merged_questions.append(question)
            │       merged_answers.append(answer)

Example:
┌─────────────────────────────────────────────────────────────┐
│ Parent (DEFAULT_SET):                                       │
│   questions: ["q1", "q2", "q3", "q4", "q5"]                │
│   answers:   ["A",  "B",  "C",  "D",  "E"]                 │
│                                                             │
│ Child (Set A):                                             │
│   questions: ["q1", "q3", "q6"]                            │
│   answers:   ["B",  "D",  "F"]                             │
│                                                             │
│ Merged Result:                                             │
│   questions: ["q1", "q2", "q3", "q4", "q5", "q6"]          │
│   answers:   ["B",  "B",  "D",  "D",  "E",  "F"]           │
│                ↑          ↑                    ↑            │
│              override   override            new            │
└─────────────────────────────────────────────────────────────┘

Return: (merged_questions_in_order, merged_answers_in_order)
```

---

## Marking Scheme Setup Flow

```
set_parsed_marking_schemes(marking_schemes, parent_evaluation_config, template)
│
├─ Initialize
│   self.section_marking_schemes = {}
│   self.question_to_scheme = {}
│   self.default_marking_scheme = None
│
├─► Parse Local Marking Schemes
│   │
│   For each (section_key, section_scheme) in marking_schemes:
│   │   │
│   │   section_marking_scheme = SectionMarkingScheme(
│   │       section_key,
│   │       section_scheme,
│   │       self.set_name,
│   │       template.global_empty_val
│   │   )
│   │
│   │   If section_key == "DEFAULT":
│   │   │   │
│   │   │   self.default_marking_scheme = section_marking_scheme
│   │   │   # Every config MUST have a DEFAULT scheme
│   │   │
│   │   Else:
│   │       │
│   │       # Custom section scheme
│   │       self.section_marking_schemes[section_key] = section_marking_scheme
│   │       │
│   │       # Map questions to this scheme
│   │       For q in section_marking_scheme.questions:
│   │           self.question_to_scheme[q] = section_marking_scheme
│   │       │
│   │       self.has_custom_marking = True
│   │       │
│   │       If section_marking_scheme.marking_type == "verdict_level_streak":
│   │           self.has_streak_marking = True
│
└─► If parent_evaluation_config exists:
    │
    update_marking_schemes_from_parent(parent_evaluation_config)
    │
    parent_marking_schemes = parent_config.section_marking_schemes
    │
    For each (parent_section_key, parent_scheme) in parent_marking_schemes:
    │   │
    │   If parent_section_key == "DEFAULT":
    │   │   continue  # Skip (each set has its own default)
    │   │
    │   # Find questions from parent scheme not yet mapped locally
    │   questions_subset = [
    │       q for q in parent_scheme.questions
    │       if q not in self.question_to_scheme
    │   ]
    │   │
    │   If len(questions_subset) == 0:
    │   │   continue  # All questions already mapped locally
    │   │
    │   # Create inherited section with remaining questions
    │   section_key = f"parent-{parent_section_key}"
    │   │
    │   subset_marking_scheme = parent_scheme.deepcopy_with_questions(questions_subset)
    │   │
    │   self.section_marking_schemes[section_key] = subset_marking_scheme
    │   │
    │   For q in questions_subset:
    │       self.question_to_scheme[q] = subset_marking_scheme
    │   │
    │   self.has_custom_marking = True
    │   │
    │   If subset_marking_scheme.marking_type == "verdict_level_streak":
    │       self.has_streak_marking = True

Example:
┌─────────────────────────────────────────────────────────────┐
│ Parent (DEFAULT_SET) schemes:                               │
│   - DEFAULT: {correct: 3, incorrect: 0, unmarked: 0}        │
│   - BONUS_SECTION: {correct: 3, incorrect: 3, unmarked: 3}  │
│     questions: ["q11", "q12", "q13"]                        │
│                                                             │
│ Child (Set A) schemes:                                     │
│   - DEFAULT: {correct: 4, incorrect: 0, unmarked: 0}        │
│   - BONUS_SECTION: {correct: 4, incorrect: 4, unmarked: 4}  │
│     questions: ["q11", "q12"]  ← Only overrides q11, q12   │
│                                                             │
│ Result in Child:                                           │
│   - DEFAULT: 4/0/0 (child's default)                       │
│   - BONUS_SECTION: 4/4/4 for q11, q12 (child override)     │
│   - parent-BONUS_SECTION: 3/3/3 for q13 (inherited)        │
│     questions: ["q13"]                                     │
└─────────────────────────────────────────────────────────────┘

END: Marking schemes fully configured
```

---

## Runtime Evaluation Flow

### Prepare and Validate OMR Response

```
prepare_and_validate_omr_response(concatenated_omr_response, allow_streak)
│
├─ self.allow_streak = allow_streak  # Enable streak bonuses if True
│
├─ reset_evaluation()
│   ├─ Reset all section streaks to 0
│   ├─ schema_verdict_counts = {correct: 0, incorrect: 0, unmarked: 0}
│   └─ Create new explanation_table if should_explain_scoring
│
├─ Validate OMR response has all required questions:
│   │
│   omr_response_keys = set(concatenated_omr_response.keys())
│   all_questions = set(self.questions_in_order)
│   │
│   missing_questions = all_questions - omr_response_keys
│   │
│   If len(missing_questions) > 0:
│   │   logger.critical(f"Missing OMR response for: {missing_questions}")
│   │   RAISE EvaluationError
│
└─ Warn about unused q-prefixed fields:
    │
    prefixed_omr_questions = {k for k in omr_response if k.startswith("q")}
    missing_prefixed = prefixed_omr_questions - all_questions
    │
    If len(missing_prefixed) > 0:
        logger.warning(f"No answer given for: {missing_prefixed}")

Ready for question-by-question matching
```

### Match Answer For Question

```
match_answer_for_question(current_score, question, marked_answer)
│
├─ answer_matcher = self.question_to_answer_matcher[question]
│   # AnswerMatcher instance for this question
│
├─ Get verdict and delta:
│   │
│   (question_verdict, delta, current_streak, updated_streak) =
│       answer_matcher.get_verdict_marking(marked_answer, self.allow_streak)
│   │
│   question_verdict examples:
│   ├─ "answer-match"         (standard correct)
│   ├─ "answer-match-A"       (multiple correct, matched "A")
│   ├─ "no-answer-match"      (incorrect)
│   └─ "unmarked"             (empty)
│   │
│   delta: Score change (e.g., +3, -1, 0)
│   current_streak: Streak before this question
│   updated_streak: Streak after this question
│
├─ Map verdict to schema verdict:
│   │
│   question_schema_verdict = AnswerMatcher.get_schema_verdict(
│       answer_matcher.answer_type,
│       question_verdict,
│       delta
│   )
│   │
│   Schema verdicts: "correct", "incorrect", "unmarked"
│
├─ Update verdict counts:
│   self.schema_verdict_counts[question_schema_verdict] += 1
│
├─► If should_explain_scoring:
│   │
│   conditionally_add_explanation(
│       answer_matcher, delta, marked_answer,
│       question_schema_verdict, question_verdict,
│       question, current_score, current_streak, updated_streak
│   )
│   │
│   Build explanation table row:
│   ├─ Marking Scheme (if has_custom_marking)
│   ├─ Question (e.g., "q1")
│   ├─ Marked (e.g., "A")
│   ├─ Answer(s) (e.g., "B" or "['A', 'B']")
│   ├─ Verdict (e.g., "Correct (answer-match)")
│   ├─ Delta (e.g., "+3.0")
│   ├─ Score (e.g., "3.0")
│   ├─ Set Mapping (if has_conditional_sets)
│   └─ Streak (if has_streak_marking, e.g., "0 -> 1")
│   │
│   explanation_table.add_row(*row)
│
└─ Return: (delta, question_verdict, answer_matcher, question_schema_verdict)

Next score = current_score + delta
```

### Explanation Export Flow

```
conditionally_export_explanation_csv(file_path)
│
└─► If should_export_explanation_csv:
    │
    ├─ explanation_table = self.get_explanation_table()
    │   # Rich Table with all question-by-question scoring details
    │
    ├─ explanation_df = table_to_df(explanation_table)
    │   # Convert Rich Table to pandas DataFrame
    │
    ├─ csv_path = evaluations_dir / f"{file_path.stem}.csv"
    │   # e.g., "IMG_001.csv"
    │
    └─ explanation_df.to_csv(
            csv_path,
            quoting=QUOTE_NONNUMERIC,
            index=False
        )
        │
        CSV Output Example:
        ┌─────────┬────────┬──────────┬──────────┬───────┬───────┐
        │ Question│ Marked │ Answer(s)│ Verdict  │ Delta │ Score │
        ├─────────┼────────┼──────────┼──────────┼───────┼───────┤
        │ q1      │ A      │ B        │ Incorrect│ -1.0  │ -1.0  │
        │ q2      │ B      │ B        │ Correct  │ 3.0   │ 2.0   │
        │ q3      │        │ C        │ Unmarked │ 0.0   │ 2.0   │
        └─────────┴────────┴──────────┴──────────┴───────┴───────┘

Return: True if exported, False otherwise
```

---

## Visualization Configuration

### Draw Question Verdicts Parsing

```
parse_draw_question_verdicts()
│
├─ Extract configuration:
│   ├─ verdict_colors         # Colors for bubble backgrounds
│   ├─ verdict_symbol_colors  # Colors for +/-/o symbols
│   └─ draw_answer_groups     # Settings for answer group visualization
│
├─ Parse verdict colors (RGB → BGR for OpenCV):
│   self.verdict_colors = {
│       "correct": MathUtils.to_bgr(verdict_colors["correct"]),
│       "neutral": MathUtils.to_bgr(verdict_colors["neutral"] ?? verdict_colors["incorrect"]),
│       "incorrect": MathUtils.to_bgr(verdict_colors["incorrect"]),
│       "bonus": MathUtils.to_bgr(verdict_colors["bonus"])
│   }
│   │
│   Example:
│   ├─ correct: [0, 255, 0] (green) → [0, 255, 0] BGR
│   ├─ incorrect: [255, 0, 0] (red) → [0, 0, 255] BGR
│   └─ bonus: [255, 255, 0] (yellow) → [0, 255, 255] BGR
│
├─ Parse symbol colors:
│   self.verdict_symbol_colors = {
│       "positive": MathUtils.to_bgr(verdict_symbol_colors["positive"]),
│       "neutral": MathUtils.to_bgr(verdict_symbol_colors["neutral"]),
│       "negative": MathUtils.to_bgr(verdict_symbol_colors["negative"]),
│       "bonus": MathUtils.to_bgr(verdict_symbol_colors["bonus"])
│   }
│
└─ Parse answer group drawing:
    self.draw_answer_groups = {
        ...draw_answer_groups,
        "color_sequence": [
            MathUtils.to_bgr(color)
            for color in draw_answer_groups["color_sequence"]
        ]
    }
    │
    color_sequence: Used to color-code answer groups
    e.g., ["A", "B", "AB"] → [green, blue, purple]

Stored for use during get_evaluation_meta_for_question()
```

### Get Evaluation Meta For Question

```
get_evaluation_meta_for_question(question_meta, is_field_marked, image_type)
│
├─ Extract from question_meta:
│   ├─ bonus_type           # None, "BONUS_FOR_ALL", "BONUS_ON_ATTEMPT"
│   ├─ question_verdict     # e.g., "answer-match", "no-answer-match"
│   ├─ question_schema_verdict  # "correct", "incorrect", "unmarked"
│   └─ delta                # Score change
│
├─ Define symbols:
│   ├─ symbol_positive = "+"
│   ├─ symbol_negative = "-"
│   ├─ symbol_neutral = "o"
│   ├─ symbol_bonus = "*"
│   └─ symbol_unmarked = ""
│
├─► CASE: is_field_marked (bubble was filled)
│   │
│   ├─ Determine symbol based on delta:
│   │   ├─ If delta > 0: symbol = "+"
│   │   ├─ If delta < 0: symbol = "-"
│   │   └─ Else:         symbol = "o"
│   │
│   └─ Determine colors:
│       │
│       If image_type == "GRAYSCALE":
│       │   color = CLR_WHITE
│       │   symbol_color = CLR_BLACK
│       │
│       Else (colored):
│           │
│           If delta > 0:
│           │   color = verdict_colors["correct"]
│           │   symbol_color = verdict_symbol_colors["positive"]
│           │
│           Elif delta < 0:
│           │   color = verdict_colors["incorrect"]
│           │   symbol_color = verdict_symbol_colors["negative"]
│           │
│           Else:
│               color = verdict_colors["neutral"]
│               symbol_color = verdict_symbol_colors["neutral"]
│           │
│           # Override for bonus (marked but didn't match answer)
│           If bonus_type != None and verdict in {UNMARKED, NO_ANSWER_MATCH}:
│               color = verdict_colors["bonus"]
│               symbol_color = verdict_symbol_colors["bonus"]
│
└─► CASE: NOT is_field_marked (bubble was empty)
    │
    ├─ Default symbol = "" (no symbol)
    │
    ├─ Special bonus handling:
    │   │
    │   If bonus_type == "BONUS_FOR_ALL":
    │   │   symbol = "+"  # All bubbles get bonus
    │   │
    │   Elif bonus_type == "BONUS_ON_ATTEMPT":
    │       │
    │       If question_schema_verdict == "unmarked":
    │       │   # Blank question
    │       │   symbol = "o"  # Neutral symbol
    │       Else:
    │           # At least one bubble marked
    │           symbol = "*"  # Bonus symbol for unmarked bubbles
    │
    └─ Colors:
        │
        If bonus_type != None:
            color = verdict_colors["bonus"]
            symbol_color = verdict_symbol_colors["bonus"]

thickness_factor = 1/12  # For drawing symbols

Return: (symbol, color, symbol_color, thickness_factor)

Used by visualization/drawing code to render verdict feedback on bubbles
```

---

## Format String Flows

### Answers Summary Format

```
get_formatted_answers_summary(answers_summary_format_string=None)
│
├─ If format_string not provided:
│   answers_summary_format_string = self.draw_answers_summary["answers_summary_format_string"]
│   # Default: "Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}"
│
├─ Format with current verdict counts:
│   answers_format = answers_summary_format_string.format(
│       **self.schema_verdict_counts
│   )
│   # e.g., "Correct: 15 Incorrect: 3 Unmarked: 2"
│
├─ Extract position and size:
│   position = self.draw_answers_summary["position"]    # [x, y]
│   size = self.draw_answers_summary["size"]            # font size
│   thickness = int(size * 2)                           # line thickness
│
└─ Return: (answers_format, position, size, thickness)

Used to draw summary text on output image
```

### Score Format

```
get_formatted_score(score)
│
├─ Format score with format string:
│   score_format = self.draw_score["score_format_string"].format(
│       score=round(score, 2)
│   )
│   # e.g., "Score: 45.00" or "Total: 45.00/60.00"
│
├─ Extract position and size:
│   position = self.draw_score["position"]    # [x, y]
│   size = self.draw_score["size"]            # font size
│   thickness = int(size * 2)                 # line thickness
│
└─ Return: (score_format, position, size, thickness)

Used to draw score text on output image
```

---

## Key Data Structures

### Instance Variables

```python
# Set identification
self.set_name: str                    # "DEFAULT_SET", "Set A", "Set B"
self.has_conditional_sets: bool       # True if this is a child set

# Question-answer configuration
self.questions_in_order: list[str]    # ["q1", "q2", ..., "qN"]
self.answers_in_order: list            # Answers (various types)

# Answer matching
self.question_to_answer_matcher: dict[str, AnswerMatcher]
    # Maps each question to its AnswerMatcher instance

# Marking schemes
self.default_marking_scheme: SectionMarkingScheme
self.section_marking_schemes: dict[str, SectionMarkingScheme]
self.question_to_scheme: dict[str, SectionMarkingScheme]
self.has_custom_marking: bool
self.has_streak_marking: bool

# Evaluation state (reset per sheet)
self.schema_verdict_counts: dict[str, int]
    # {"correct": 15, "incorrect": 3, "unmarked": 2}
self.explanation_table: Table | None
self.allow_streak: bool

# Output configuration
self.draw_score: dict
self.draw_answers_summary: dict
self.draw_question_verdicts: dict
self.draw_detected_bubble_texts: dict
self.should_explain_scoring: bool
self.should_export_explanation_csv: bool

# Visualization colors (if draw_question_verdicts enabled)
self.verdict_colors: dict[str, tuple[int, int, int]]
self.verdict_symbol_colors: dict[str, tuple[int, int, int]]
self.draw_answer_groups: dict

# Files to exclude from processing
self.exclude_files: list[Path]
```

### Question-to-Scheme Mapping Example

```python
# Config with custom schemes:
marking_schemes = {
    "DEFAULT": {...},
    "BONUS_SECTION": {
        "questions": ["q11", "q12", "q13"],
        "marking": {...}
    },
    "HIGH_VALUE": {
        "questions": ["q1", "q2", "q3"],
        "marking": {...}
    }
}

# Resulting maps:
section_marking_schemes = {
    "BONUS_SECTION": SectionMarkingScheme(...),
    "HIGH_VALUE": SectionMarkingScheme(...)
}

question_to_scheme = {
    "q1": section_marking_schemes["HIGH_VALUE"],
    "q2": section_marking_schemes["HIGH_VALUE"],
    "q3": section_marking_schemes["HIGH_VALUE"],
    "q11": section_marking_schemes["BONUS_SECTION"],
    "q12": section_marking_schemes["BONUS_SECTION"],
    "q13": section_marking_schemes["BONUS_SECTION"]
}

# Questions q4-q10, q14-q20 not in map → use default_marking_scheme
```

---

## Browser Migration Notes

### TypeScript Interface

```typescript
interface EvaluationConfigForSet {
    setName: string;
    questionsInOrder: string[];
    answersInOrder: (string | string[] | [string, number][])[];

    // Answer matching
    questionToAnswerMatcher: Map<string, AnswerMatcher>;

    // Marking schemes
    defaultMarkingScheme: SectionMarkingScheme;
    sectionMarkingSchemes: Map<string, SectionMarkingScheme>;
    questionToScheme: Map<string, SectionMarkingScheme>;
    hasCustomMarking: boolean;
    hasStreakMarking: boolean;
    hasConditionalSets: boolean;

    // Evaluation state
    schemaVerdictCounts: {
        correct: number;
        incorrect: number;
        unmarked: number;
    };
    explanationTable: ExplanationRow[] | null;
    allowStreak: boolean;

    // Output configuration
    shouldExplainScoring: boolean;
    shouldExportExplanationCsv: boolean;
    drawScore: DrawScoreConfig;
    drawAnswersSummary: DrawAnswersSummaryConfig;
    drawQuestionVerdicts: DrawQuestionVerdictsConfig;

    // Methods
    prepareAndValidateOmrResponse(response: OmrResponse, allowStreak: boolean): void;
    matchAnswerForQuestion(currentScore: number, question: string, markedAnswer: string): MatchResult;
    getFormattedAnswersSummary(): FormattedText;
    getFormattedScore(score: number): FormattedText;
    getEvaluationMetaForQuestion(questionMeta: QuestionMeta, isFieldMarked: boolean, imageType: string): EvaluationMeta;
    resetEvaluation(): void;
}
```

### CSV Parsing in Browser

```typescript
// Replace pandas.read_csv with Papa Parse or custom parser
import Papa from 'papaparse';

async function parseCsvQuestionAnswers(
    file: File,
    options: AnswerKeyOptions
): Promise<[string[], any[]]> {
    const csvPath = options.answer_key_csv_path;

    // Read CSV file
    const csvText = await file.text();

    const result = Papa.parse(csvText, {
        header: false,
        skipEmptyLines: true,
        transform: (value, column) => {
            if (column === 0) {
                // Question column
                return value.trim();
            } else {
                // Answer column
                return parseAnswerColumn(value);
            }
        }
    });

    const questionsInOrder = result.data.map(row => row[0]);
    const answersInOrder = result.data.map(row => row[1]);

    return [questionsInOrder, answersInOrder];
}

function parseAnswerColumn(answerStr: string): string | string[] | [string, number][] {
    // Remove whitespace
    const cleaned = answerStr.replace(/\s/g, '');

    if (cleaned.startsWith('[')) {
        // Weighted: "[['A', 2], ['B', 1]]"
        return JSON.parse(cleaned);
    } else if (cleaned.includes(',')) {
        // Multiple correct: "A,B,C"
        return cleaned.split(',');
    } else {
        // Standard: "A"
        return cleaned;
    }
}
```

### Image-based Answer Key in Browser

```typescript
async function parseImageQuestionAnswers(
    imageFile: File,
    template: Template,
    tuningConfig: TuningConfig
): Promise<[string[], string[]]> {
    // Read image file
    const imageBitmap = await createImageBitmap(imageFile);
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(imageBitmap, 0, 0);

    // Convert to grayscale Mat
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const grayImage = rgbaToGray(imageData);
    const coloredImage = imageData;

    // Process through full pipeline
    const context = await template.processFile(imageFile.name, grayImage, coloredImage);
    const omrResponse = context.omrResponse;

    // Extract non-empty answers
    const emptyValue = template.globalEmptyVal;
    const emptyRegex = emptyValue === '' ? /^$/ : new RegExp(`${emptyValue}+`);

    const questionsInOrder: string[] = [];
    const answersInOrder: string[] = [];

    for (const [question, answer] of Object.entries(omrResponse)) {
        if (!question.startsWith('q')) continue;
        if (emptyRegex.test(answer)) continue;

        questionsInOrder.push(question);
        answersInOrder.push(answer);
    }

    return [questionsInOrder.sort(), answersInOrder];
}
```

### Explanation Export in Browser

```typescript
function exportExplanationCsv(
    explanationTable: ExplanationRow[],
    fileName: string
): void {
    // Convert explanation table to CSV
    const csv = Papa.unparse(explanationTable, {
        quotes: true,
        header: true
    });

    // Trigger download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `${fileName}.csv`;
    link.click();

    URL.revokeObjectURL(url);
}
```

### Zod Validation Schema

```typescript
import { z } from 'zod';

const AnswerItemSchema = z.union([
    z.string(),                                    // Standard: "A"
    z.array(z.string()),                          // Multiple: ["A", "B"]
    z.array(z.tuple([z.string(), z.number()]))    // Weighted: [["A", 2], ["B", 1]]
]);

const MarkingSchemeSchema = z.object({
    marking_type: z.enum(['default', 'verdict_level_streak', 'section_level_streak']).optional(),
    marking: z.object({
        correct: z.union([z.number(), z.array(z.number())]),
        incorrect: z.union([z.number(), z.array(z.number())]),
        unmarked: z.union([z.number(), z.array(z.number())])
    }),
    questions: z.array(z.string()).optional()
});

const EvaluationConfigSchema = z.object({
    source_type: z.enum(['local', 'csv', 'image_and_csv']),
    options: z.object({
        questions_in_order: z.array(z.string()).optional(),
        answers_in_order: z.array(AnswerItemSchema).optional(),
        answer_key_csv_path: z.string().optional(),
        answer_key_image_path: z.string().optional()
    }),
    outputs_configuration: z.object({
        should_explain_scoring: z.boolean(),
        should_export_explanation_csv: z.boolean().optional(),
        draw_score: z.object({
            enabled: z.boolean(),
            position: z.tuple([z.number(), z.number()]),
            size: z.number(),
            score_format_string: z.string().optional()
        }),
        draw_answers_summary: z.object({
            enabled: z.boolean(),
            position: z.tuple([z.number(), z.number()]),
            size: z.number(),
            answers_summary_format_string: z.string().optional()
        }),
        draw_question_verdicts: z.object({
            enabled: z.boolean(),
            verdict_colors: z.record(z.string()),
            verdict_symbol_colors: z.record(z.string()),
            draw_answer_groups: z.object({
                color_sequence: z.array(z.string())
            })
        }).optional()
    }),
    marking_schemes: z.record(MarkingSchemeSchema),
    conditional_sets: z.array(z.object({
        name: z.string(),
        matcher: z.object({
            format_string: z.string(),
            match_regex: z.string()
        }),
        evaluation: z.lazy(() => EvaluationConfigSchema)
    })).optional()
});
```

---

## Related Files

**Dependencies**:
- `src/processors/evaluation/evaluation_config.py` - Parent evaluator with conditional set matching
- `src/processors/evaluation/answer_matcher.py` - Individual answer matching logic
- `src/processors/evaluation/section_marking_scheme.py` - Section-level scoring rules
- `src/utils/parsing.py` - Field string parsing, config merging
- `src/schemas/constants.py` - Verdict types, answer types

**Used By**:
- `src/processors/evaluation/evaluation_config.py` - Creates instances per set
- Template processing pipeline - Uses for scoring OMR responses

**Configuration Files**:
- `evaluation.json` - Answer keys, marking schemes, conditional sets
- `answer-key.csv` - CSV-based answer keys
- Answer key images - Scanned answer sheets for auto-generation
