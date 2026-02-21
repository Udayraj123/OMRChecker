# Evaluation Config For Set Constraints

**Module**: Domain - Evaluation - Config For Set
**Python Reference**: `src/processors/evaluation/evaluation_config_for_set.py`
**Last Updated**: 2026-02-21

---

## Performance Constraints

### Memory Usage

**Python Implementation**:
```python
# Instance reused across multiple OMR sheets
class EvaluationConfigForSet:
    def __init__(self, set_name, ...):
        # Shared state across all sheets in this set
        self.questions_in_order: list[str]           # ~1KB for 100 questions
        self.answers_in_order: list                  # ~2KB for 100 answers
        self.question_to_answer_matcher: dict        # ~10KB for 100 questions

    def prepare_and_validate_omr_response(self, omr_response, allow_streak):
        # Per-sheet state (reset on each sheet)
        self.schema_verdict_counts = {...}          # ~100 bytes
        self.explanation_table = Table(...)         # ~10KB for 100 rows
        # Reset streaks in all section schemes
```

**Memory Profile**:
- **Per-set overhead**: 15-20 KB for 100 questions
- **Per-sheet overhead**: 10-15 KB with explanation enabled
- **Peak memory**: During answer key image processing (full pipeline run)

**Browser Considerations**:
- **Shared instance**: One EvaluationConfigForSet per set (default + conditionals)
- **Reset per sheet**: Clear explanation table and verdict counts
- **Garbage collection**: Explanation tables can be large for hundreds of questions
- **Memory limit**: Browser heap typically 1-2GB (mobile: 256-512MB)

**Optimization Strategies**:
```typescript
class EvaluationConfigForSet {
    // Shared immutable state
    private readonly questionsInOrder: readonly string[];
    private readonly answersInOrder: readonly any[];
    private readonly questionToAnswerMatcher: ReadonlyMap<string, AnswerMatcher>;

    // Mutable per-sheet state
    private schemaVerdictCounts: VerdictCounts;
    private explanationTable: ExplanationRow[] | null;

    resetEvaluation(): void {
        // Clear per-sheet state
        this.schemaVerdictCounts = { correct: 0, incorrect: 0, unmarked: 0 };
        this.explanationTable = this.shouldExplainScoring ? [] : null;

        // Reset all section streaks
        for (const scheme of this.sectionMarkingSchemes.values()) {
            scheme.resetAllStreaks();
        }
    }

    // Optional: Stream explanation to IndexedDB for large sheets
    async exportExplanationToIndexedDb(fileId: string): Promise<void> {
        if (!this.explanationTable) return;

        const db = await openExplanationDb();
        const tx = db.transaction('explanations', 'readwrite');

        for (const row of this.explanationTable) {
            await tx.store.add({ fileId, ...row });
        }

        await tx.done;

        // Clear from memory
        this.explanationTable = null;
    }
}
```

---

### Processing Time

**Python Timings** (100 questions, default set):
```
Initialization:
├─ parse_local_question_answers:        <1ms
├─ parse_csv_question_answers:          5-10ms (pandas CSV read)
├─ parse_image_question_answers:        500-2000ms (full pipeline)
├─ merge_with_parent:                   <1ms
├─ set_parsed_marking_schemes:          <1ms
├─ parse_answers_and_map_questions:     1-2ms
└─ validate_format_strings:             <1ms
Total (local):                          5-15ms
Total (image):                          500-2000ms

Per-sheet evaluation (100 questions):
├─ prepare_and_validate_omr_response:   <1ms
├─ match_answer_for_question × 100:     5-10ms
├─ prepare_explanation_table:           2-5ms (Rich Table)
└─ export_explanation_csv:              10-20ms (pandas to_csv)
Total per sheet:                        17-35ms
```

**Browser Timings** (estimated):
```
Initialization:
├─ parse local:                         <1ms (native JS)
├─ parse CSV:                           10-20ms (Papa Parse)
├─ parse image:                         1000-5000ms (OpenCV.js pipeline)
└─ other steps:                         5-10ms
Total (local):                          10-30ms
Total (image):                          1000-5000ms

Per-sheet evaluation (100 questions):
├─ prepare and validate:                <1ms
├─ match × 100:                         10-20ms (no pandas overhead)
├─ prepare explanation:                 5-10ms (DOM table)
└─ export explanation CSV:              20-50ms (Papa.unparse + download)
Total per sheet:                        35-80ms
```

**Performance Targets**:
- **Initialization**: <50ms for local/CSV, <5s for image
- **Per-sheet evaluation**: <100ms for 100 questions
- **Conditional set matching**: <5ms per sheet
- **Explanation export**: <100ms per sheet

**Browser Optimization**:
```typescript
// Lazy initialization of expensive resources
class EvaluationConfigForSet {
    private answerMatcherCache?: Map<string, AnswerMatcher>;

    private getAnswerMatcher(question: string): AnswerMatcher {
        if (!this.answerMatcherCache) {
            // Build cache on first use
            this.answerMatcherCache = new Map();
            for (let i = 0; i < this.questionsInOrder.length; i++) {
                const q = this.questionsInOrder[i];
                const answer = this.answersInOrder[i];
                const scheme = this.getMarkingSchemeForQuestion(q);
                this.answerMatcherCache.set(q, new AnswerMatcher(answer, scheme));
            }
        }
        return this.answerMatcherCache.get(question)!;
    }

    // Use Web Worker for image-based answer key generation
    async parseImageQuestionAnswersInWorker(
        imageFile: File,
        template: Template
    ): Promise<[string[], any[]]> {
        const worker = new Worker('answer-key-worker.js');

        return new Promise((resolve, reject) => {
            worker.onmessage = (e) => {
                resolve([e.data.questions, e.data.answers]);
                worker.terminate();
            };
            worker.onerror = reject;

            worker.postMessage({ imageFile, template });
        });
    }
}
```

---

## Data Constraints

### Answer Key Formats

**Question Naming**:
```python
# Valid question identifiers
questions_in_order = [
    "q1", "q2", ..., "q100",           # Simple numeric
    "q01", "q02", ..., "q100",         # Zero-padded
    "section1_q1", "section1_q2",      # Prefixed
    "Q1A", "Q1B",                      # Alpha suffix
]

# Field string expansion
"q1..20"      → ["q1", "q2", ..., "q20"]
"q01..20"     → ["q01", "q02", ..., "q20"]
"q1a..1d"     → ["q1a", "q1b", "q1c", "q1d"]

# Invalid patterns
"1", "2", "3"                # Must start with non-digit or have "q" prefix
"q_11..20"                   # Underscore not supported in ranges
```

**Answer Formats**:
```python
# Standard answer (single correct)
answer = "A"                           # String

# Multiple correct answers
answer = ["A", "B", "AB"]              # List of strings

# Multiple correct weighted answers
answer = [
    ["A", 2],                          # Full credit
    ["B", 1],                          # Partial credit
    ["AB", 2.5],                       # Bonus credit
    ["C", 0]                           # No credit
]

# CSV representation
"A"                                    # Standard
"A,B,AB"                               # Multiple (comma-separated)
"[['A', 2], ['B', 1], ['AB', 2.5]]"    # Weighted (JSON array)
```

**Constraints**:
- **Question count**: No hard limit (tested up to 500 questions)
- **Answer length**: No hard limit (multi-character answers supported)
- **Set name length**: No limit (used as dict key)
- **Weighted scores**: Support integers, floats, fractions ("1/2", "2/3")

**Browser Validation**:
```typescript
const QuestionIdSchema = z.string().regex(
    /^([^\.\d]+\d+|[^\.\d]+\d+\.\{2,3\}\d+)$/,
    'Invalid question ID format'
);

const StandardAnswerSchema = z.string().min(1);

const MultipleCorrectSchema = z.array(z.string()).min(2);

const WeightedAnswerSchema = z.array(
    z.tuple([z.string(), z.number()])
).min(1);

const AnswerSchema = z.union([
    StandardAnswerSchema,
    MultipleCorrectSchema,
    WeightedAnswerSchema
]);

// Validate answer key
function validateAnswerKey(
    questions: string[],
    answers: any[]
): void {
    if (questions.length !== answers.length) {
        throw new Error(
            `Questions (${questions.length}) and answers (${answers.length}) count mismatch`
        );
    }

    for (let i = 0; i < questions.length; i++) {
        QuestionIdSchema.parse(questions[i]);
        AnswerSchema.parse(answers[i]);
    }
}
```

---

### Marking Schemes

**Section Constraints**:
```python
marking_schemes = {
    # DEFAULT is required
    "DEFAULT": {
        "correct": 3,      # Required
        "incorrect": -1,   # Required
        "unmarked": 0      # Required
    },

    # Custom sections
    "SECTION_A": {
        "marking_type": "default",  # Optional, default: "default"
        "questions": ["q1..10"],    # Required for non-DEFAULT
        "marking": {                # Required
            "correct": 4,
            "incorrect": -1,
            "unmarked": 0
        }
    },

    # Streak-based section
    "STREAK_SECTION": {
        "marking_type": "verdict_level_streak",
        "questions": ["q11..20"],
        "marking": {
            "correct": [1, 2, 3, 5, 8],      # Array for streak bonuses
            "incorrect": [-1, -2, -3],        # Array for increasing penalties
            "unmarked": 0
        }
    }
}
```

**Constraints**:
- **DEFAULT scheme**: Required in every config
- **Section overlap**: Questions cannot appear in multiple sections
- **Missing questions**: All scheme questions must exist in answer key
- **Streak arrays**: Length should match or exceed question count
- **Score types**: Support int, float, fractions, arrays (for streaks)
- **Negative scores**: Allowed (for penalties)

**Validation Example**:
```typescript
interface MarkingScheme {
    marking_type?: 'default' | 'verdict_level_streak' | 'section_level_streak';
    questions?: string[];
    marking: {
        correct: number | number[];
        incorrect: number | number[];
        unmarked: number | number[];
    };
}

function validateMarkingSchemes(
    schemes: Record<string, MarkingScheme>,
    questions: string[]
): void {
    // Ensure DEFAULT exists
    if (!schemes.DEFAULT) {
        throw new Error('DEFAULT marking scheme is required');
    }

    // Track question assignments
    const assignedQuestions = new Set<string>();

    for (const [key, scheme] of Object.entries(schemes)) {
        if (key === 'DEFAULT') continue;

        if (!scheme.questions || scheme.questions.length === 0) {
            throw new Error(`Scheme '${key}' missing questions`);
        }

        // Check for overlaps
        for (const q of scheme.questions) {
            if (assignedQuestions.has(q)) {
                throw new Error(`Question '${q}' assigned to multiple sections`);
            }
            assignedQuestions.add(q);

            // Check question exists in answer key
            if (!questions.includes(q)) {
                throw new Error(`Question '${q}' in scheme '${key}' not in answer key`);
            }
        }

        // Validate streak arrays
        if (scheme.marking_type?.includes('streak')) {
            const { correct, incorrect, unmarked } = scheme.marking;

            if (Array.isArray(correct) && correct.length < scheme.questions.length) {
                console.warn(
                    `Scheme '${key}': correct array length (${correct.length}) ` +
                    `< questions count (${scheme.questions.length})`
                );
            }
        }
    }
}
```

---

### Conditional Sets

**Matcher Constraints**:
```python
conditional_sets = [
    {
        "name": "Set A",
        "matcher": {
            "format_string": "{q21}",           # Must be valid Python format string
            "match_regex": "A"                  # Must be valid regex
        },
        "evaluation": {
            # Can override any part of default config
            "source_type": "local",
            "options": {
                "questions_in_order": ["q1..20"],
                "answers_in_order": [...]
            },
            "marking_schemes": {...}
        }
    }
]
```

**Format String Variables**:
```python
# Available in format_string:
- {q1}, {q2}, ..., {qN}     # Any question from OMR response
- {file_path}               # Full path: "/path/to/IMG_001.jpg"
- {file_name}               # Just name: "IMG_001.jpg"

# Examples:
"{q21}"                     # Match on single question
"{q1}_{q2}"                 # Combine multiple questions
"{file_name}"               # Match on file name pattern
```

**Regex Patterns**:
```python
# Match exact value
"match_regex": "A"                    # Matches "A" in formatted string

# Match pattern
"match_regex": "Set[ABC]"             # Matches "SetA", "SetB", "SetC"
"match_regex": "^[0-9]{2}$"           # Matches two digits
"match_regex": "IMG_.*\.jpg"          # Matches file pattern
```

**Constraints**:
- **Set name uniqueness**: No duplicate set names
- **Format string validity**: Must be valid Python format string
- **Regex validity**: Must be valid regex pattern
- **Variable availability**: Format string variables must exist in OMR response
- **Evaluation order**: First matched set wins (order matters)
- **Parent inheritance**: Child sets can override parent's questions/schemes

**Browser Implementation**:
```typescript
interface ConditionalSet {
    name: string;
    matcher: {
        formatString: string;
        matchRegex: string;
    };
    evaluation: EvaluationConfig;
}

class EvaluationConfig {
    private conditionalSets: ConditionalSet[] = [];

    getMatchingSet(
        omrResponse: Record<string, string>,
        filePath: string,
        fileName: string
    ): string | null {
        const formattingFields = {
            ...omrResponse,
            file_path: filePath,
            file_name: fileName
        };

        for (const set of this.conditionalSets) {
            try {
                // Simple template string replacement
                let formatted = set.matcher.formatString;
                for (const [key, value] of Object.entries(formattingFields)) {
                    formatted = formatted.replace(`{${key}}`, value);
                }

                const regex = new RegExp(set.matcher.matchRegex);
                if (regex.test(formatted)) {
                    return set.name;
                }
            } catch (e) {
                console.error(`Error matching set '${set.name}':`, e);
                continue;
            }
        }

        return null; // Use default set
    }
}
```

---

## Output Constraints

### Visualization Settings

**Draw Score**:
```python
"draw_score": {
    "enabled": true,
    "position": [650, 750],              # [x, y] in pixels
    "size": 1.5,                         # Font scale
    "score_format_string": "Score: {score}"  # Format string
}
```

**Constraints**:
- **Position**: Must fit within image bounds
- **Size**: Positive float (typically 0.5-3.0)
- **Format string**: Must include `{score}` placeholder

**Draw Answers Summary**:
```python
"draw_answers_summary": {
    "enabled": true,
    "position": [650, 820],
    "size": 0.85,
    "answers_summary_format_string": "Correct: {correct} Incorrect: {incorrect} Unmarked: {unmarked}"
}
```

**Constraints**:
- **Position**: Must fit within image bounds
- **Size**: Positive float
- **Format string**: Can include `{correct}`, `{incorrect}`, `{unmarked}`

**Draw Question Verdicts**:
```python
"draw_question_verdicts": {
    "enabled": true,
    "verdict_colors": {
        "correct": "#00FF00",            # Green
        "incorrect": "#FF0000",          # Red
        "neutral": "#FFFF00",            # Yellow
        "bonus": "#00FFFF"               # Cyan
    },
    "verdict_symbol_colors": {
        "positive": "#00FF00",
        "negative": "#FF0000",
        "neutral": "#FFFF00",
        "bonus": "#FF00FF"
    },
    "draw_answer_groups": {
        "color_sequence": ["#FF0000", "#00FF00", "#0000FF"]
    }
}
```

**Constraints**:
- **Colors**: Hex RGB format (#RRGGBB) or named colors
- **Color sequence**: Array of colors for answer groups
- **Symbols**: Fixed set (+, -, o, *, "")

**Browser Color Handling**:
```typescript
interface VerdictColors {
    correct: string;
    incorrect: string;
    neutral: string;
    bonus: string;
}

function parseHexColor(hex: string): [number, number, number] {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) {
        throw new Error(`Invalid hex color: ${hex}`);
    }
    return [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16)
    ];
}

// OpenCV.js uses BGR, Canvas uses RGB
function hexToBgr(hex: string): [number, number, number] {
    const [r, g, b] = parseHexColor(hex);
    return [b, g, r]; // Swap R and B for BGR
}

function hexToRgb(hex: string): string {
    const [r, g, b] = parseHexColor(hex);
    return `rgb(${r}, ${g}, ${b})`;
}
```

---

### Explanation Table

**Structure**:
```python
# Columns (conditionally included):
- Marking Scheme      # If has_custom_marking
- Question
- Marked
- Answer(s)
- Verdict
- Delta
- Score
- Set Mapping         # If has_conditional_sets
- Streak              # If has_streak_marking and allow_streak
```

**Constraints**:
- **Row count**: Equal to number of questions
- **Column count**: 6-9 depending on configuration
- **Memory**: ~100 bytes per row (10KB for 100 questions)
- **Export format**: CSV with QUOTE_NONNUMERIC

**Browser Table Rendering**:
```typescript
interface ExplanationRow {
    markingScheme?: string;
    question: string;
    marked: string;
    answers: string;
    verdict: string;
    delta: string;
    score: string;
    setMapping?: string;
    streak?: string;
}

function renderExplanationTable(rows: ExplanationRow[]): HTMLTableElement {
    const table = document.createElement('table');
    table.className = 'explanation-table';

    // Header
    const thead = table.createTHead();
    const headerRow = thead.insertRow();
    const columns = [
        rows[0].markingScheme !== undefined && 'Marking Scheme',
        'Question',
        'Marked',
        'Answer(s)',
        'Verdict',
        'Delta',
        'Score',
        rows[0].setMapping !== undefined && 'Set Mapping',
        rows[0].streak !== undefined && 'Streak'
    ].filter(Boolean);

    for (const col of columns) {
        const th = document.createElement('th');
        th.textContent = col as string;
        headerRow.appendChild(th);
    }

    // Body
    const tbody = table.createTBody();
    for (const row of rows) {
        const tr = tbody.insertRow();
        for (const col of columns) {
            const td = tr.insertCell();
            td.textContent = row[col as keyof ExplanationRow] || '';
        }
    }

    return table;
}
```

---

## Source Type Constraints

### Local Source

```python
"source_type": "local"
```

**Requirements**:
- `questions_in_order`: Required (array of field strings)
- `answers_in_order`: Required (array of answers)

**Constraints**:
- Arrays must have equal length
- Questions must be valid field identifiers
- Answers must match supported formats

---

### CSV Source

```python
"source_type": "csv"
```

**Requirements**:
- `answer_key_csv_path`: Required (path to CSV file)
- `questions_in_order`: Optional (if provided, validates against CSV)

**CSV Format**:
```csv
q1,A
q2,B
q3,A,B
q4,"[['A', 2], ['B', 1]]"
```

**Constraints**:
- CSV must have exactly 2 columns (no header)
- Column 1: Question IDs
- Column 2: Answers (parsed via `parse_answer_column`)
- File must exist and be readable

**Browser CSV Reading**:
```typescript
async function readCsvFile(file: File): Promise<[string[], any[]]> {
    const text = await file.text();

    const result = Papa.parse(text, {
        header: false,
        skipEmptyLines: true,
        dynamicTyping: false // Keep as strings for custom parsing
    });

    const questions: string[] = [];
    const answers: any[] = [];

    for (const row of result.data as string[][]) {
        if (row.length !== 2) {
            throw new Error(`Invalid CSV row: expected 2 columns, got ${row.length}`);
        }
        questions.push(row[0].trim());
        answers.push(parseAnswerColumn(row[1]));
    }

    return [questions, answers];
}
```

---

### Image Source

```python
"source_type": "image_and_csv"
```

**Requirements**:
- `answer_key_csv_path`: Required (may not exist if image fallback used)
- `answer_key_image_path`: Required (used if CSV missing)
- `questions_in_order`: Optional

**Processing Flow**:
1. Try to read CSV
2. If CSV missing, process image through full pipeline
3. Extract non-empty answers from OMR response
4. Generate answer key from detected values

**Constraints**:
- Image must be processable by template
- All questions must have non-empty detected values
- Image file excluded from batch processing (added to exclude_files)
- Processing time: 500-5000ms (full pipeline run)

**Browser Image Processing**:
```typescript
async function generateAnswerKeyFromImage(
    imageFile: File,
    template: Template,
    tuningConfig: TuningConfig
): Promise<[string[], string[]]> {
    // Show progress indicator
    const progress = showProgress('Generating answer key from image...');

    try {
        // Process image (may take 1-5 seconds)
        const context = await template.processFile(imageFile, tuningConfig);
        const omrResponse = context.omrResponse;

        // Extract non-empty answers
        const emptyRegex = template.globalEmptyVal === ''
            ? /^$/
            : new RegExp(`${template.globalEmptyVal}+`);

        const questions: string[] = [];
        const answers: string[] = [];

        for (const [q, a] of Object.entries(omrResponse)) {
            if (!q.startsWith('q')) continue;
            if (emptyRegex.test(a)) {
                throw new Error(`Empty answer detected for question ${q}`);
            }
            questions.push(q);
            answers.push(a);
        }

        return [questions.sort(), answers];
    } finally {
        progress.close();
    }
}
```

---

## File Exclusion

**Purpose**: Prevent answer key images from being processed as student sheets

```python
self.exclude_files: list[Path] = []

# When answer key image processed:
self.exclude_files.append(image_path)

# Parent EvaluationConfig aggregates:
self.exclude_files = (
    self.default_evaluation_config.get_exclude_files() +
    sum([set_config.get_exclude_files() for set_config in conditional_sets], [])
)

# During file discovery:
if file_path in evaluation_config.get_exclude_files():
    continue  # Skip this file
```

**Browser Implementation**:
```typescript
class EvaluationConfig {
    private excludeFiles: Set<string> = new Set();

    addExcludeFile(fileName: string): void {
        this.excludeFiles.add(fileName);
    }

    shouldExcludeFile(fileName: string): boolean {
        return this.excludeFiles.has(fileName);
    }

    // During batch processing:
    async processBatch(files: File[]): Promise<Result[]> {
        const results: Result[] = [];

        for (const file of files) {
            if (this.shouldExcludeFile(file.name)) {
                console.log(`Skipping excluded file: ${file.name}`);
                continue;
            }

            const result = await this.processFile(file);
            results.push(result);
        }

        return results;
    }
}
```

---

## Browser-Specific Migration Constraints

### Local Storage Limits

**Problem**: Evaluation configs can be large (especially with image-based answer keys)

**Solution**: Store in IndexedDB instead of localStorage

```typescript
interface StoredEvaluation {
    id: string;
    name: string;
    config: EvaluationConfig;
    timestamp: number;
}

async function saveEvaluationConfig(
    name: string,
    config: EvaluationConfig
): Promise<void> {
    const db = await openDatabase();
    const tx = db.transaction('evaluations', 'readwrite');

    await tx.store.put({
        id: crypto.randomUUID(),
        name,
        config: JSON.parse(JSON.stringify(config)), // Deep clone
        timestamp: Date.now()
    });

    await tx.done;
}

async function loadEvaluationConfig(name: string): Promise<EvaluationConfig | null> {
    const db = await openDatabase();
    const tx = db.transaction('evaluations', 'readonly');
    const stored = await tx.store.get(name);

    return stored ? deserializeConfig(stored.config) : null;
}
```

---

### Worker Thread Constraints

**Answer Key Image Processing**:
- Should run in Web Worker to avoid blocking UI
- Transfer template config to worker
- Return questions/answers via postMessage

```typescript
// Main thread
async function parseImageAnswerKey(
    imageFile: File,
    template: Template
): Promise<[string[], any[]]> {
    const worker = new Worker('/workers/answer-key-worker.js', { type: 'module' });

    return new Promise((resolve, reject) => {
        worker.onmessage = (e) => {
            if (e.data.error) {
                reject(new Error(e.data.error));
            } else {
                resolve([e.data.questions, e.data.answers]);
            }
            worker.terminate();
        };

        worker.onerror = reject;

        // Transfer image file
        worker.postMessage({
            imageFile,
            templateConfig: template.toJSON()
        });
    });
}

// Worker thread (answer-key-worker.js)
self.onmessage = async (e) => {
    try {
        const { imageFile, templateConfig } = e.data;

        // Reconstruct template in worker
        const template = Template.fromJSON(templateConfig);

        // Process image
        const context = await template.processFile(imageFile);

        // Extract answers
        const questions = [];
        const answers = [];
        for (const [q, a] of Object.entries(context.omrResponse)) {
            if (q.startsWith('q') && a !== '') {
                questions.push(q);
                answers.push(a);
            }
        }

        self.postMessage({ questions, answers });
    } catch (error) {
        self.postMessage({ error: error.message });
    }
};
```

---

### Performance Budget

**Target Metrics** (100 questions):
- **Config initialization**: <50ms (local), <5s (image)
- **Per-sheet evaluation**: <100ms
- **Explanation export**: <100ms
- **Memory footprint**: <5MB per config instance

**Monitoring**:
```typescript
class EvaluationConfigForSet {
    async initialize(): Promise<void> {
        const startTime = performance.now();

        // Initialization logic
        await this.parseQuestionAnswers();
        await this.setupMarkingSchemes();

        const duration = performance.now() - startTime;

        // Log performance metric
        performance.measure('evaluation-init', {
            start: startTime,
            duration
        });

        if (duration > 5000) {
            console.warn(`Slow evaluation init: ${duration}ms`);
        }
    }

    matchAnswerForQuestion(
        currentScore: number,
        question: string,
        markedAnswer: string
    ): MatchResult {
        const startTime = performance.now();

        const result = this._matchAnswerForQuestion(currentScore, question, markedAnswer);

        const duration = performance.now() - startTime;
        if (duration > 10) {
            console.warn(`Slow answer match for ${question}: ${duration}ms`);
        }

        return result;
    }
}
```

---

## Related Files

**Dependencies**:
- `src/processors/evaluation/evaluation_config.py` - Parent config manager
- `src/processors/evaluation/answer_matcher.py` - Answer type detection & matching
- `src/processors/evaluation/section_marking_scheme.py` - Section-level scoring
- `src/utils/parsing.py` - Field string parsing, CSV parsing
- `src/utils/image.py` - Image reading for answer key generation
- Template pipeline - Full OMR processing for image-based answer keys

**Configuration Schema**:
- JSON Schema validation for evaluation.json structure
- Zod schema for browser runtime validation
- CSV format specification (2 columns, no header)

**Performance Profiling**:
- Python: cProfile, memory_profiler
- Browser: Performance API, Chrome DevTools
