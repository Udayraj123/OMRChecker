export type ScanResult = {
  id?: number | string;
  applicationNumber: string;
  lrn: string;
  surname: string;
  name: string;
  middleName: string;
  examDate?: string;
  sourceFileName?: string;
  languageProficiency: string;
  mathematics: string;
  science: string;
  logicAndAbstractReasoning: string;
  generalKnowledge: string;
  mechanicalReasoning: string;
  languageProficiencyDetected?: number;
  mathematicsDetected?: number;
  scienceDetected?: number;
  logicAndAbstractReasoningDetected?: number;
  generalKnowledgeDetected?: number;
  mechanicalReasoningDetected?: number;
  languageProficiencyScore?: number;
  mathematicsScore?: number;
  scienceScore?: number;
  logicAndAbstractReasoningScore?: number;
  generalKnowledgeScore?: number;
  mechanicalReasoningScore?: number;
  fileName: string;
  checkedImagePath?: string;
  deletedAt?: string | null;
};

export const ANSWER_KEY_STORE = "minsu-answer-key-editor-v3";
export const answerChoices = ["A", "B", "C", "D", "E"] as const;

export type AnswerChoice = (typeof answerChoices)[number];
export type AnswerKeyState = Record<string, Array<AnswerChoice | "">>;

export const defaultAnswerKeyState: AnswerKeyState = {
  "Language Proficiency": [
    "B", "A", "C", "B", "B",
    "A", "B", "A", "C", "B",
    "C", "B", "C", "D", "D",
    "A", "E", "B", "D", "B",
    "E",
    "B", "D", "B", "B", "C",
    "B", "B", "B", "A", "B",
    "B", "C", "C", "B",
  ],
  Mathematics: [
    "C", "B", "A", "B", "A",
    "B", "B", "B", "A", "B",
    "B", "A", "E", "C", "D",
    "C", "A", "C", "E", "D",
    "D", "C", "D", "A", "A",
    "C", "B", "D", "C", "A",
    "C", "D", "B", "E", "D",
  ],
  Science: [
    "E", "B", "D", "B", "C",
    "B", "B", "C", "A", "C",
    "B", "C", "B", "A", "B",
    "A", "A", "A", "A", "B",
    "D", "B", "B", "C", "C",
    "D", "C", "A", "C", "C",
    "C", "B", "B", "B", "B",
  ],
  "Logic and Abstract Reasoning": [
    "C", "D", "C", "B", "A",
    "C", "A", "D", "A", "C",
    "D", "C", "C", "B", "D",
    "B", "A", "A", "D", "D",
  ],
  "General Knowledge": [
    "B", "B", "B", "C", "B",
    "B", "C", "B", "B", "B",
    "B", "A", "B", "D", "C",
    "C", "B", "B", "B", "C",
    "C", "C", "B", "B", "B",
    "C", "B", "B", "B", "B",
    "D", "A", "C", "A", "B",
  ],
  "Mechanical Reasoning": [
    "B", "B", "C", "C", "C",
    "C", "C", "A", "C", "C",
    "C", "B", "B", "B", "B",
    "C", "D", "C", "D", "D",
  ],
};

type SubjectScoreConfig = {
  subject: string;
  answerKey: keyof ScanResult;
  detectedKey: keyof ScanResult;
  scoreKey: keyof ScanResult;
  total: number;
};

export const subjectScoreConfig: SubjectScoreConfig[] = [
  {
    subject: "Language Proficiency",
    answerKey: "languageProficiency",
    detectedKey: "languageProficiencyDetected",
    scoreKey: "languageProficiencyScore",
    total: 35,
  },
  {
    subject: "Mathematics",
    answerKey: "mathematics",
    detectedKey: "mathematicsDetected",
    scoreKey: "mathematicsScore",
    total: 35,
  },
  {
    subject: "Science",
    answerKey: "science",
    detectedKey: "scienceDetected",
    scoreKey: "scienceScore",
    total: 35,
  },
  {
    subject: "Logic and Abstract Reasoning",
    answerKey: "logicAndAbstractReasoning",
    detectedKey: "logicAndAbstractReasoningDetected",
    scoreKey: "logicAndAbstractReasoningScore",
    total: 20,
  },
  {
    subject: "General Knowledge",
    answerKey: "generalKnowledge",
    detectedKey: "generalKnowledgeDetected",
    scoreKey: "generalKnowledgeScore",
    total: 35,
  },
  {
    subject: "Mechanical Reasoning",
    answerKey: "mechanicalReasoning",
    detectedKey: "mechanicalReasoningDetected",
    scoreKey: "mechanicalReasoningScore",
    total: 20,
  },
];

export const excelHeaders = [
  "Application number",
  "LRN",
  "surname",
  "name",
  "middle name",
  "Date",
  "Language proficiency",
  "Mathematics",
  "Science",
  "Logic and Abstract Reasoning",
  "General Knowledge",
  "Mechanical Reasoning",
];

function textOrFallback(value: string | undefined, fallback: string) {
  return value && value.trim() ? value : fallback;
}

function subjectExportValue(
  score: number | undefined,
  answers: string,
  detected: number | undefined,
  total: number,
) {
  if (typeof score === "number") return score;
  const fallbackDetected = answers && answers.trim() ? Math.min(total, answers.trim().length) : 0;
  return `${detected ?? fallbackDetected}/${total} detected`;
}

export function resultToExcelRow(result: ScanResult) {
  return [
    textOrFallback(result.applicationNumber, "Blank"),
    textOrFallback(result.lrn, "Blank"),
    textOrFallback(result.surname, "No name"),
    textOrFallback(result.name, "No name"),
    textOrFallback(result.middleName, "No name"),
    textOrFallback(result.examDate, "Blank"),
    ...subjectScoreConfig.map((subject) =>
      subjectExportValue(
        result[subject.scoreKey] as number | undefined,
        result[subject.answerKey] as string,
        result[subject.detectedKey] as number | undefined,
        subject.total,
      ),
    ),
  ];
}

export function scoreSubjectAnswers(
  detectedAnswers: string,
  correctAnswers: Array<string | undefined>,
  total: number,
) {
  let score = 0;

  for (let index = 0; index < total; index += 1) {
    const detected = detectedAnswers[index]?.toUpperCase() || "";
    const correct = correctAnswers[index]?.toUpperCase() || "";
    if (correct && detected === correct) score += 1;
  }

  return score;
}

export function scoreResultWithAnswerKey(result: ScanResult, answerKey: AnswerKeyState) {
  const scoredResult = { ...result };

  for (const subject of subjectScoreConfig) {
    const answers = String(result[subject.answerKey] || "");
    const correctAnswers = answerKey[subject.subject] || [];
    Object.assign(scoredResult, {
      [subject.scoreKey]: scoreSubjectAnswers(answers, correctAnswers, subject.total),
    });
  }

  return scoredResult;
}

export const seedResults: ScanResult[] = [];

const STORE_KEY = "minsu-scan-results";
const UPLOAD_LOG_STORE_KEY = "minsu-upload-logs";
export const UPLOAD_LOG_EVENT = "minsu-upload-logs-updated";

export type UploadLog = {
  id: string;
  file: string;
  date: string;
  user: string;
  result: string;
  status: "Ready" | "Scanning" | "Completed" | "Review" | "Failed";
  size?: number;
};

function isLegacySampleResult(result: ScanResult) {
  return result.applicationNumber === "APP-2026-047281" || result.fileName === "seed-kenji-mercurio-yonaha";
}

export function readStoredResults(): ScanResult[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    const results = raw ? (JSON.parse(raw) as ScanResult[]) : [];
    const filtered = results.filter((result) => !isLegacySampleResult(result));
    if (filtered.length !== results.length) {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(filtered));
    }
    return filtered;
  } catch {
    return [];
  }
}

export function storeResults(results: ScanResult[]) {
  window.localStorage.setItem(STORE_KEY, JSON.stringify(results.filter((result) => !isLegacySampleResult(result))));
  window.dispatchEvent(new Event("minsu-scan-results-updated"));
}

export function readStoredUploadLogs(): UploadLog[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(UPLOAD_LOG_STORE_KEY);
    return raw ? (JSON.parse(raw) as UploadLog[]) : [];
  } catch {
    return [];
  }
}

export function storeUploadLogs(logs: UploadLog[]) {
  window.localStorage.setItem(UPLOAD_LOG_STORE_KEY, JSON.stringify(logs.slice(0, 100)));
  window.dispatchEvent(new Event(UPLOAD_LOG_EVENT));
}

export function upsertUploadLogs(logs: UploadLog[]) {
  const existing = readStoredUploadLogs();
  const nextById = new Map(existing.map((log) => [log.id, log]));

  for (const log of logs) {
    nextById.set(log.id, { ...nextById.get(log.id), ...log });
  }

  storeUploadLogs(
    Array.from(nextById.values()).sort((left, right) => Date.parse(right.date) - Date.parse(left.date)),
  );
}
