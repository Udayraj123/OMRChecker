import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, rm, writeFile, copyFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import {
  defaultAnswerKeyState,
  scoreResultWithAnswerKey,
  type AnswerKeyState,
  type ScanResult,
} from "@/components/scan-store";
import { saveScanResults } from "@/lib/db";

export const runtime = "nodejs";

const repoRoot = process.env.OMR_REPO_ROOT || "";

function parseCsvLine(line: string) {
  const values: string[] = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"' && quoted && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
}

function buildSubject(values: Record<string, string>, prefix: string, count: number) {
  return Array.from({ length: count }, (_, index) => values[`${prefix}${index + 1}`] || "").join("");
}

function countDetected(values: Record<string, string>, prefix: string, count: number) {
  return Array.from({ length: count }, (_, index) => values[`${prefix}${index + 1}`] || "").filter(Boolean).length;
}

function safeBaseName(fileName: string) {
  return path.basename(fileName, path.extname(fileName)).replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "") || "scan";
}

function isSupportedExtension(extension: string) {
  return [".pdf", ".jpg", ".jpeg", ".png"].includes(extension.toLowerCase());
}

type MetadataResult = {
  applicationNumber?: string;
  lrn?: string;
  examDate?: string;
};

type AnswerResponse = Record<string, string>;

function parseAnswerKey(value: FormDataEntryValue | null) {
  if (typeof value !== "string" || !value.trim()) return null;

  try {
    const payload = JSON.parse(value) as { answers?: AnswerKeyState };
    return payload.answers || null;
  } catch {
    return null;
  }
}

function buildSubjectFromResponses(values: AnswerResponse, prefix: string, count: number) {
  return Array.from({ length: count }, (_, index) => values[`${prefix}${index + 1}`] || "").join("");
}

function countResponses(values: AnswerResponse, prefix: string, count: number) {
  return Array.from({ length: count }, (_, index) => values[`${prefix}${index + 1}`] || "").filter(Boolean).length;
}

function applyMinSuAnswers(result: ScanResult, answers: AnswerResponse | undefined) {
  if (!answers) return result;

  return {
    ...result,
    languageProficiency: buildSubjectFromResponses(answers, "L", 35),
    mathematics: buildSubjectFromResponses(answers, "M", 35),
    science: buildSubjectFromResponses(answers, "S", 35),
    logicAndAbstractReasoning: buildSubjectFromResponses(answers, "LA", 20),
    generalKnowledge: buildSubjectFromResponses(answers, "G", 35),
    mechanicalReasoning: buildSubjectFromResponses(answers, "ME", 20),
    languageProficiencyDetected: countResponses(answers, "L", 35),
    mathematicsDetected: countResponses(answers, "M", 35),
    scienceDetected: countResponses(answers, "S", 35),
    logicAndAbstractReasoningDetected: countResponses(answers, "LA", 20),
    generalKnowledgeDetected: countResponses(answers, "G", 35),
    mechanicalReasoningDetected: countResponses(answers, "ME", 20),
  };
}

function toResult(
  headers: string[],
  row: string[],
  sourceByGeneratedFile: Map<string, string>,
  metadataByGeneratedFile: Map<string, MetadataResult>,
): ScanResult {
  const values = Object.fromEntries(headers.map((header, index) => [header, row[index] || ""]));
  const fileName = values.file_id || values.file_name || "";
  const metadata = metadataByGeneratedFile.get(fileName) || {};

  return {
    applicationNumber: metadata.applicationNumber || values.ApplicationNumber || values.application_number || "",
    lrn: metadata.lrn || values.LRN || values.lrn || "",
    surname: values.Surname || values.surname || "",
    name: values.Name || values.name || "",
    middleName: values.MiddleName || values.middle_name || "",
    examDate: metadata.examDate || values.Date || values.ExamDate || values.exam_date || "",
    sourceFileName: sourceByGeneratedFile.get(fileName) || fileName,
    languageProficiency: buildSubject(values, "L", 35),
    mathematics: buildSubject(values, "M", 35),
    science: buildSubject(values, "S", 35),
    logicAndAbstractReasoning: buildSubject(values, "LA", 20),
    generalKnowledge: buildSubject(values, "G", 35),
    mechanicalReasoning: buildSubject(values, "ME", 20),
    languageProficiencyDetected: countDetected(values, "L", 35),
    mathematicsDetected: countDetected(values, "M", 35),
    scienceDetected: countDetected(values, "S", 35),
    logicAndAbstractReasoningDetected: countDetected(values, "LA", 20),
    generalKnowledgeDetected: countDetected(values, "G", 35),
    mechanicalReasoningDetected: countDetected(values, "ME", 20),
    fileName,
    checkedImagePath: values.output_path || "",
  };
}

function extractMetadata(imagePath: string) {
  return new Promise<MetadataResult>((resolve) => {
    const pythonCommand = process.env.OMR_PYTHON || "python";
    const scriptPath = path.join(process.cwd(), "scripts", "extract_metadata.py");
    const child = spawn(pythonCommand, [scriptPath, imagePath], {
      cwd: repoRoot,
      shell: false,
    });

    let stdout = "";
    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    child.on("close", () => {
      try {
        resolve(JSON.parse(stdout.trim()) as MetadataResult);
      } catch {
        resolve({});
      }
    });
    child.on("error", () => resolve({}));
  });
}

function readMinSuAnswers(imagePath: string) {
  return new Promise<AnswerResponse | undefined>((resolve) => {
    const pythonCommand = process.env.OMR_PYTHON || "python";
    const scriptPath = path.join(process.cwd(), "scripts", "read_minsu_answers.py");
    const templatePath = path.join(repoRoot, "inputs", "template.json");
    const referencePath = path.join(repoRoot, "inputs", "image.jpg");
    const child = spawn(pythonCommand, [scriptPath, imagePath, templatePath, referencePath], {
      cwd: repoRoot,
      shell: false,
    });

    let stdout = "";
    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    child.on("close", () => {
      try {
        resolve(JSON.parse(stdout.trim()) as AnswerResponse);
      } catch {
        resolve(undefined);
      }
    });
    child.on("error", () => resolve(undefined));
  });
}

function runPython(inputDir: string, outputDir: string) {
  return new Promise<void>((resolve, reject) => {
    const pythonCommand = process.env.OMR_PYTHON || "python";
    const child = spawn(pythonCommand, ["main.py", "--inputDir", inputDir, "--outputDir", outputDir], {
      cwd: repoRoot,
      shell: true,
    });

    let output = "";
    child.stdout.on("data", (data) => {
      output += data.toString();
    });
    child.stderr.on("data", (data) => {
      output += data.toString();
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(output || `OMRChecker exited with code ${code}`));
      }
    });
  });
}

function renderPdfToImages(pdfPath: string, inputDir: string, baseName: string) {
  return new Promise<string[]>((resolve, reject) => {
    const pythonCommand = process.env.OMR_PYTHON || "python";
    const script = `
import fitz
import json
import pathlib
import sys

pdf_path = sys.argv[1]
output_dir = pathlib.Path(sys.argv[2])
base_name = sys.argv[3]
document = fitz.open(pdf_path)
generated = []

for index, page in enumerate(document):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    name = f"{base_name}-page-{index + 1}.png"
    pixmap.save(str(output_dir / name))
    generated.append(name)

print(json.dumps(generated))
`;
    const child = spawn(pythonCommand, ["-c", script, pdfPath, inputDir, baseName], {
      cwd: repoRoot,
      shell: false,
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `PDF conversion exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout.trim()) as string[]);
      } catch {
        reject(new Error("PDF conversion did not return generated image names."));
      }
    });
  });
}

export async function POST(request: Request) {
  if (!repoRoot) {
    return NextResponse.json({ error: "OMR_REPO_ROOT is not configured." }, { status: 500 });
  }

  const formData = await request.formData();
  const files = formData.getAll("files").filter((file): file is File => file instanceof File);
  const answerKey = parseAnswerKey(formData.get("answerKey")) || defaultAnswerKeyState;

  if (files.length === 0) {
    return NextResponse.json({ error: "No files uploaded." }, { status: 400 });
  }

  const runId = `${Date.now()}`;
  const runRoot = path.join(process.cwd(), ".scan-runs", runId);
  const inputDir = path.join(runRoot, "inputs");
  const outputDir = path.join(runRoot, "outputs");
  const sourcePdfDir = path.join(runRoot, "source-pdfs");
  const sourceByGeneratedFile = new Map<string, string>();
  const metadataByGeneratedFile = new Map<string, MetadataResult>();
  const answersByGeneratedFile = new Map<string, AnswerResponse>();

  await mkdir(inputDir, { recursive: true });
  await mkdir(outputDir, { recursive: true });
  await mkdir(sourcePdfDir, { recursive: true });
  await copyFile(path.join(repoRoot, "inputs", "template.json"), path.join(inputDir, "template.json"));
  await copyFile(path.join(repoRoot, "inputs", "config.json"), path.join(inputDir, "config.json"));

  try {
    for (const file of files) {
      const extension = path.extname(file.name).toLowerCase();
      if (!isSupportedExtension(extension)) continue;
      const bytes = Buffer.from(await file.arrayBuffer());
      if (extension === ".pdf") {
        const pdfPath = path.join(sourcePdfDir, file.name);
        await writeFile(pdfPath, bytes);
        const generatedImages = await renderPdfToImages(pdfPath, inputDir, safeBaseName(file.name));
        for (const generatedImage of generatedImages) {
          const generatedImagePath = path.join(inputDir, generatedImage);
          sourceByGeneratedFile.set(generatedImage, file.name);
          metadataByGeneratedFile.set(generatedImage, await extractMetadata(generatedImagePath));
          const minsuAnswers = await readMinSuAnswers(generatedImagePath);
          if (minsuAnswers) answersByGeneratedFile.set(generatedImage, minsuAnswers);
        }
      } else {
        const imagePath = path.join(inputDir, file.name);
        await writeFile(imagePath, bytes);
        sourceByGeneratedFile.set(file.name, file.name);
        metadataByGeneratedFile.set(file.name, await extractMetadata(imagePath));
        const minsuAnswers = await readMinSuAnswers(imagePath);
        if (minsuAnswers) answersByGeneratedFile.set(file.name, minsuAnswers);
      }
    }

    if (sourceByGeneratedFile.size === 0) {
      await rm(runRoot, { recursive: true, force: true });
      return NextResponse.json({ error: "Only PDF, JPEG, and PNG files are supported." }, { status: 400 });
    }

    await runPython(inputDir, outputDir);
    const resultsDir = path.join(outputDir, "Results");
    const resultFiles = (await readdir(resultsDir)).filter((file) => file.endsWith(".csv"));
    const latest = resultFiles[0];

    if (!latest) {
      return NextResponse.json({ results: [] });
    }

    const csv = await readFile(path.join(resultsDir, latest), "utf-8");
    const [headerLine, ...dataLines] = csv.trim().split(/\r?\n/);
    const headers = parseCsvLine(headerLine);
    const detectedResults = dataLines
      .filter(Boolean)
      .map((line) => {
        const result = toResult(headers, parseCsvLine(line), sourceByGeneratedFile, metadataByGeneratedFile);
        return applyMinSuAnswers(result, answersByGeneratedFile.get(result.fileName));
      });
    const results = answerKey
      ? detectedResults.map((result) => scoreResultWithAnswerKey(result, answerKey))
      : detectedResults;
    let dbWarning = "";

    try {
      await saveScanResults(results);
    } catch (error) {
      dbWarning = error instanceof Error ? error.message : "Unable to save scan results to MySQL.";
    }

    return NextResponse.json({ results, dbWarning });
  } catch (error) {
    await rm(runRoot, { recursive: true, force: true });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Scan failed." },
      { status: 500 },
    );
  }
}
