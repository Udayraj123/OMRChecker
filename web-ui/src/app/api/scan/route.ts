import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
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

function safeBaseName(fileName: string) {
  return path.basename(fileName, path.extname(fileName)).replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "") || "scan";
}

function isSupportedExtension(extension: string) {
  return [".pdf", ".jpg", ".jpeg", ".png"].includes(extension.toLowerCase());
}

type FastScanEntry = {
  imagePath?: string;
  pdfPath?: string;
  fileName: string;
  sourceFileName: string;
};

function parseAnswerKey(value: FormDataEntryValue | null) {
  if (typeof value !== "string" || !value.trim()) return null;

  try {
    const payload = JSON.parse(value) as { answers?: AnswerKeyState };
    return payload.answers || null;
  } catch {
    return null;
  }
}

function runFastScan(manifestPath: string) {
  return new Promise<ScanResult[]>((resolve, reject) => {
    const pythonCommand = process.env.OMR_PYTHON || "python";
    const scriptPath = path.join(process.cwd(), "scripts", "fast_scan_minsu.py");
    const templatePath = path.join(repoRoot, "inputs", "template.json");
    const referencePath = path.join(repoRoot, "inputs", "image.jpg");
    const child = spawn(pythonCommand, [scriptPath, manifestPath, templatePath, referencePath], {
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
        reject(new Error(stderr || `Fast scanner exited with code ${code}`));
        return;
      }

      try {
        const payload = JSON.parse(stdout.trim()) as { results?: ScanResult[] };
        resolve(payload.results || []);
      } catch {
        reject(new Error(stderr || "Fast scanner did not return valid JSON."));
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
  const sourcePdfDir = path.join(runRoot, "source-pdfs");
  const fastScanEntries: FastScanEntry[] = [];

  await mkdir(inputDir, { recursive: true });
  await mkdir(sourcePdfDir, { recursive: true });

  try {
    for (const file of files) {
      const extension = path.extname(file.name).toLowerCase();
      if (!isSupportedExtension(extension)) continue;
      const bytes = Buffer.from(await file.arrayBuffer());
      if (extension === ".pdf") {
        const pdfPath = path.join(sourcePdfDir, file.name);
        await writeFile(pdfPath, bytes);
        fastScanEntries.push({
          pdfPath,
          fileName: safeBaseName(file.name),
          sourceFileName: file.name,
        });
      } else {
        const imagePath = path.join(inputDir, file.name);
        await writeFile(imagePath, bytes);
        fastScanEntries.push({
          imagePath,
          fileName: file.name,
          sourceFileName: file.name,
        });
      }
    }

    if (fastScanEntries.length === 0) {
      await rm(runRoot, { recursive: true, force: true });
      return NextResponse.json({ error: "Only PDF, JPEG, and PNG files are supported." }, { status: 400 });
    }

    const manifestPath = path.join(runRoot, "fast-scan-manifest.json");
    await writeFile(manifestPath, JSON.stringify({ images: fastScanEntries }), "utf-8");
    const detectedResults = await runFastScan(manifestPath);
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
