"use client";

import { motion } from "framer-motion";
import {
  ArrowDownToLine,
  BadgeCheck,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Download,
  Eye,
  FileImage,
  FileSpreadsheet,
  Filter,
  KeyRound,
  Loader2,
  Plus,
  RefreshCw,
  ScanLine,
  Search,
  Settings2,
  Sparkles,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell, Card, StatusBadge } from "./app-shell";
import { answerKeys, itemAnalysis, metrics, recentScans, students } from "./data";
import {
  ANSWER_KEY_STORE,
  UPLOAD_LOG_EVENT,
  answerChoices,
  defaultAnswerKeyState,
  excelHeaders,
  readStoredUploadLogs,
  readStoredResults,
  resultToExcelRow,
  seedResults,
  subjectScoreConfig,
  storeResults,
  upsertUploadLogs,
  type AnswerChoice,
  type AnswerKeyState,
  type ScanResult,
  type UploadLog,
} from "./scan-store";

function ActionButton({
  children,
  variant = "primary",
  className = "",
  ...props
}: {
  children: React.ReactNode;
  variant?: "primary" | "secondary";
  className?: string;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${
        variant === "primary"
          ? "bg-slate-900 text-white shadow-lg shadow-slate-900/15 hover:-translate-y-0.5 hover:bg-slate-800 dark:bg-sky-300 dark:text-slate-950"
          : "border border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50 dark:border-white/10 dark:bg-white/5 dark:text-slate-200 dark:hover:bg-white/10"
      } ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

function escapeHtml(value: string | number) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function downloadXls(filename: string, rows: Array<Array<string | number>>) {
  const tableRows = rows
    .map((row, rowIndex) => {
      const tag = rowIndex === 0 ? "th" : "td";
      return `<tr>${row.map((cell) => `<${tag}>${escapeHtml(cell)}</${tag}>`).join("")}</tr>`;
    })
    .join("");
  const html = `<!doctype html><html><head><meta charset="utf-8" /></head><body><table>${tableRows}</table></body></html>`;
  const blob = new Blob([html], { type: "application/vnd.ms-excel;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function displayBlank(value: string | number | undefined, fallback = "Blank") {
  if (value === undefined || value === null || String(value).trim() === "") return fallback;
  return String(value);
}

function subjectSummary(
  result: ScanResult | undefined,
  scoreKey: keyof ScanResult,
  answerKey: keyof ScanResult,
  detectedKey: keyof ScanResult,
  total: number,
) {
  if (!result) return "Pending scan";
  const score = result[scoreKey];
  if (typeof score === "number") return `${score}/${total}`;
  const detected = result[detectedKey];
  if (typeof detected === "number") return `${detected}/${total} detected`;
  const answers = result[answerKey];
  return `${String(answers || "").length}/${total} detected`;
}

function shortAnswers(value: string) {
  if (!value) return "Blank";
  return value.length > 18 ? `${value.slice(0, 18)}...` : value;
}

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
const MAX_UPLOAD_LABEL = "4 MB";

function fileExtension(file: File) {
  return file.name.split(".").pop()?.toLowerCase() || "";
}

function isSupportedUpload(file: File) {
  return ["pdf", "jpg", "jpeg", "png"].includes(fileExtension(file));
}

function isWithinUploadLimit(file: File) {
  return file.size <= MAX_UPLOAD_BYTES;
}

function uploadLogId(file: File) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function fileSizeLabel(file: File) {
  return `${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

function formatHistoryDate(value: string | undefined) {
  if (!value) return "Today";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function resultHistoryLog(result: ScanResult, index: number): UploadLog {
  return {
    id: String(result.id ?? `result-${result.fileName || result.sourceFileName || index}-${index}`),
    file: result.sourceFileName || result.fileName || "scan-result",
    date: result.examDate || new Date().toISOString(),
    user: "Registrar",
    result: "1 row",
    status: "Completed",
  };
}

async function readJsonPayload<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!text.trim()) return {} as T;

  try {
    return JSON.parse(text) as T;
  } catch {
    const message = text.startsWith("Request Entity Too Large")
      ? `File is too large for Vercel upload. Maximum allowed size is ${MAX_UPLOAD_LABEL}.`
      : text.slice(0, 160);
    return { error: message } as T;
  }
}

function isPdfFile(file: File | undefined) {
  return !!file && fileExtension(file) === "pdf";
}

function resultKey(row: ScanResult, index: number) {
  return String(row.id ?? `${row.fileName || "scan"}-${row.applicationNumber || row.lrn || index}-${index}`);
}

function normalizedDuplicateValue(value: string | undefined) {
  const normalized = (value || "").trim().toLowerCase();
  if (!normalized || normalized === "blank" || normalized === "no name") return "";
  return normalized;
}

function fullNameDuplicateValue(row: ScanResult) {
  const parts = [row.surname, row.name, row.middleName].map(normalizedDuplicateValue).filter(Boolean);
  return parts.length >= 2 ? parts.join(" ") : "";
}

function buildDuplicateReasons(rows: ScanResult[], keys: string[]) {
  const reasons: Record<string, string[]> = {};
  const fields = [
    { label: "Application number", value: (row: ScanResult) => row.applicationNumber },
    { label: "LRN", value: (row: ScanResult) => row.lrn },
    { label: "Full name", value: fullNameDuplicateValue },
    { label: "Source file", value: (row: ScanResult) => row.sourceFileName || row.fileName },
  ];

  fields.forEach((field) => {
    const groups = new Map<string, string[]>();

    rows.forEach((row, index) => {
      const value = normalizedDuplicateValue(field.value(row));
      if (!value) return;
      groups.set(value, [...(groups.get(value) || []), keys[index]]);
    });

    groups.forEach((groupKeys) => {
      if (groupKeys.length < 2) return;
      groupKeys.forEach((key) => {
        reasons[key] = [...(reasons[key] || []), field.label];
      });
    });
  });

  return reasons;
}

function MetricCards() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        return (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <Card className="p-5">
              <div className="flex items-center justify-between">
                <div className="rounded-lg bg-sky-50 p-2.5 text-sky-700 dark:bg-sky-300/10 dark:text-sky-200">
                  <Icon size={22} />
                </div>
                <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-400/15 dark:text-emerald-200">
                  {metric.delta}
                </span>
              </div>
              <p className="mt-5 text-sm text-slate-500 dark:text-slate-400">{metric.label}</p>
              <p className="mt-1 text-3xl font-semibold tracking-tight">{metric.value}</p>
            </Card>
          </motion.div>
        );
      })}
    </div>
  );
}

export function DashboardPage() {
  return (
    <AppShell title="AI Scanner Dashboard">
      <div className="space-y-6">
        <section className="grid gap-6 xl:grid-cols-[1.45fr_0.9fr]">
          <Card className="ai-grid overflow-hidden p-6 sm:p-8">
            <div className="flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
              <div className="max-w-2xl">
                <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-sm font-semibold text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-200">
                  <Sparkles size={16} />
                  Python OMRChecker pipeline connected
                </div>
                <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
                  Scan entrance exam sheets with AI-assisted validation.
                </h2>
                <p className="mt-4 max-w-xl text-slate-600 dark:text-slate-300">
                  Upload OMR sheets, read MinSU answer bubbles, review LRN and barcode metadata, and export clean CSV reports for admissions.
                </p>
                <div className="mt-6 flex flex-wrap gap-3">
                  <ActionButton>
                    <UploadCloud size={18} />
                    Upload sheets
                  </ActionButton>
                  <ActionButton variant="secondary">
                    <Settings2 size={18} />
                    Configure template
                  </ActionButton>
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white/85 p-4 dark:border-white/10 dark:bg-slate-950/45">
                <div className="mb-4 flex items-center justify-between">
                  <p className="font-semibold">Live processing</p>
                  <span className="text-sm text-emerald-600 dark:text-emerald-300">98.4% confidence</span>
                </div>
                <div className="space-y-3">
                  {["Marker alignment", "Bubble intensity", "LRN extraction", "CSV writer"].map((step, index) => (
                    <div key={step} className="flex items-center gap-3 rounded-lg bg-slate-50 p-3 dark:bg-white/5">
                      <CheckCircle2 className="text-emerald-600" size={18} />
                      <div className="flex-1">
                        <p className="text-sm font-medium">{step}</p>
                        <div className="mt-2 h-1.5 rounded-full bg-slate-200 dark:bg-slate-700">
                          <motion.div
                            className="h-full rounded-full bg-emerald-500"
                            initial={{ width: 0 }}
                            animate={{ width: `${86 + index * 3}%` }}
                            transition={{ duration: 1, delay: index * 0.12 }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Today</p>
                <h3 className="text-xl font-semibold">Scan queue</h3>
              </div>
              <div className="rounded-lg bg-slate-900 p-2 text-white dark:bg-sky-300 dark:text-slate-950">
                <ScanLine size={20} />
              </div>
            </div>
            <div className="mt-6 space-y-4">
              {recentScans.slice(0, 3).map((scan) => (
                <div key={scan.batch} className="flex items-center justify-between gap-4 rounded-lg border border-slate-100 p-3 dark:border-white/10">
                  <div>
                    <p className="font-medium">{scan.batch}</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">{scan.sheets} sheets · {scan.time}</p>
                  </div>
                  <StatusBadge status={scan.status} />
                </div>
              ))}
            </div>
          </Card>
        </section>

        <MetricCards />
        <RecentScans />
      </div>
    </AppShell>
  );
}

function RecentScans() {
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-slate-200 p-5 dark:border-white/10 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-lg font-semibold">Recent scans</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">Latest batches from the entrance examination scanner.</p>
        </div>
        <ActionButton variant="secondary">
          View all
          <ChevronRight size={17} />
        </ActionButton>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase text-slate-500 dark:bg-white/5 dark:text-slate-400">
            <tr>
              <th className="px-5 py-3">Batch</th>
              <th className="px-5 py-3">Sheets</th>
              <th className="px-5 py-3">Status</th>
              <th className="px-5 py-3">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-white/10">
            {recentScans.map((scan) => (
              <tr key={scan.batch} className="transition hover:bg-slate-50/80 dark:hover:bg-white/5">
                <td className="px-5 py-4 font-medium">{scan.batch}</td>
                <td className="px-5 py-4">{scan.sheets}</td>
                <td className="px-5 py-4"><StatusBadge status={scan.status} /></td>
                <td className="px-5 py-4 text-slate-500 dark:text-slate-400">{scan.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export function UploadScannerPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState("");
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [viewerIndex, setViewerIndex] = useState(0);
  const viewerFile = selectedFiles[viewerIndex];
  const viewerUrl = useMemo(() => (viewerFile ? URL.createObjectURL(viewerFile) : ""), [viewerFile]);
  const currentScanResult =
    scanResults.find((result) => result.sourceFileName === viewerFile?.name || result.fileName === viewerFile?.name) ||
    scanResults[viewerIndex] ||
    scanResults[0];
  const metadataRows = [
    ["Source file", displayBlank(currentScanResult?.sourceFileName || viewerFile?.name)],
    ["Scan status", currentScanResult ? "Scanned" : viewerFile ? "Ready to scan" : "Blank"],
    ["Application number", displayBlank(currentScanResult?.applicationNumber)],
    ["LRN", displayBlank(currentScanResult?.lrn)],
    ["Surname", displayBlank(currentScanResult?.surname, "No name")],
    ["Name", displayBlank(currentScanResult?.name, "No name")],
    ["Middle name", displayBlank(currentScanResult?.middleName, "No name")],
    ["Date", displayBlank(currentScanResult?.examDate)],
  ];
  const subjectRows = subjectScoreConfig.map((subject) => [
    subject.subject,
    subjectSummary(currentScanResult, subject.scoreKey, subject.answerKey, subject.detectedKey, subject.total),
  ]);
  const uploads: Array<{ name: string; progress: number; type: string; count: string }> = [];

  useEffect(() => {
    return () => {
      if (viewerUrl) URL.revokeObjectURL(viewerUrl);
    };
  }, [viewerUrl]);

  const startScan = async (filesToScan = selectedFiles) => {
    if (filesToScan.length === 0) {
      setScanError("Pumili muna ng PDF, JPEG, or PNG answer sheet bago mag-scan.");
      return;
    }

    const validFiles = filesToScan.filter(isSupportedUpload);
    if (validFiles.length === 0) {
      setScanError("PDF, JPEG, at PNG lang ang supported.");
      return;
    }

    const oversizedFiles = validFiles.filter((file) => !isWithinUploadLimit(file));
    if (oversizedFiles.length > 0) {
      setScanError(
        `${oversizedFiles[0].name} is ${fileSizeLabel(oversizedFiles[0])}. Maximum upload size on Vercel is ${MAX_UPLOAD_LABEL}. Compress or split the PDF before scanning.`,
      );
      return;
    }

    setIsScanning(true);
    setScanError("");
    upsertUploadLogs(
      validFiles.map((file) => ({
        id: uploadLogId(file),
        file: file.name,
        date: new Date().toISOString(),
        user: "Registrar",
        result: "Processing",
        status: "Scanning",
        size: file.size,
      })),
    );

    const formData = new FormData();
    validFiles.forEach((file) => formData.append("files", file));
    const savedAnswerKey = window.localStorage.getItem(ANSWER_KEY_STORE);
    if (savedAnswerKey) {
      formData.append("answerKey", savedAnswerKey);
    }

    try {
      const response = await fetch("/api/scan", {
        method: "POST",
        body: formData,
      });
      const payload = await readJsonPayload<{ results?: ScanResult[]; error?: string; dbWarning?: string }>(response);

      if (!response.ok) {
        throw new Error(payload.error || "Scan failed.");
      }

      const nextResults = payload.results || [];
      setScanResults(nextResults);
      storeResults([...readStoredResults(), ...nextResults]);
      upsertUploadLogs(
        validFiles.map((file, index) => {
          const fileResults = nextResults.filter((result) => result.sourceFileName === file.name || result.fileName === file.name);
          const rowCount = fileResults.length || (nextResults[index] ? 1 : 0);
          return {
            id: uploadLogId(file),
            file: file.name,
            date: new Date().toISOString(),
            user: "Registrar",
            result: rowCount === 1 ? "1 row" : `${rowCount} rows`,
            status: rowCount > 0 ? "Completed" : "Review",
            size: file.size,
          };
        }),
      );
      if (payload.dbWarning) {
        setScanError(`Saved locally. MySQL not loaded: ${payload.dbWarning}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Hindi natuloy ang scan.";
      upsertUploadLogs(
        validFiles.map((file) => ({
          id: uploadLogId(file),
          file: file.name,
          date: new Date().toISOString(),
          user: "Registrar",
          result: message.slice(0, 80),
          status: "Failed",
          size: file.size,
        })),
      );
      setScanError(message);
    } finally {
      setIsScanning(false);
    }
  };

  const addFiles = (files: FileList | null) => {
    if (!files) return;
    const incomingFiles = Array.from(files);
    const nextFiles = incomingFiles.filter(isSupportedUpload);

    if (nextFiles.length === 0) {
      setSelectedFiles([]);
      setScanError("PDF, JPEG, at PNG lang ang supported.");
      setScanResults([]);
      return;
    }

    const oversizedFiles = nextFiles.filter((file) => !isWithinUploadLimit(file));
    if (oversizedFiles.length > 0) {
      setSelectedFiles(nextFiles);
      setViewerIndex(0);
      setScanResults([]);
      upsertUploadLogs(
        oversizedFiles.map((file) => ({
          id: uploadLogId(file),
          file: file.name,
          date: new Date().toISOString(),
          user: "Registrar",
          result: `${fileSizeLabel(file)} file`,
          status: "Failed",
          size: file.size,
        })),
      );
      setScanError(
        `${oversizedFiles[0].name} is ${fileSizeLabel(oversizedFiles[0])}. Maximum upload size on Vercel is ${MAX_UPLOAD_LABEL}. Compress or split the PDF before scanning.`,
      );
      return;
    }

    setSelectedFiles(nextFiles);
    setViewerIndex(0);
    setScanError(nextFiles.length < incomingFiles.length ? "May hindi sinamang file. PDF, JPEG, at PNG lang ang supported." : "");
    setScanResults([]);
    upsertUploadLogs(
      nextFiles.map((file) => ({
        id: uploadLogId(file),
        file: file.name,
        date: new Date().toISOString(),
        user: "Registrar",
        result: "Ready to scan",
        status: "Ready",
        size: file.size,
      })),
    );
  };

  return (
    <AppShell title="Upload Scanner" eyebrow="Batch processing">
      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card className="p-5 sm:p-8">
          <div
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              addFiles(event.dataTransfer.files);
            }}
            className="rounded-lg border-2 border-dashed border-sky-200 bg-sky-50/70 p-8 text-center transition hover:border-emerald-300 hover:bg-emerald-50/60 dark:border-sky-300/20 dark:bg-sky-300/5 dark:hover:bg-emerald-400/10"
          >
            <motion.div
              animate={{ y: [0, -8, 0] }}
              transition={{ repeat: Infinity, duration: 2.4 }}
              className="mx-auto flex h-20 w-20 items-center justify-center rounded-lg bg-white text-sky-700 shadow-xl shadow-sky-900/10 dark:bg-white/10 dark:text-sky-200"
            >
              <UploadCloud size={38} />
            </motion.div>
            <h2 className="mt-6 text-2xl font-semibold">Drop 100+ OMR sheets here</h2>
            <p className="mx-auto mt-2 max-w-xl text-slate-600 dark:text-slate-300">
              Supports PDF, JPEG, and PNG batches. PDF pages are converted before Python OMRChecker processing.
            </p>
            <input
              ref={inputRef}
              className="hidden"
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/jpeg,image/png"
              onChange={(event) => addFiles(event.target.files)}
            />
            <div className="mt-6 flex flex-wrap justify-center gap-3">
              <ActionButton onClick={() => inputRef.current?.click()}>
                <FileImage size={18} />
                Choose files
              </ActionButton>
              <ActionButton variant="secondary" onClick={() => startScan()} disabled={isScanning}>
                {isScanning ? <Loader2 className="animate-spin" size={18} /> : <ScanLine size={18} />}
                {isScanning ? "Scanning..." : "Scan now"}
              </ActionButton>
            </div>
            {selectedFiles.length > 0 && (
              <p className="mt-4 text-sm font-semibold text-emerald-700 dark:text-emerald-300">
                {selectedFiles.length} file{selectedFiles.length === 1 ? "" : "s"} ready for scanning
              </p>
            )}
            {scanError && (
              <p className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700 dark:border-rose-400/20 dark:bg-rose-400/10 dark:text-rose-200">
                {scanError}
              </p>
            )}
          </div>

          <div className="mt-8 space-y-4">
            {selectedFiles.map((file, index) => (
              <button
                type="button"
                key={`${file.name}-${file.lastModified}`}
                onClick={() => setViewerIndex(index)}
                className={`w-full rounded-lg border p-4 text-left transition ${
                  index === viewerIndex
                    ? "border-emerald-300 bg-emerald-50/70 dark:border-emerald-400/30 dark:bg-emerald-400/10"
                    : "border-emerald-200 bg-emerald-50/40 hover:border-emerald-300 dark:border-emerald-400/20 dark:bg-emerald-400/5"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="font-semibold">{file.name}</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Selected file · {fileSizeLabel(file)}
                    </p>
                  </div>
                  <span className={`text-sm font-semibold ${isWithinUploadLimit(file) ? "text-emerald-700 dark:text-emerald-200" : "text-rose-700 dark:text-rose-200"}`}>
                    {isWithinUploadLimit(file) ? "Ready" : "Too large"}
                  </span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-emerald-100 dark:bg-emerald-950/60">
                  <motion.div
                    className="h-full rounded-full bg-emerald-500"
                    initial={{ width: 0 }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </button>
            ))}
            {uploads.map((upload) => (
              <div key={upload.name} className="rounded-lg border border-slate-200 p-4 dark:border-white/10">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="font-semibold">{upload.name}</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">{upload.type} · {upload.count}</p>
                  </div>
                  <span className="text-sm font-semibold text-sky-700 dark:text-sky-200">{upload.progress}%</span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-slate-100 dark:bg-slate-800">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-sky-500 to-emerald-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${upload.progress}%` }}
                    transition={{ duration: 0.8 }}
                  />
                </div>
              </div>
            ))}
            {scanResults.length > 0 && (
              <div className="rounded-lg border border-sky-200 bg-sky-50 p-4 dark:border-sky-400/20 dark:bg-sky-400/10">
                <p className="font-semibold text-sky-900 dark:text-sky-100">
                  {scanResults.length} scanned result{scanResults.length === 1 ? "" : "s"} saved to Scan Results.
                </p>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                  The table and Excel download now use the actual OMRChecker CSV output.
                </p>
              </div>
            )}
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-emerald-50 p-2 text-emerald-700 dark:bg-emerald-400/15 dark:text-emerald-200">
              <Eye size={22} />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Answer sheet viewer</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Preview the selected scan before processing.</p>
            </div>
          </div>
          <div className="mt-6 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 dark:border-white/10 dark:bg-white/5">
            {viewerFile && viewerUrl && isPdfFile(viewerFile) ? (
              <iframe src={viewerUrl} title={viewerFile.name} className="h-[620px] w-full bg-white" />
            ) : viewerFile && viewerUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={viewerUrl} alt={viewerFile.name} className="max-h-[620px] w-full object-contain" />
            ) : (
              <div className="flex min-h-[360px] flex-col items-center justify-center p-8 text-center text-slate-500">
                <FileImage className="mb-3" size={38} />
                <p className="font-semibold">No answer sheet selected</p>
                <p className="mt-1 text-sm">Choose a PDF, JPEG, or PNG sheet to preview it here.</p>
              </div>
            )}
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-white/10 dark:bg-white/5">
              <p className="text-sm font-semibold uppercase text-slate-500 dark:text-slate-400">Detected student fields</p>
              <div className="mt-4 space-y-2">
                {metadataRows.map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between gap-4 rounded-lg bg-slate-50 px-3 py-2 text-sm dark:bg-slate-900/40">
                    <span className="text-slate-500 dark:text-slate-400">{label}</span>
                    <span className="max-w-[55%] truncate font-semibold">{value}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-white/10 dark:bg-white/5">
              <p className="text-sm font-semibold uppercase text-slate-500 dark:text-slate-400">Answers / subject scores</p>
              <div className="mt-4 space-y-2">
                {subjectRows.map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between gap-4 rounded-lg bg-slate-50 px-3 py-2 text-sm dark:bg-slate-900/40">
                    <span className="text-slate-500 dark:text-slate-400">{label}</span>
                    <span className="font-mono text-xs font-semibold">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="mt-8 space-y-5">
            {["Detect markers", "Align MinSU template", "Read answers", "Write CSV"].map((stage, index) => (
              <div key={stage} className="flex items-center gap-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 dark:bg-white/10">
                  {index === 2 ? <Loader2 className="animate-spin text-sky-600" size={20} /> : <BadgeCheck className="text-emerald-600" size={20} />}
                </div>
                <div className="flex-1">
                  <p className="font-medium">{stage}</p>
                  <div className="mt-2 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
                    <motion.div
                      className="h-full rounded-full bg-sky-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, 30 + index * 17)}%` }}
                      transition={{ duration: 1, delay: index * 0.1 }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

export function ResultsPage() {
  const [storedResults, setStoredResults] = useState<ScanResult[]>([]);
  const [loadError, setLoadError] = useState("");
  const [deleteError, setDeleteError] = useState("");
  const [loadedFromMySql, setLoadedFromMySql] = useState(false);
  const [filterMode, setFilterMode] = useState<"all" | "duplicates">("all");
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const displayResults = useMemo(
    () => (storedResults.length > 0 ? storedResults : loadedFromMySql ? [] : seedResults),
    [loadedFromMySql, storedResults],
  );
  const resultKeys = useMemo(
    () => displayResults.map(resultKey),
    [displayResults],
  );
  const duplicateReasonsByKey = useMemo(() => buildDuplicateReasons(displayResults, resultKeys), [displayResults, resultKeys]);
  const resultRows = displayResults.map((row, index) => ({
    row,
    key: resultKeys[index],
    duplicateReasons: duplicateReasonsByKey[resultKeys[index]] || [],
  }));
  const visibleRows = filterMode === "duplicates" ? resultRows.filter((entry) => entry.duplicateReasons.length > 0) : resultRows;
  const visibleKeys = visibleRows.map((entry) => entry.key);
  const duplicateCount = resultRows.filter((entry) => entry.duplicateReasons.length > 0).length;
  const validSelectedKeys = selectedKeys.filter((key) => resultKeys.includes(key));
  const selectedResults = displayResults.filter((_, index) => validSelectedKeys.includes(resultKeys[index]));
  const allSelected = visibleKeys.length > 0 && visibleKeys.every((key) => validSelectedKeys.includes(key));

  useEffect(() => {
    const refresh = async () => {
      try {
        const response = await fetch("/api/results");
        const payload = (await response.json()) as {
          results?: ScanResult[];
          error?: string;
          dbAvailable?: boolean;
          warning?: string;
        };

        if (!response.ok) {
          throw new Error(payload.error || "Unable to load MySQL results.");
        }

        setStoredResults(payload.results || []);
        setLoadedFromMySql(payload.dbAvailable !== false);
        setLoadError("");
      } catch (error) {
        setStoredResults(readStoredResults());
        setLoadedFromMySql(false);
        setLoadError(error instanceof Error ? error.message : "Unable to load MySQL results.");
      }
    };

    refresh();
    window.addEventListener("minsu-scan-results-updated", refresh);
    return () => window.removeEventListener("minsu-scan-results-updated", refresh);
  }, []);

  const downloadResults = (filename: string, rows: ScanResult[]) => {
    downloadXls(filename, [excelHeaders, ...rows.map(resultToExcelRow)]);
  };

  const downloadSelectedOrAll = () => {
    const rows = selectedResults.length > 0 ? selectedResults : visibleRows.map((entry) => entry.row);
    downloadResults("minsu-scan-results.xls", rows);
  };

  const toggleAll = () => {
    setSelectedKeys((current) => {
      if (allSelected) return current.filter((key) => !visibleKeys.includes(key));
      return Array.from(new Set([...current, ...visibleKeys]));
    });
  };

  const toggleRow = (key: string) => {
    setSelectedKeys((current) => (current.includes(key) ? current.filter((item) => item !== key) : [...current, key]));
  };

  const deleteResult = async (row: ScanResult, key: string) => {
    setDeleteError("");

    try {
      if (row.id) {
        const response = await fetch("/api/results", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: row.id }),
        });
        const payload = (await response.json()) as { error?: string };

        if (!response.ok) {
          throw new Error(payload.error || "Unable to delete scan result.");
        }
      }

      setStoredResults((current) => {
        const base = current.length > 0 ? current : displayResults;
        const next = base.filter((item, index) => resultKey(item, index) !== key);
        if (!row.id) storeResults(next);
        return next;
      });
      setSelectedKeys((current) => current.filter((item) => item !== key));
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : "Hindi natuloy ang soft delete.");
    }
  };

  return (
    <AppShell title="Scan Results" eyebrow="XLS export">
      <Card className="overflow-hidden">
        <div className="flex flex-col gap-4 border-b border-slate-200 p-5 dark:border-white/10 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h2 className="text-xl font-semibold">Validated answer sheets</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Select one row, multiple rows, or all rows, then download them into one Excel file.
            </p>
            {loadError && (
              <p className="mt-2 text-sm font-semibold text-amber-700 dark:text-amber-300">
                MySQL not loaded: {loadError}
              </p>
            )}
            {deleteError && (
              <p className="mt-2 text-sm font-semibold text-rose-700 dark:text-rose-300">
                Delete failed: {deleteError}
              </p>
            )}
            <p className="mt-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
              Duplicate detection: {duplicateCount} row{duplicateCount === 1 ? "" : "s"} flagged by Application Number, LRN, Full Name, or Source File.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <ActionButton
              variant={filterMode === "duplicates" ? "primary" : "secondary"}
              onClick={() => setFilterMode((current) => (current === "duplicates" ? "all" : "duplicates"))}
            >
              <Filter size={17} />
              {filterMode === "duplicates" ? "Show all" : "Duplicates only"}
            </ActionButton>
            <ActionButton variant="secondary" onClick={toggleAll}>
              <CheckCircle2 size={17} />
              {allSelected ? "Clear selection" : "Select all"}
            </ActionButton>
            <ActionButton variant="secondary" onClick={downloadSelectedOrAll}><FileSpreadsheet size={17} /> XLS</ActionButton>
            <ActionButton onClick={downloadSelectedOrAll}><ArrowDownToLine size={17} /> Download all</ActionButton>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1700px] text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500 dark:bg-white/5">
              <tr>
                {["Select", ...excelHeaders, "Duplicate", "Answer preview", "Download", "Delete"].map((head) => (
                  <th key={head} className="px-5 py-3">{head}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-white/10">
              {visibleRows.map(({ row, key, duplicateReasons }) => {
                const answerPreview = [
                  `L:${shortAnswers(row.languageProficiency)}`,
                  `M:${shortAnswers(row.mathematics)}`,
                  `S:${shortAnswers(row.science)}`,
                  `LA:${shortAnswers(row.logicAndAbstractReasoning)}`,
                  `G:${shortAnswers(row.generalKnowledge)}`,
                  `ME:${shortAnswers(row.mechanicalReasoning)}`,
                ].join(" | ");

                return (
                <tr key={key} className="transition hover:bg-slate-50 dark:hover:bg-white/5">
                  <td className="px-5 py-4">
                    <input
                      type="checkbox"
                      checked={validSelectedKeys.includes(key)}
                      onChange={() => toggleRow(key)}
                      className="h-4 w-4 rounded border-slate-300 text-emerald-600"
                      aria-label={`Select ${row.fileName || row.lrn || "scan result"}`}
                    />
                  </td>
                  {resultToExcelRow(row).map((cell, index) => (
                    <td key={`${row.fileName}-${index}`} className="px-5 py-4 font-mono text-xs text-slate-600 dark:text-slate-300">
                      {cell}
                    </td>
                  ))}
                  <td className="px-5 py-4">
                    {duplicateReasons.length > 0 ? (
                      <span className="inline-flex rounded-full bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-700 dark:bg-amber-400/10 dark:text-amber-200">
                        {duplicateReasons.join(", ")}
                      </span>
                    ) : (
                      <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">Unique</span>
                    )}
                  </td>
                  <td className="max-w-[360px] truncate px-5 py-4 font-mono text-xs text-slate-600 dark:text-slate-300">
                    {answerPreview}
                  </td>
                  <td className="px-5 py-4">
                    <button
                      type="button"
                      onClick={() => downloadResults(`minsu-result-${row.lrn || row.fileName || "scan"}.xls`, [row])}
                      className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-emerald-300 hover:text-emerald-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200"
                    >
                      <Download size={15} />
                      Excel
                    </button>
                  </td>
                  <td className="px-5 py-4">
                    <button
                      type="button"
                      onClick={() => void deleteResult(row, key)}
                      className="inline-flex items-center gap-2 rounded-lg border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-700 transition hover:border-rose-300 hover:bg-rose-50 dark:border-rose-400/20 dark:bg-white/5 dark:text-rose-200 dark:hover:bg-rose-400/10"
                    >
                      <Trash2 size={15} />
                      Delete
                    </button>
                  </td>
                </tr>
                );
              })}
              {visibleRows.length === 0 && (
                <tr>
                  <td colSpan={17} className="px-5 py-10 text-center text-sm font-semibold text-slate-500 dark:text-slate-400">
                    {filterMode === "duplicates" ? "No duplicate scan results detected." : "No active scan results."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </AppShell>
  );
}

export function AnalyticsPage() {
  return (
    <AppShell title="Analytics" eyebrow="Item analysis">
      <div className="space-y-6">
        <MetricCards />
        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Passing rate trend</h2>
              <BarChart3 className="text-sky-600" />
            </div>
            <div className="mt-8 flex h-72 items-end gap-3">
              {[62, 68, 71, 66, 73, 76, 82, 79, 84, 88, 86, 91].map((height, index) => (
                <div key={index} className="flex flex-1 flex-col items-center gap-2">
                  <motion.div
                    className="w-full rounded-t-lg bg-gradient-to-t from-sky-600 to-emerald-400"
                    initial={{ height: 0 }}
                    animate={{ height: `${height}%` }}
                    transition={{ duration: 0.7, delay: index * 0.03 }}
                  />
                  <span className="text-xs text-slate-500">{index + 1}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card className="p-6">
            <h2 className="text-xl font-semibold">Difficult questions</h2>
            <div className="mt-6 space-y-4">
              {itemAnalysis.map((item) => (
                <div key={item.item}>
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="font-medium">{item.item} · {item.subject}</span>
                    <span className="text-slate-500">{item.difficulty}% correct</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-800">
                    <div className="h-full rounded-full bg-amber-400" style={{ width: `${item.difficulty}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}

function createEmptyAnswerKeyState(): AnswerKeyState {
  return Object.fromEntries(
    answerKeys.map((key) => [
      key.subject,
      Array.from({ length: key.items }, (_, index) => defaultAnswerKeyState[key.subject]?.[index] || ""),
    ]),
  );
}

function countAnswered(answers: Array<AnswerChoice | "">) {
  return answers.filter(Boolean).length;
}

function readInitialAnswerKeyName() {
  if (typeof window === "undefined") return "MinSU Entrance Examination Answer Key";

  try {
    const stored = window.localStorage.getItem(ANSWER_KEY_STORE);
    if (!stored) return "MinSU Entrance Examination Answer Key";
    const payload = JSON.parse(stored) as { name?: string };
    return payload.name || "MinSU Entrance Examination Answer Key";
  } catch {
    return "MinSU Entrance Examination Answer Key";
  }
}

function readInitialAnswerKeyAnswers() {
  const emptyAnswers = createEmptyAnswerKeyState();
  if (typeof window === "undefined") return emptyAnswers;

  try {
    const stored = window.localStorage.getItem(ANSWER_KEY_STORE);
    if (!stored) return emptyAnswers;
    const payload = JSON.parse(stored) as { answers?: AnswerKeyState };
    return payload.answers ? { ...emptyAnswers, ...payload.answers } : emptyAnswers;
  } catch {
    return emptyAnswers;
  }
}

export function StudentsPage() {
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () => students.filter((student) => `${student.lrn} ${student.name}`.toLowerCase().includes(query.toLowerCase())),
    [query],
  );

  return (
    <AppShell title="Students" eyebrow="LRN records">
      <Card className="overflow-hidden">
        <div className="flex flex-col gap-4 border-b border-slate-200 p-5 dark:border-white/10 md:flex-row md:items-center md:justify-between">
          <div className="flex max-w-md flex-1 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-white/10 dark:bg-white/5">
            <Search size={17} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="w-full bg-transparent text-sm outline-none"
              placeholder="Search LRN or student name"
            />
          </div>
          <ActionButton><Plus size={17} /> Add student</ActionButton>
        </div>
        <SimpleTable
          heads={["LRN", "Name", "Campus", "Scan history", "Latest status"]}
          rows={filtered.map((student) => [student.lrn, student.name, student.campus, `${student.scans} scans`, student.latest])}
          statusIndex={4}
        />
      </Card>
    </AppShell>
  );
}

export function AnswerKeysPage() {
  const [answerKeyName, setAnswerKeyName] = useState(readInitialAnswerKeyName);
  const [activeSubject, setActiveSubject] = useState(answerKeys[0]?.subject || "");
  const [subjectAnswers, setSubjectAnswers] = useState<AnswerKeyState>(readInitialAnswerKeyAnswers);
  const [saveMessage, setSaveMessage] = useState("");
  const activeKey = answerKeys.find((key) => key.subject === activeSubject) || answerKeys[0];
  const activeAnswers = subjectAnswers[activeKey.subject] || [];
  const totalAnswered = answerKeys.reduce((total, key) => total + countAnswered(subjectAnswers[key.subject] || []), 0);
  const totalItems = answerKeys.reduce((total, key) => total + key.items, 0);

  const updateAnswer = (subject: string, itemIndex: number, choice: AnswerChoice) => {
    setSubjectAnswers((current) => {
      const currentAnswers = current[subject] || [];
      const nextAnswers = [...currentAnswers];
      nextAnswers[itemIndex] = nextAnswers[itemIndex] === choice ? "" : choice;
      return { ...current, [subject]: nextAnswers };
    });
    setSaveMessage("");
  };

  const clearActiveSubject = () => {
    setSubjectAnswers((current) => ({
      ...current,
      [activeKey.subject]: Array.from({ length: activeKey.items }, () => ""),
    }));
    setSaveMessage("");
  };

  const saveAnswerKey = () => {
    window.localStorage.setItem(
      ANSWER_KEY_STORE,
      JSON.stringify({
        name: answerKeyName,
        answers: subjectAnswers,
        savedAt: new Date().toISOString(),
      }),
    );
    setSaveMessage("Answer key saved in this browser.");
  };

  return (
    <AppShell title="Answer Key Management" eyebrow="Exam configuration">
      <div className="grid gap-6 xl:grid-cols-[0.78fr_1.22fr]">
        <Card className="p-6">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-slate-900 p-2 text-white dark:bg-sky-300 dark:text-slate-950">
              <KeyRound size={22} />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Edit answer key</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">Set the official A-D answers per subject.</p>
            </div>
          </div>
          <div className="mt-6 space-y-5">
            <label className="block">
              <span className="text-sm font-medium">Answer key name</span>
              <input
                value={answerKeyName}
                onChange={(event) => {
                  setAnswerKeyName(event.target.value);
                  setSaveMessage("");
                }}
                className="mt-2 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 outline-none transition focus:border-sky-400 dark:border-white/10 dark:bg-white/5"
              />
            </label>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-white/10 dark:bg-white/5">
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">Completion</p>
              <p className="mt-1 text-3xl font-semibold tracking-tight">{totalAnswered}/{totalItems}</p>
              <div className="mt-3 h-2 rounded-full bg-slate-200 dark:bg-slate-800">
                <div
                  className="h-full rounded-full bg-emerald-500 transition-all"
                  style={{ width: `${totalItems > 0 ? (totalAnswered / totalItems) * 100 : 0}%` }}
                />
              </div>
            </div>
            <div className="space-y-2">
              {answerKeys.map((key) => {
                const answered = countAnswered(subjectAnswers[key.subject] || []);
                const isActive = key.subject === activeKey.subject;

                return (
                  <button
                    key={key.subject}
                    type="button"
                    onClick={() => setActiveSubject(key.subject)}
                    className={`w-full rounded-lg border px-4 py-3 text-left transition ${
                      isActive
                        ? "border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-400/30 dark:bg-emerald-400/10 dark:text-emerald-100"
                        : "border-slate-200 bg-white hover:border-sky-200 hover:bg-sky-50/60 dark:border-white/10 dark:bg-white/5 dark:hover:bg-white/10"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="font-semibold">{key.subject}</span>
                      <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                        {answered}/{key.items}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{key.code} - {key.items} items</p>
                  </button>
                );
              })}
            </div>
            <div className="flex flex-wrap gap-2">
              <ActionButton onClick={saveAnswerKey}><ClipboardList size={17} /> Save answer key</ActionButton>
              <ActionButton variant="secondary" onClick={clearActiveSubject}><RefreshCw size={17} /> Clear subject</ActionButton>
            </div>
            {saveMessage && (
              <p className="rounded-lg bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-700 dark:bg-emerald-400/10 dark:text-emerald-200">
                {saveMessage}
              </p>
            )}
          </div>
        </Card>
        <Card className="overflow-hidden">
          <div className="flex flex-col gap-3 border-b border-slate-200 p-5 dark:border-white/10 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-xl font-semibold">{activeKey.subject}</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {answerKeyName} - select one answer per item.
              </p>
            </div>
            <span className="rounded-full bg-sky-50 px-3 py-1 text-sm font-semibold text-sky-700 dark:bg-sky-400/10 dark:text-sky-200">
              {countAnswered(activeAnswers)}/{activeKey.items} answered
            </span>
          </div>
          <div className="max-h-[720px] overflow-y-auto p-5">
            <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
              {Array.from({ length: activeKey.items }, (_, index) => {
                const itemNumber = index + 1;
                const selected = activeAnswers[index] || "";

                return (
                  <div
                    key={`${activeKey.subject}-${itemNumber}`}
                    className="rounded-lg border border-slate-200 bg-white p-3 dark:border-white/10 dark:bg-white/5"
                  >
                    <div className="mb-3 flex items-center justify-between">
                      <span className="text-sm font-semibold">No. {itemNumber}</span>
                      <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                        {selected || "Blank"}
                      </span>
                    </div>
                    <div className="grid grid-cols-4 gap-2">
                      {answerChoices.map((choice) => (
                        <button
                          key={choice}
                          type="button"
                          onClick={() => updateAnswer(activeKey.subject, index, choice)}
                          className={`h-10 rounded-lg border text-sm font-semibold transition ${
                            selected === choice
                              ? "border-emerald-500 bg-emerald-500 text-white shadow-sm shadow-emerald-600/20"
                              : "border-slate-200 bg-slate-50 text-slate-700 hover:border-sky-300 hover:bg-sky-50 dark:border-white/10 dark:bg-slate-900/40 dark:text-slate-200 dark:hover:bg-white/10"
                          }`}
                        >
                          {choice}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

export function HistoryPage() {
  const [logs, setLogs] = useState<UploadLog[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");

  const refreshLogs = async () => {
    setIsRefreshing(true);

    const nextLogs = new Map<string, UploadLog>();

    for (const result of readStoredResults().map(resultHistoryLog)) {
      nextLogs.set(result.id, result);
    }

    try {
      const response = await fetch("/api/results", { cache: "no-store" });
      const payload = (await readJsonPayload<{
        results?: ScanResult[];
        dbAvailable?: boolean;
      }>(response));

      if (response.ok && payload.dbAvailable !== false) {
        (payload.results || []).forEach((result, index) => {
          const log = resultHistoryLog(result, index);
          nextLogs.set(log.id, log);
        });
      }
    } catch {
      // Local browser logs are still enough for the audit trail when the database is offline.
    }

    for (const log of readStoredUploadLogs()) {
      nextLogs.set(log.id, log);
    }

    setLogs(
      Array.from(nextLogs.values()).sort((left, right) => Date.parse(right.date) - Date.parse(left.date)),
    );
    setLastUpdated(new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" }));
    setIsRefreshing(false);
  };

  useEffect(() => {
    const update = () => void refreshLogs();
    const firstRefresh = window.setTimeout(update, 0);
    const interval = window.setInterval(update, 3000);

    window.addEventListener(UPLOAD_LOG_EVENT, update);
    window.addEventListener("minsu-scan-results-updated", update);
    window.addEventListener("storage", update);

    return () => {
      window.clearTimeout(firstRefresh);
      window.clearInterval(interval);
      window.removeEventListener(UPLOAD_LOG_EVENT, update);
      window.removeEventListener("minsu-scan-results-updated", update);
      window.removeEventListener("storage", update);
    };
  }, []);

  return (
    <AppShell title="History" eyebrow="Audit logs">
      <Card className="overflow-hidden">
        <div className="flex flex-col gap-3 border-b border-slate-200 p-5 dark:border-white/10 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-xl font-semibold">Previous uploads and reports</h2>
              <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-semibold text-emerald-800 dark:bg-emerald-400/15 dark:text-emerald-200">
                Live
              </span>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Auto-refreshing scan logs for admissions audit trails{lastUpdated ? ` · Updated ${lastUpdated}` : ""}.
            </p>
          </div>
          <ActionButton variant="secondary" onClick={() => void refreshLogs()} disabled={isRefreshing}>
            <RefreshCw className={isRefreshing ? "animate-spin" : ""} size={17} />
            Refresh logs
          </ActionButton>
        </div>
        <SimpleTable
          heads={["File", "Date", "User", "Result", "Status"]}
          rows={logs.map((item) => [item.file, formatHistoryDate(item.date), item.user, item.result, item.status])}
          statusIndex={4}
          emptyMessage="No upload logs yet. Upload a sheet to create the first live audit record."
        />
      </Card>
    </AppShell>
  );
}

function SimpleTable({
  heads,
  rows,
  statusIndex,
  emptyMessage = "No records found.",
}: {
  heads: string[];
  rows: string[][];
  statusIndex?: number;
  emptyMessage?: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase text-slate-500 dark:bg-white/5">
          <tr>{heads.map((head) => <th key={head} className="px-5 py-3">{head}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-white/10">
          {rows.length === 0 && (
            <tr>
              <td className="px-5 py-8 text-center text-slate-500 dark:text-slate-400" colSpan={heads.length}>
                {emptyMessage}
              </td>
            </tr>
          )}
          {rows.map((row) => (
            <tr key={row.join("-")} className="transition hover:bg-slate-50 dark:hover:bg-white/5">
              {row.map((cell, index) => (
                <td key={`${cell}-${index}`} className={`px-5 py-4 ${index === 0 ? "font-medium" : "text-slate-600 dark:text-slate-300"}`}>
                  {statusIndex === index ? <StatusBadge status={cell} /> : cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
