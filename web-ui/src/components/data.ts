import {
  BarChart3,
  BookOpenCheck,
  Clock3,
  FileCheck2,
  Gauge,
  GraduationCap,
  History,
  LayoutDashboard,
  ScanLine,
  UploadCloud,
  UsersRound,
} from "lucide-react";

export const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload Scanner", icon: UploadCloud },
  { href: "/results", label: "Scan Results", icon: FileCheck2 },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/students", label: "Students", icon: UsersRound },
  { href: "/answer-keys", label: "Answer Keys", icon: BookOpenCheck },
  { href: "/history", label: "History", icon: History },
];

export const metrics = [
  { label: "Sheets scanned", value: "12,480", delta: "+18.4%", icon: ScanLine },
  { label: "Average score", value: "82.6", delta: "+4.1 pts", icon: Gauge },
  { label: "Passing rate", value: "76%", delta: "+7.2%", icon: GraduationCap },
  { label: "Queue time", value: "01:42", delta: "-36 sec", icon: Clock3 },
];

export const recentScans = [
  { batch: "MINSU-ENT-2026-A", sheets: 248, status: "Validated", time: "8 min ago" },
  { batch: "MINSU-ENT-2026-B", sheets: 186, status: "Review", time: "22 min ago" },
  { batch: "MINSU-ENT-2026-C", sheets: 312, status: "Exported", time: "1 hr ago" },
  { batch: "MINSU-ENT-2026-D", sheets: 94, status: "Validated", time: "2 hrs ago" },
];

export const results = [
  { lrn: "109482030121", barcode: "MSU-26-00091", name: "Aira Mendoza", score: 91, status: "Passed", preview: "A C D B E" },
  { lrn: "109482030145", barcode: "MSU-26-00104", name: "Jerome Cruz", score: 86, status: "Passed", preview: "B D A C C" },
  { lrn: "109482030178", barcode: "MSU-26-00127", name: "Trisha Santos", score: 74, status: "Review", preview: "C A E B D" },
  { lrn: "109482030219", barcode: "MSU-26-00163", name: "Miguel Reyes", score: 68, status: "Retake", preview: "D B C A E" },
  { lrn: "109482030244", barcode: "MSU-26-00188", name: "Lea Francisco", score: 94, status: "Passed", preview: "A A C D B" },
];

export const students = [
  { lrn: "109482030121", name: "Aira Mendoza", campus: "Main Campus", scans: 3, latest: "Passed" },
  { lrn: "109482030145", name: "Jerome Cruz", campus: "Calapan", scans: 2, latest: "Passed" },
  { lrn: "109482030178", name: "Trisha Santos", campus: "Bongabong", scans: 2, latest: "Review" },
  { lrn: "109482030219", name: "Miguel Reyes", campus: "Main Campus", scans: 1, latest: "Retake" },
  { lrn: "109482030244", name: "Lea Francisco", campus: "Calapan", scans: 4, latest: "Passed" },
];

export const answerKeys = [
  { subject: "Language Proficiency", code: "LP-35", items: 35, version: "v3.1", updated: "Today" },
  { subject: "Mathematics", code: "MATH-35", items: 35, version: "v3.1", updated: "Today" },
  { subject: "Science", code: "SCI-35", items: 35, version: "v3.1", updated: "Yesterday" },
  { subject: "Logic and Abstract Reasoning", code: "LAR-20", items: 20, version: "v2.4", updated: "May 5" },
  { subject: "Mechanical Reasoning", code: "MR-20", items: 20, version: "v2.4", updated: "May 5" },
  { subject: "General Knowledge", code: "GK-35", items: 35, version: "v3.1", updated: "Today" },
];

export const history = [
  { file: "batch_a_248_sheets.zip", date: "May 7, 2026", user: "Registrar", result: "248 rows", status: "Completed" },
  { file: "batch_b_review_queue.pdf", date: "May 7, 2026", user: "Admissions", result: "18 flagged", status: "Review" },
  { file: "mock_exam_recheck.zip", date: "May 6, 2026", user: "Testing Office", result: "94 rows", status: "Completed" },
  { file: "barcode_lrn_test.pdf", date: "May 5, 2026", user: "IT Admin", result: "Exported", status: "Completed" },
];

export const itemAnalysis = [
  { item: "Q12", difficulty: 38, subject: "Mathematics" },
  { item: "Q47", difficulty: 42, subject: "Science" },
  { item: "Q83", difficulty: 45, subject: "Logic" },
  { item: "Q126", difficulty: 49, subject: "Mechanical" },
  { item: "Q151", difficulty: 52, subject: "General Knowledge" },
];
