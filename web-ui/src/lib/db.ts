import mysql from "mysql2/promise";
import type { ScanResult } from "@/components/scan-store";

const DB_NAME = process.env.MYSQL_DATABASE || "minsu_omr_scanner";

let pool: mysql.Pool | null = null;

function getServerPool(database?: string) {
  return mysql.createPool({
    host: process.env.MYSQL_HOST || "127.0.0.1",
    port: Number(process.env.MYSQL_PORT || 3306),
    user: process.env.MYSQL_USER || "root",
    password: process.env.MYSQL_PASSWORD || "",
    database,
    waitForConnections: true,
    connectionLimit: 10,
    namedPlaceholders: true,
  });
}

export function getDbPool() {
  if (!pool) {
    pool = getServerPool(DB_NAME);
  }

  return pool;
}

export async function ensureDatabase() {
  const serverPool = getServerPool();

  await serverPool.query(`CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci`);
  await serverPool.end();

  await getDbPool().query(`
    CREATE TABLE IF NOT EXISTS scan_results (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      application_number VARCHAR(100) NOT NULL DEFAULT '',
      lrn VARCHAR(100) NOT NULL DEFAULT '',
      surname VARCHAR(150) NOT NULL DEFAULT '',
      given_name VARCHAR(150) NOT NULL DEFAULT '',
      middle_name VARCHAR(150) NOT NULL DEFAULT '',
      exam_date VARCHAR(50) NOT NULL DEFAULT '',
      source_file_name VARCHAR(255) NOT NULL DEFAULT '',
      language_proficiency TEXT NOT NULL,
      mathematics TEXT NOT NULL,
      science TEXT NOT NULL,
      logic_and_abstract_reasoning TEXT NOT NULL,
      general_knowledge TEXT NOT NULL,
      mechanical_reasoning TEXT NOT NULL,
      language_proficiency_detected INT NOT NULL DEFAULT 0,
      mathematics_detected INT NOT NULL DEFAULT 0,
      science_detected INT NOT NULL DEFAULT 0,
      logic_and_abstract_reasoning_detected INT NOT NULL DEFAULT 0,
      general_knowledge_detected INT NOT NULL DEFAULT 0,
      mechanical_reasoning_detected INT NOT NULL DEFAULT 0,
      language_proficiency_score INT NOT NULL DEFAULT 0,
      mathematics_score INT NOT NULL DEFAULT 0,
      science_score INT NOT NULL DEFAULT 0,
      logic_and_abstract_reasoning_score INT NOT NULL DEFAULT 0,
      general_knowledge_score INT NOT NULL DEFAULT 0,
      mechanical_reasoning_score INT NOT NULL DEFAULT 0,
      file_name VARCHAR(255) NOT NULL DEFAULT '',
      checked_image_path TEXT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      deleted_at TIMESTAMP NULL,
      INDEX idx_lrn (lrn),
      INDEX idx_application_number (application_number),
      INDEX idx_file_name (file_name),
      INDEX idx_deleted_at (deleted_at)
    )
  `);
  const scoreColumns = [
    ["exam_date", "VARCHAR(50) NOT NULL DEFAULT ''"],
    ["source_file_name", "VARCHAR(255) NOT NULL DEFAULT ''"],
    ["language_proficiency_detected", "INT NOT NULL DEFAULT 0"],
    ["mathematics_detected", "INT NOT NULL DEFAULT 0"],
    ["science_detected", "INT NOT NULL DEFAULT 0"],
    ["logic_and_abstract_reasoning_detected", "INT NOT NULL DEFAULT 0"],
    ["general_knowledge_detected", "INT NOT NULL DEFAULT 0"],
    ["mechanical_reasoning_detected", "INT NOT NULL DEFAULT 0"],
    ["language_proficiency_score", "INT NOT NULL DEFAULT 0"],
    ["mathematics_score", "INT NOT NULL DEFAULT 0"],
    ["science_score", "INT NOT NULL DEFAULT 0"],
    ["logic_and_abstract_reasoning_score", "INT NOT NULL DEFAULT 0"],
    ["general_knowledge_score", "INT NOT NULL DEFAULT 0"],
    ["mechanical_reasoning_score", "INT NOT NULL DEFAULT 0"],
    ["deleted_at", "TIMESTAMP NULL"],
  ];

  for (const [column, definition] of scoreColumns) {
    try {
      await getDbPool().query(`ALTER TABLE scan_results ADD COLUMN ${column} ${definition}`);
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes("Duplicate column")) {
        throw error;
      }
    }
  }

  try {
    await getDbPool().query("ALTER TABLE scan_results ADD INDEX idx_deleted_at (deleted_at)");
  } catch (error) {
    if (!(error instanceof Error) || !error.message.includes("Duplicate key name")) {
      throw error;
    }
  }

  await getDbPool().query("DELETE FROM scan_results WHERE application_number = 'APP-2026-047281'");
}

export async function saveScanResults(results: ScanResult[]) {
  if (results.length === 0) return;

  await ensureDatabase();

  const values = results.map((result) => [
    result.applicationNumber,
    result.lrn,
    result.surname,
    result.name,
    result.middleName,
    result.examDate || "",
    result.sourceFileName || result.fileName,
    result.languageProficiency,
    result.mathematics,
    result.science,
    result.logicAndAbstractReasoning,
    result.generalKnowledge,
    result.mechanicalReasoning,
    result.languageProficiencyDetected || 0,
    result.mathematicsDetected || 0,
    result.scienceDetected || 0,
    result.logicAndAbstractReasoningDetected || 0,
    result.generalKnowledgeDetected || 0,
    result.mechanicalReasoningDetected || 0,
    result.languageProficiencyScore || 0,
    result.mathematicsScore || 0,
    result.scienceScore || 0,
    result.logicAndAbstractReasoningScore || 0,
    result.generalKnowledgeScore || 0,
    result.mechanicalReasoningScore || 0,
    result.fileName,
    result.checkedImagePath || null,
  ]);

  await getDbPool().query(
    `
      INSERT INTO scan_results (
        application_number,
        lrn,
        surname,
        given_name,
        middle_name,
        exam_date,
        source_file_name,
        language_proficiency,
        mathematics,
        science,
        logic_and_abstract_reasoning,
        general_knowledge,
        mechanical_reasoning,
        language_proficiency_detected,
        mathematics_detected,
        science_detected,
        logic_and_abstract_reasoning_detected,
        general_knowledge_detected,
        mechanical_reasoning_detected,
        language_proficiency_score,
        mathematics_score,
        science_score,
        logic_and_abstract_reasoning_score,
        general_knowledge_score,
        mechanical_reasoning_score,
        file_name,
        checked_image_path
      )
      VALUES ?
    `,
    [values],
  );
}

type ScanResultRow = {
  id: number;
  application_number: string;
  lrn: string;
  surname: string;
  given_name: string;
  middle_name: string;
  exam_date: string;
  source_file_name: string;
  language_proficiency: string;
  mathematics: string;
  science: string;
  logic_and_abstract_reasoning: string;
  general_knowledge: string;
  mechanical_reasoning: string;
  language_proficiency_detected: number;
  mathematics_detected: number;
  science_detected: number;
  logic_and_abstract_reasoning_detected: number;
  general_knowledge_detected: number;
  mechanical_reasoning_detected: number;
  language_proficiency_score: number;
  mathematics_score: number;
  science_score: number;
  logic_and_abstract_reasoning_score: number;
  general_knowledge_score: number;
  mechanical_reasoning_score: number;
  file_name: string;
  checked_image_path: string | null;
  deleted_at: Date | string | null;
};

export async function listScanResults(): Promise<ScanResult[]> {
  await ensureDatabase();

  const [rows] = await getDbPool().query<mysql.RowDataPacket[]>(
    `
      SELECT
        id,
        application_number,
        lrn,
        surname,
        given_name,
        middle_name,
        exam_date,
        source_file_name,
        language_proficiency,
        mathematics,
        science,
        logic_and_abstract_reasoning,
        general_knowledge,
        mechanical_reasoning,
        language_proficiency_detected,
        mathematics_detected,
        science_detected,
        logic_and_abstract_reasoning_detected,
        general_knowledge_detected,
        mechanical_reasoning_detected,
        language_proficiency_score,
        mathematics_score,
        science_score,
        logic_and_abstract_reasoning_score,
        general_knowledge_score,
        mechanical_reasoning_score,
        file_name,
        checked_image_path,
        deleted_at
      FROM scan_results
      WHERE deleted_at IS NULL
      ORDER BY id DESC
      LIMIT 500
    `,
  );

  return (rows as ScanResultRow[]).map((row) => ({
    id: row.id,
    applicationNumber: row.application_number,
    lrn: row.lrn,
    surname: row.surname,
    name: row.given_name,
    middleName: row.middle_name,
    examDate: row.exam_date,
    sourceFileName: row.source_file_name,
    languageProficiency: row.language_proficiency,
    mathematics: row.mathematics,
    science: row.science,
    logicAndAbstractReasoning: row.logic_and_abstract_reasoning,
    generalKnowledge: row.general_knowledge,
    mechanicalReasoning: row.mechanical_reasoning,
    languageProficiencyDetected: row.language_proficiency_detected,
    mathematicsDetected: row.mathematics_detected,
    scienceDetected: row.science_detected,
    logicAndAbstractReasoningDetected: row.logic_and_abstract_reasoning_detected,
    generalKnowledgeDetected: row.general_knowledge_detected,
    mechanicalReasoningDetected: row.mechanical_reasoning_detected,
    languageProficiencyScore: row.language_proficiency_score,
    mathematicsScore: row.mathematics_score,
    scienceScore: row.science_score,
    logicAndAbstractReasoningScore: row.logic_and_abstract_reasoning_score,
    generalKnowledgeScore: row.general_knowledge_score,
    mechanicalReasoningScore: row.mechanical_reasoning_score,
    fileName: row.file_name,
    checkedImagePath: row.checked_image_path || undefined,
    deletedAt: row.deleted_at ? String(row.deleted_at) : null,
  }));
}

export async function softDeleteScanResult(id: number) {
  await ensureDatabase();

  await getDbPool().query("UPDATE scan_results SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?", [id]);
}
