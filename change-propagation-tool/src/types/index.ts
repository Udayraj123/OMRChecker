/**
 * Type definitions for the change propagation tool
 */

export interface FileMapping {
  python: string;
  typescript: string | null;
  status: 'synced' | 'partial' | 'not_started';
  phase: number | string;
  priority: 'high' | 'medium' | 'low';
  lastSyncedCommit: string | null;
  lastPythonChange: string | null;
  lastTypescriptChange: string | null;
  testFile?: string;
  classes?: ClassMapping[];
  functions?: FunctionMapping[] | string[];
  notes?: string;
}

export interface FunctionMapping {
  python: string;
  typescript: string;
  synced: boolean;
  notes?: string;
}

export interface ClassMapping {
  python: string;
  typescript: string;
  synced: boolean;
  methods?: MethodMapping[];
  note?: string;
}

export interface MethodMapping {
  python: string;
  typescript: string;
  synced: boolean;
  notes?: string;
}


export interface FileMappingData {
  version: string;
  metadata?: Record<string, any>;
  phases?: Record<string, PhaseDefinition>;
  mappings: FileMapping[];
  statistics?: {
    total: number;
    synced: number;
    partial: number;
    not_started: number;
    phase1: number;
    phase2: number;
    future: number;
  };
}

export interface PhaseDefinition {
  name: string;
  description: string;
}

export interface ChangeReport {
  timestamp: string;
  total_files_changed: number;
  changes: FileChange[];
}

export interface FileChange {
  pythonFile: string;
  typescriptFile: string | null;
  status: string;
  phase: number | string;
  priority: string;
  classes: ClassChange[];
  functions: string[];
}

export interface ClassChange {
  type: 'added' | 'modified' | 'deleted';
  name: string;
  lineRange: [number, number] | null;
  methods: MethodChange[];
}

export interface MethodChange {
  type: 'added' | 'modified' | 'deleted';
  name: string;
  lineRange: [number, number] | null;
  changeDetails: Record<string, any>;
}

