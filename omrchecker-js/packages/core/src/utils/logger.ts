/**
 * Logger utility for OMRChecker.
 *
 * Migrated from src/utils/logger.py.
 * Wraps browser console methods with per-level filtering.
 */

const DEFAULT_LOG_LEVEL_MAP = {
  critical: true,
  error: true,
  warning: true,
  info: true,
  debug: true,
} as const;

type LogLevelMap = { [K in keyof typeof DEFAULT_LOG_LEVEL_MAP]?: boolean };

export class Logger {
  name: string;
  showLogsByType: Record<string, boolean>;

  constructor(name: string) {
    this.name = name;
    this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP };
  }

  setLogLevels(levels: LogLevelMap): void {
    this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP, ...levels };
  }

  resetLogLevels(): void {
    this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP };
  }

  private logutil(methodType: string, ...msg: unknown[]): void {
    if (this.showLogsByType[methodType] === false) return;
    const str = msg.map(v => (typeof v === 'string' ? v : String(v))).join(' ');
    if (methodType === 'critical' || methodType === 'error') {
      console.error(str);
    } else if (methodType === 'warning') {
      console.warn(str);
    } else if (methodType === 'debug') {
      console.debug(str);
    } else {
      console.log(str);
    }
  }

  debug(...msg: unknown[]): void {
    this.logutil('debug', ...msg);
  }

  info(...msg: unknown[]): void {
    this.logutil('info', ...msg);
  }

  warning(...msg: unknown[]): void {
    this.logutil('warning', ...msg);
  }

  error(...msg: unknown[]): void {
    this.logutil('error', ...msg);
  }

  critical(...msg: unknown[]): void {
    this.logutil('critical', ...msg);
  }
}

export const logger = new Logger('omrchecker');
