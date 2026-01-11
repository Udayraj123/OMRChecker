/**
 * Logger utility for OMRChecker
 *
 * TypeScript port of src/utils/logger.py using consola
 */

import { ConsolaInstance, createConsola } from 'consola';

interface LogLevelMap {
  fatal: boolean;
  error: boolean;
  warn: boolean;
  info: boolean;
  debug: boolean;
}

const DEFAULT_LOG_LEVEL_MAP: LogLevelMap = {
  fatal: true,
  error: true,
  warn: true,
  info: true,
  debug: true,
};

export class Logger {
  private consola: ConsolaInstance;
  private showLogsByType: LogLevelMap;

  constructor(tag: string) {
    this.consola = createConsola({
      defaults: {
        tag,
      },
    });
    this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP };
  }

  setLogLevels(levels: Partial<LogLevelMap>): void {
    this.showLogsByType = { ...this.showLogsByType, ...levels };
  }

  resetLogLevels(): void {
    this.showLogsByType = { ...DEFAULT_LOG_LEVEL_MAP };
  }

  debug(...messages: string[]): void {
    if (this.showLogsByType.debug) {
      this.consola.debug(messages.join(' '));
    }
  }

  info(...messages: string[]): void {
    if (this.showLogsByType.info) {
      this.consola.info(messages.join(' '));
    }
  }

  warn(...messages: string[]): void {
    if (this.showLogsByType.warn) {
      this.consola.warn(messages.join(' '));
    }
  }

  error(...messages: string[]): void {
    if (this.showLogsByType.error) {
      this.consola.error(messages.join(' '));
    }
  }

  fatal(...messages: string[]): void {
    if (this.showLogsByType.fatal) {
      this.consola.fatal(messages.join(' '));
    }
  }
}

// Default logger instance
export const logger = new Logger('omrchecker');

