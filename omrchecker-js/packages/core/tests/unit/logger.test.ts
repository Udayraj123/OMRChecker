/**
 * Unit tests for Logger utility.
 *
 * Translated from src/tests/utils/__tests__/test_logger.py (10 tests).
 */

import { describe, it, expect } from 'vitest';
import { Logger } from '../../src/utils/logger';

describe('Logger', () => {
  it('test_logger_creation: creates logger with non-null showLogsByType', () => {
    const log = new Logger('test_logger');
    expect(log).not.toBeNull();
    expect(log.showLogsByType).not.toBeNull();
  });

  it('test_logger_debug_message: debug does not throw', () => {
    const log = new Logger('test_debug');
    expect(() => log.debug('Test debug message')).not.toThrow();
  });

  it('test_logger_info_message: info does not throw', () => {
    const log = new Logger('test_info');
    expect(() => log.info('Test info message')).not.toThrow();
  });

  it('test_logger_warning_message: warning does not throw', () => {
    const log = new Logger('test_warning');
    expect(() => log.warning('Test warning message')).not.toThrow();
  });

  it('test_logger_error_message: error does not throw', () => {
    const log = new Logger('test_error');
    expect(() => log.error('Test error message')).not.toThrow();
  });

  it('test_logger_critical_message: critical does not throw', () => {
    const log = new Logger('test_critical');
    expect(() => log.critical('Test critical message')).not.toThrow();
  });

  it('test_logger_set_levels: setLogLevels merges with defaults', () => {
    const log = new Logger('test_levels');
    log.setLogLevels({ debug: false, info: true });
    expect(log.showLogsByType['debug']).toBe(false);
    expect(log.showLogsByType['info']).toBe(true);
  });

  it('test_logger_reset_levels: resetLogLevels restores debug to true', () => {
    const log = new Logger('test_reset');
    log.setLogLevels({ debug: false });
    log.resetLogLevels();
    expect(log.showLogsByType['debug']).toBe(true);
  });

  it('test_logger_multiple_messages: info with multiple args does not throw', () => {
    const log = new Logger('test_multi');
    expect(() => log.info('Message 1', 'Message 2', 'Message 3')).not.toThrow();
  });

  it('test_logger_custom_separator: info with multiple args (no sep param) does not throw', () => {
    const log = new Logger('test_sep');
    expect(() => log.info('Part1', 'Part2')).not.toThrow();
  });
});
