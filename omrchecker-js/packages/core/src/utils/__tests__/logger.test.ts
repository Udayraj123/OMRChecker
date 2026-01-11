/**
 * Tests for logger utility
 *
 * TypeScript port of src/tests/test_logger.py
 */

import { describe, it, expect } from 'vitest';
import { Logger } from '../logger';

describe('Logger', () => {
  it('should create logger with default settings', () => {
    const log = new Logger('test_logger');
    expect(log).toBeDefined();
    expect(log['showLogsByType']).toBeDefined();
  });

  it('should log debug message', () => {
    const log = new Logger('test_debug');
    // Should not throw exception
    expect(() => log.debug('Test debug message')).not.toThrow();
  });

  it('should log info message', () => {
    const log = new Logger('test_info');
    expect(() => log.info('Test info message')).not.toThrow();
  });

  it('should log warning message', () => {
    const log = new Logger('test_warning');
    expect(() => log.warn('Test warning message')).not.toThrow();
  });

  it('should log error message', () => {
    const log = new Logger('test_error');
    expect(() => log.error('Test error message')).not.toThrow();
  });

  it('should log fatal message', () => {
    const log = new Logger('test_fatal');
    expect(() => log.fatal('Test fatal message')).not.toThrow();
  });

  it('should set log levels', () => {
    const log = new Logger('test_levels');
    log.setLogLevels({ debug: false, info: true });
    expect(log['showLogsByType'].debug).toBe(false);
    expect(log['showLogsByType'].info).toBe(true);
  });

  it('should reset log levels', () => {
    const log = new Logger('test_reset');
    log.setLogLevels({ debug: false });
    log.resetLogLevels();
    expect(log['showLogsByType'].debug).toBe(true);
  });

  it('should log multiple messages', () => {
    const log = new Logger('test_multi');
    expect(() => log.info('Message 1', 'Message 2', 'Message 3')).not.toThrow();
  });
});

