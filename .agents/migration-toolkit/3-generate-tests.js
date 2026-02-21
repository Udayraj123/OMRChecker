#!/usr/bin/env node

/**
 * OMRChecker Migration Toolkit - Test Scaffolding Generator
 * Converts pytest structure to Vitest
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
  pythonTestRoot: path.join(__dirname, '../../src/tests'),
  targetRoot: path.join(__dirname, '../../omrchecker-js/tests'),
};

// Test file discoveries
const testFiles = [];
const fixtures = new Set();

/**
 * Scan Python test file and extract test structure
 */
async function scanPythonTestFile(filePath) {
  const content = await fs.readFile(filePath, 'utf-8');
  const relativePath = path.relative(CONFIG.pythonTestRoot, filePath);

  const tests = [];
  const testFunctionRegex = /def\s+(test_\w+)\((.*?)\):/g;
  let match;

  while ((match = testFunctionRegex.exec(content)) !== null) {
    const [, testName, params] = match;
    const paramList = params.split(',').map(p => p.trim()).filter(Boolean);

    // Extract fixtures from parameters
    paramList.forEach(param => {
      if (!param.includes('=') && !param.startsWith('*')) {
        fixtures.add(param);
      }
    });

    tests.push({
      name: testName,
      params: paramList,
      pythonName: testName,
    });
  }

  // Extract test classes
  const testClassRegex = /class\s+(Test\w+):/g;
  const classes = [];

  while ((match = testClassRegex.exec(content)) !== null) {
    classes.push(match[1]);
  }

  return {
    relativePath,
    tests,
    classes,
    fixtures: Array.from(fixtures),
  };
}

/**
 * Generate Vitest test file from pytest structure
 */
function generateVitestTestFile(testData) {
  const { relativePath, tests, classes } = testData;

  // Determine imports based on file path
  const modulePath = relativePath
    .replace(/\/__tests__\//g, '/')
    .replace(/\.py$/, '')
    .replace(/test_/g, '');

  const imports = [];
  imports.push(`import { describe, it, expect, beforeEach, afterEach } from 'vitest';`);

  // Add module imports (inferred from test file location)
  if (relativePath.includes('processors/')) {
    const processorType = relativePath.split('/')[1];
    imports.push(`import * as ${processorType} from '@/processors/${processorType}';`);
  } else if (relativePath.includes('core/')) {
    imports.push(`import * as core from '@/core';`);
  } else if (relativePath.includes('utils/')) {
    const utilType = relativePath.split('/')[1];
    imports.push(`import * as ${utilType} from '@/utils/${utilType}';`);
  }

  // Add fixture imports
  imports.push(`import { mockTemplate, minimalArgs } from '../fixtures';`);

  const testContent = [];

  // Generate test suites
  if (classes.length > 0) {
    classes.forEach(className => {
      const suiteName = className.replace(/^Test/, '');
      testContent.push(`describe('${suiteName}', () => {`);
      testContent.push(`  // TODO: Add setup/teardown if needed`);
      testContent.push(`  `);
      testContent.push(`  // Test cases will be added here`);
      testContent.push(`});`);
      testContent.push('');
    });
  }

  // Generate individual tests
  if (tests.length > 0) {
    const describeBlock = relativePath.split('/').pop().replace(/test_|\.py/g, '');
    testContent.push(`describe('${describeBlock}', () => {`);

    tests.forEach(test => {
      const testDescription = test.name.replace(/test_/g, '').replace(/_/g, ' ');
      testContent.push(`  it('should ${testDescription}', () => {`);
      testContent.push(`    // TODO: Implement test`);

      // Add fixture hints
      if (test.params.length > 0) {
        testContent.push(`    // Uses fixtures: ${test.params.join(', ')}`);
      }

      testContent.push(`    expect(true).toBe(true); // Placeholder`);
      testContent.push(`  });`);
      testContent.push('');
    });

    testContent.push(`});`);
  }

  return [
    '// Auto-generated from pytest structure',
    '// Implement test bodies manually\n',
    ...imports,
    '',
    ...testContent,
  ].join('\n');
}

/**
 * Walk directory tree and scan test files
 */
async function walkDirectory(dir, targetDir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relativePath = path.relative(CONFIG.pythonTestRoot, fullPath);
    const targetPath = path.join(targetDir, relativePath);

    if (entry.isDirectory()) {
      // Skip special directories
      if (entry.name === '__pycache__' || entry.name === '.pytest_cache') {
        continue;
      }

      // Create corresponding directory in target
      await fs.mkdir(targetPath, { recursive: true });
      await walkDirectory(fullPath, targetDir);
    } else if (entry.isFile() && entry.name.startsWith('test_') && entry.name.endsWith('.py')) {
      // Scan test file
      const testData = await scanPythonTestFile(fullPath);
      testFiles.push(testData);

      // Generate Vitest file
      const vitestContent = generateVitestTestFile(testData);
      const vitestPath = targetPath.replace(/\.py$/, '.test.ts');

      await fs.writeFile(vitestPath, vitestContent);
      console.log(`   ✓ ${vitestPath.replace(targetDir + '/', '')}`);
    } else if (entry.isFile() && entry.name === 'conftest.py') {
      // Handle conftest.py - extract fixtures
      await extractFixtures(fullPath);
    }
  }
}

/**
 * Extract fixtures from conftest.py
 */
async function extractFixtures(confTestPath) {
  const content = await fs.readFile(confTestPath, 'utf-8');

  const fixtureRegex = /@pytest\.fixture\s+def\s+(\w+)\((.*?)\):/g;
  let match;

  while ((match = fixtureRegex.exec(content)) !== null) {
    const [, fixtureName] = match;
    fixtures.add(fixtureName);
  }
}

/**
 * Generate fixtures file
 */
async function generateFixtures() {
  const fixturesPath = path.join(CONFIG.targetRoot, 'fixtures.ts');

  const fixtureContent = `// Auto-generated test fixtures
// Converted from pytest fixtures in conftest.py

/**
 * Mock template fixture
 */
export const mockTemplate = {
  path: '/test/template.json',
  fieldBlocks: {},
  preProcessors: [],
  tuningConfig: {
    thresholding: {
      gammaLow: 0.7,
      minGapTwoBubbles: 30,
    },
    outputs: {
      showImageLevel: 0,
    },
  },
};

/**
 * Minimal args fixture
 */
export const minimalArgs = {
  debug: false,
  outputMode: 'default',
  setLayout: false,
};

/**
 * Minimal template JSON fixture
 */
export const minimalTemplateJson = {
  templateDimensions: [1000, 800],
  bubbleDimensions: [20, 20],
  fieldBlocks: {},
  preProcessors: [],
};

/**
 * Minimal evaluation JSON fixture
 */
export const minimalEvaluationJson = {
  sourceType: 'local',
  options: {
    questionsInOrder: ['q1', 'q2'],
    answersInOrder: ['A', 'B'],
  },
  markingSchemes: {
    DEFAULT: {
      correct: 4,
      incorrect: -1,
      unmarked: 0,
    },
  },
};

// Add more fixtures as needed
`;

  await fs.writeFile(fixturesPath, fixtureContent);
  console.log(`   ✓ fixtures.ts (${fixtures.size} fixtures)\n`);
}

/**
 * Main execution
 */
async function main() {
  console.log('🧪 OMRChecker Test Scaffolding Generator');
  console.log('=========================================\n');

  // Ensure target directory exists
  await fs.mkdir(CONFIG.targetRoot, { recursive: true });

  console.log('📁 Scanning pytest files...');
  await walkDirectory(CONFIG.pythonTestRoot, CONFIG.targetRoot);

  console.log(`\n📝 Generating fixtures...`);
  await generateFixtures();

  // Summary
  console.log('✅ Test scaffolding complete!');
  console.log(`\n📊 Summary:`);
  console.log(`   ${testFiles.length} test files converted`);
  console.log(`   ${fixtures.size} fixtures extracted`);
  console.log(`\n📂 Output: ${CONFIG.targetRoot}\n`);

  console.log('⚠️  Note: Test bodies are placeholders - implement manually');
  console.log('   Use migration skill for reference:\n');
  console.log('   .agents/skills/omrchecker-migration-skill/modules/foundation/testing.md\n');

  console.log('Next step: Run file migration');
  console.log('  export ANTHROPIC_API_KEY="your-key"');
  console.log('  node 4-migrate-files.js\n');
}

// Run
main().catch(console.error);
