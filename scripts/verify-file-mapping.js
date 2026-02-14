#!/usr/bin/env node
/**
 * Verify FILE_MAPPING.json: check that all mapped Python and TypeScript files exist.
 * Run from repo root: node scripts/verify-file-mapping.js
 */

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const mappingPath = path.join(root, 'FILE_MAPPING.json');
const data = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));

const results = {
  pythonMissing: [],
  typescriptMissing: [],
  pythonExists: 0,
  typescriptExists: 0,
  typescriptSkipped: [], // N/A, REMOVED, etc.
  pythonSkipped: [],    // entries with no python path (TS-only)
};

for (const m of data.mappings) {
  const py = m.python;
  const ts = m.typescript;
  const status = m.status || '';

  // Check Python path (if it's a real path, not empty)
  if (py && typeof py === 'string' && py.startsWith('src/')) {
    const pyPath = path.join(root, py);
    if (fs.existsSync(pyPath)) {
      results.pythonExists++;
    } else {
      results.pythonMissing.push({ python: py, status, expected: status === 'not_started' });
    }
  } else if (py && typeof py === 'string') {
    results.pythonSkipped.push(py);
  }

  // Check TypeScript path (only if it's a real path)
  if (ts && typeof ts === 'string' && ts.startsWith('omrchecker-js/') && !ts.includes('REMOVED') && !ts.startsWith('N/A')) {
    const tsPaths = ts.split(/\s+or\s+/).map((p) => p.trim());
    let found = false;
    for (const tp of tsPaths) {
      const tsPath = path.join(root, tp);
      if (fs.existsSync(tsPath)) {
        results.typescriptExists++;
        found = true;
        break;
      }
    }
    if (!found) {
      results.typescriptMissing.push({ typescript: ts, status, expected: status === 'not_started' });
    }
  } else if (ts && (ts.includes('N/A') || ts.includes('REMOVED'))) {
    results.typescriptSkipped.push(ts.substring(0, 60) + (ts.length > 60 ? '...' : ''));
  }
}

console.log('=== FILE_MAPPING.json verification ===\n');
console.log('Python files (mapped with path src/...):');
console.log('  Exist:', results.pythonExists);
if (results.pythonMissing.length) {
  const expected = results.pythonMissing.filter((x) => x.expected);
  const unexpected = results.pythonMissing.filter((x) => !x.expected);
  console.log('  MISSING:', results.pythonMissing.length, '(expected not_started:', expected.length, ', mapping/structure issues:', unexpected.length, ')');
  unexpected.forEach(({ python, status }) => console.log('    [CHECK]', python, '(', status, ')'));
  expected.forEach(({ python, status }) => console.log('    [OK]', python, '(', status, ')'));
}

console.log('\nTypeScript files (mapped with path omrchecker-js/...):');
console.log('  Exist:', results.typescriptExists);
if (results.typescriptMissing.length) {
  const expected = results.typescriptMissing.filter((x) => x.expected);
  const unexpected = results.typescriptMissing.filter((x) => !x.expected);
  console.log('  MISSING:', results.typescriptMissing.length, '(expected not_started:', expected.length, ', mapping/structure issues:', unexpected.length, ')');
  unexpected.forEach(({ typescript, status }) => console.log('    [CHECK]', typescript, '(', status, ')'));
  expected.forEach(({ typescript, status }) => console.log('    [OK]', typescript, '(', status, ')'));
}

console.log('\nSummary:');
console.log('  Mappings with Python path checked:', results.pythonExists + results.pythonMissing.length);
console.log('  Mappings with TypeScript path checked:', results.typescriptExists + results.typescriptMissing.length);
console.log('  TypeScript entries N/A or REMOVED (not checked):', results.typescriptSkipped.length);

const hasUnexpected =
  results.pythonMissing.some((x) => !x.expected) || results.typescriptMissing.some((x) => !x.expected);
process.exit(hasUnexpected ? 1 : 0);
