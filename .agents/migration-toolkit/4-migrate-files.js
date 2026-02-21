#!/usr/bin/env node

/**
 * OMRChecker Migration Toolkit - File Migration
 * Uses Claude API to translate Python files to TypeScript
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import Anthropic from '@anthropic-ai/sdk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
  anthropicApiKey: process.env.ANTHROPIC_API_KEY,
  model: 'claude-sonnet-4.5',
  maxConcurrent: 3,
  sourceRoot: path.join(__dirname, '../../src'),
  targetRoot: path.join(__dirname, '../../omrchecker-js/src'),
  migrationSkillPath: path.join(__dirname, '../skills/omrchecker-migration-skill'),
  skipPatterns: [
    /\/tests\//,
    /\/__pycache__\//,
    /\.pyc$/,
    /conftest\.py$/,
    // Skip ML model files (create dummy placeholders instead)
    /ml_addons/,
    /yolo/,
    /ocr/,
  ],
  progressFile: path.join(__dirname, 'migration-progress.json'),
  logFile: path.join(__dirname, 'migration-log.json'),
};

// State
let anthropic;
let progress = {
  completed: [],
  failed: [],
  skipped: [],
  totalFiles: 0,
  startTime: null,
};

const migrationLog = [];

/**
 * Load progress from previous run
 */
async function loadProgress() {
  try {
    const data = await fs.readFile(CONFIG.progressFile, 'utf-8');
    progress = JSON.parse(data);
    console.log(`   Loaded progress: ${progress.completed.length} completed, ${progress.failed.length} failed\n`);
  } catch (err) {
    // No previous progress, start fresh
    progress.startTime = new Date().toISOString();
  }
}

/**
 * Save progress
 */
async function saveProgress() {
  await fs.writeFile(CONFIG.progressFile, JSON.stringify(progress, null, 2));
}

/**
 * Save migration log
 */
async function saveMigrationLog() {
  await fs.writeFile(CONFIG.logFile, JSON.stringify(migrationLog, null, 2));
}

/**
 * Discover Python files to migrate
 */
async function discoverPythonFiles(dir, files = []) {
  const entries = await fs.readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      // Skip special directories
      if (entry.name === '__pycache__' || entry.name === 'venv' || entry.name === '.git') {
        continue;
      }
      await discoverPythonFiles(fullPath, files);
    } else if (entry.isFile() && entry.name.endsWith('.py')) {
      // Check if should skip
      const shouldSkip = CONFIG.skipPatterns.some(pattern => pattern.test(fullPath));

      if (!shouldSkip) {
        files.push(fullPath);
      }
    }
  }

  return files;
}

/**
 * Load migration skill documentation
 */
async function loadMigrationSkill() {
  const skillDocs = {
    boundaries: '',
    quickRef: '',
    errorHandling: '',
    logging: '',
    configuration: '',
    validation: '',
  };

  try {
    // Load key foundation docs
    skillDocs.boundaries = await fs.readFile(
      path.join(CONFIG.migrationSkillPath, 'core/boundaries.md'),
      'utf-8'
    );

    skillDocs.quickRef = await fs.readFile(
      path.join(CONFIG.migrationSkillPath, 'core/quick-ref.md'),
      'utf-8'
    );

    skillDocs.errorHandling = await fs.readFile(
      path.join(CONFIG.migrationSkillPath, 'modules/foundation/error-handling.md'),
      'utf-8'
    );

    skillDocs.logging = await fs.readFile(
      path.join(CONFIG.migrationSkillPath, 'modules/foundation/logging.md'),
      'utf-8'
    );

    skillDocs.configuration = await fs.readFile(
      path.join(CONFIG.migrationSkillPath, 'modules/foundation/configuration.md'),
      'utf-8'
    );

    skillDocs.validation = await fs.readFile(
      path.join(CONFIG.migrationSkillPath, 'modules/foundation/validation.md'),
      'utf-8'
    );
  } catch (err) {
    console.warn(`   ⚠️  Could not load all migration skill docs: ${err.message}`);
  }

  return skillDocs;
}

/**
 * Build migration context for Claude
 */
function buildMigrationContext(pythonFilePath, pythonCode, skillDocs) {
  const relativePath = path.relative(CONFIG.sourceRoot, pythonFilePath);

  return `You are migrating OMRChecker from Python to TypeScript for browser/web environment.

**File to migrate**: ${relativePath}

**Migration Context**:
- Target: Browser/Web (client-side JavaScript/TypeScript)
- OpenCV (cv2) → OpenCV.js (WASM) - API is nearly identical! Main differences: memory management (.delete() calls), async loading
- NumPy → TypedArrays + custom helpers
- Pydantic → Zod validation
- File I/O → Browser File API
- ThreadPoolExecutor → Web Workers
- Rich terminal → Canvas/HTML rendering
- pytest → Vitest
- ML Models (YOLO, OCR, Barcode) → Create dummy placeholders (will be implemented later)

**Python → TypeScript Mappings**:
${skillDocs.quickRef}

**Boundaries** (What to migrate, what to skip):
${skillDocs.boundaries}

**Error Handling**:
${skillDocs.errorHandling}

**Logging**:
${skillDocs.logging}

**Configuration**:
${skillDocs.configuration}

**Validation**:
${skillDocs.validation}

**Instructions**:
1. Translate the Python code to TypeScript
2. Use browser-compatible equivalents for all operations
3. Preserve all business logic and edge cases
4. Add TypeScript type annotations
5. Convert snake_case to camelCase for variables/functions
6. Keep class names in PascalCase
7. Add JSDoc comments for complex logic
8. Use ES modules (import/export)
9. Handle async operations with Promises/async-await

**OpenCV.js Translation**:
- cv2.imread() → cv.imread()
- cv2.threshold() → cv.threshold()
- Keep function names identical, just change cv2 → cv
- Add .delete() calls for Mat objects to free memory
- Wrap in try/finally for cleanup

**ML Models**:
- For YOLO, OCR, Barcode detection: Create dummy functions that return empty results
- Add comment: // TODO: Implement ML model (optional feature)
- No TensorFlow.js/Tesseract.js code needed

**Web Workers**:
- ThreadPoolExecutor → Add TODO comment for Web Worker implementation
- Keep synchronous for now, can parallelize later

**Python Code**:
\`\`\`python
${pythonCode}
\`\`\`

Output ONLY the TypeScript code without explanations. Use proper TypeScript syntax and browser-compatible patterns.`;
}

/**
 * Migrate a single Python file using Claude API
 */
async function migrateFile(pythonFilePath, skillDocs) {
  const relativePath = path.relative(CONFIG.sourceRoot, pythonFilePath);

  // Check if already completed
  if (progress.completed.includes(relativePath)) {
    return { status: 'skipped', reason: 'already completed' };
  }

  try {
    console.log(`   🔄 ${relativePath}`);

    // Read Python file
    const pythonCode = await fs.readFile(pythonFilePath, 'utf-8');

    // Build migration prompt
    const migrationContext = buildMigrationContext(pythonFilePath, pythonCode, skillDocs);

    // Call Claude API
    const response = await anthropic.messages.create({
      model: CONFIG.model,
      max_tokens: 8000,
      messages: [
        {
          role: 'user',
          content: migrationContext,
        },
      ],
    });

    const typescriptCode = response.content[0].text;

    // Determine target path
    const targetPath = path.join(
      CONFIG.targetRoot,
      relativePath.replace(/\.py$/, '.ts')
    );

    // Ensure target directory exists
    await fs.mkdir(path.dirname(targetPath), { recursive: true });

    // Write TypeScript file
    await fs.writeFile(targetPath, typescriptCode);

    // Update progress
    progress.completed.push(relativePath);
    await saveProgress();

    // Log
    migrationLog.push({
      file: relativePath,
      status: 'success',
      timestamp: new Date().toISOString(),
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    });

    console.log(`      ✓ ${relativePath.replace(/\.py$/, '.ts')}`);

    return { status: 'success', targetPath };
  } catch (err) {
    console.error(`      ✗ Failed: ${err.message}`);

    progress.failed.push({ file: relativePath, error: err.message });
    await saveProgress();

    migrationLog.push({
      file: relativePath,
      status: 'failed',
      error: err.message,
      timestamp: new Date().toISOString(),
    });

    return { status: 'failed', error: err.message };
  }
}

/**
 * Migrate files in parallel with concurrency limit
 */
async function migrateFilesParallel(files, skillDocs) {
  const results = [];
  const queue = [...files];

  async function worker() {
    while (queue.length > 0) {
      const file = queue.shift();
      if (file) {
        const result = await migrateFile(file, skillDocs);
        results.push({ file, ...result });

        // Save log after each file
        await saveMigrationLog();
      }
    }
  }

  // Create worker pool
  const workers = Array(CONFIG.maxConcurrent).fill(null).map(() => worker());
  await Promise.all(workers);

  return results;
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 OMRChecker File Migration');
  console.log('============================\n');

  // Check API key
  if (!CONFIG.anthropicApiKey) {
    console.error('❌ Error: ANTHROPIC_API_KEY environment variable not set');
    console.error('   Set it with: export ANTHROPIC_API_KEY="your-key-here"\n');
    process.exit(1);
  }

  // Initialize Anthropic client
  anthropic = new Anthropic({
    apiKey: CONFIG.anthropicApiKey,
  });

  console.log('📋 Configuration:');
  console.log(`   Model: ${CONFIG.model}`);
  console.log(`   Max Concurrent: ${CONFIG.maxConcurrent}`);
  console.log(`   Source: ${CONFIG.sourceRoot}`);
  console.log(`   Target: ${CONFIG.targetRoot}\n`);

  // Load progress
  console.log('📂 Loading progress...');
  await loadProgress();

  // Load migration skill
  console.log('📚 Loading migration skill documentation...');
  const skillDocs = await loadMigrationSkill();
  console.log('   ✓ Migration skill loaded\n');

  // Discover Python files
  console.log('🔍 Discovering Python files...');
  const pythonFiles = await discoverPythonFiles(CONFIG.sourceRoot);
  progress.totalFiles = pythonFiles.length;

  console.log(`   Found ${pythonFiles.length} Python files to migrate\n`);

  // Filter out already completed
  const remainingFiles = pythonFiles.filter(file => {
    const relativePath = path.relative(CONFIG.sourceRoot, file);
    return !progress.completed.includes(relativePath);
  });

  console.log(`   ${remainingFiles.length} files remaining\n`);

  if (remainingFiles.length === 0) {
    console.log('✅ All files already migrated!\n');
    return;
  }

  // Start migration
  console.log(`🔄 Starting migration (${CONFIG.maxConcurrent} concurrent workers)...\n`);

  const startTime = Date.now();
  await migrateFilesParallel(remainingFiles, skillDocs);
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);

  // Final summary
  console.log('\n✅ Migration complete!');
  console.log(`\n📊 Summary:`);
  console.log(`   Total files: ${progress.totalFiles}`);
  console.log(`   Completed: ${progress.completed.length}`);
  console.log(`   Failed: ${progress.failed.length}`);
  console.log(`   Duration: ${duration}s`);
  console.log(`   Rate: ${(progress.completed.length / (duration / 60)).toFixed(1)} files/min`);

  if (progress.failed.length > 0) {
    console.log(`\n❌ Failed files:`);
    progress.failed.forEach(({ file, error }) => {
      console.log(`   - ${file}: ${error}`);
    });
  }

  console.log(`\n📂 Output: ${CONFIG.targetRoot}`);
  console.log(`📄 Log: ${CONFIG.logFile}`);
  console.log(`📄 Progress: ${CONFIG.progressFile}\n`);

  // Calculate total tokens used
  const totalInputTokens = migrationLog.reduce((sum, log) => sum + (log.inputTokens || 0), 0);
  const totalOutputTokens = migrationLog.reduce((sum, log) => sum + (log.outputTokens || 0), 0);

  console.log(`📊 API Usage:`);
  console.log(`   Input tokens: ${totalInputTokens.toLocaleString()}`);
  console.log(`   Output tokens: ${totalOutputTokens.toLocaleString()}`);
  console.log(`   Total tokens: ${(totalInputTokens + totalOutputTokens).toLocaleString()}\n`);

  console.log('⚠️  Next steps:');
  console.log('   1. Review generated TypeScript files');
  console.log('   2. Implement TODO comments (complex logic)');
  console.log('   3. Test compilation: cd omrchecker-js && npm run type-check');
  console.log('   4. Run tests: npm test');
  console.log('   5. Implement complex operations (OpenCV.js, Web Workers, ML models)\n');
}

// Run
main().catch(console.error);
