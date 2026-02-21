#!/usr/bin/env node

/**
 * OMRChecker Migration Toolkit - Interface Generator
 * Scans Python dataclasses/Pydantic models and generates TypeScript interfaces
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
  pythonSrcRoot: path.join(__dirname, '../../src'),
  targetRoot: path.join(__dirname, '../../omrchecker-js/src/types'),
  migrationSkillPath: path.join(__dirname, '../skills/omrchecker-migration-skill'),
};

// Type mappings Python → TypeScript
const TYPE_MAPPINGS = {
  'str': 'string',
  'int': 'number',
  'float': 'number',
  'bool': 'boolean',
  'Path': 'string',
  'dict': 'Record',
  'list': 'Array',
  'tuple': 'readonly',
  'None': 'null',
  'Any': 'any',
  'Optional': 'undefined',
};

// Discovered interfaces
const interfaces = [];
const zodSchemas = [];
const errorClasses = [];

/**
 * Scan Python file for dataclasses and Pydantic models
 */
async function scanPythonFile(filePath) {
  const content = await fs.readFile(filePath, 'utf-8');
  const relativePath = path.relative(CONFIG.pythonSrcRoot, filePath);

  // Find all dataclasses
  const dataclassRegex = /@dataclass\s+class\s+(\w+):\s*([\s\S]*?)(?=\n(?:class|def|@|\Z))/g;
  let match;

  while ((match = dataclassRegex.exec(content)) !== null) {
    const [, className, classBody] = match;
    const fields = parseDataclassFields(classBody);

    interfaces.push({
      name: className,
      fields,
      source: relativePath,
      type: 'dataclass',
    });
  }

  // Find Pydantic models
  const pydanticRegex = /class\s+(\w+)\(BaseModel\):\s*([\s\S]*?)(?=\nclass |\Z)/g;

  while ((match = pydanticRegex.exec(content)) !== null) {
    const [, className, classBody] = match;
    const fields = parseDataclassFields(classBody);

    zodSchemas.push({
      name: className,
      fields,
      source: relativePath,
      type: 'pydantic',
    });
  }

  // Find exception classes
  const exceptionRegex = /class\s+(\w+)\((\w*Exception|OMRCheckerError)\):/g;

  while ((match = exceptionRegex.exec(content)) !== null) {
    const [, className, baseClass] = match;

    errorClasses.push({
      name: className,
      baseClass,
      source: relativePath,
    });
  }
}

/**
 * Parse dataclass fields
 */
function parseDataclassFields(classBody) {
  const fields = [];
  const fieldRegex = /^\s*(\w+):\s*([^=\n]+)(?:\s*=\s*(.+))?$/gm;
  let match;

  while ((match = fieldRegex.exec(classBody)) !== null) {
    const [, name, typeAnnotation, defaultValue] = match;

    // Skip if it's a method or property
    if (typeAnnotation.includes('(') || name.startsWith('_')) continue;

    const tsType = pythonTypeToTypeScript(typeAnnotation.trim());
    const optional = defaultValue !== undefined || typeAnnotation.includes('Optional') || typeAnnotation.includes('|') && typeAnnotation.includes('None');

    fields.push({
      name: toCamelCase(name),
      pythonName: name,
      type: tsType,
      optional,
      defaultValue: defaultValue?.trim(),
    });
  }

  return fields;
}

/**
 * Convert Python type annotation to TypeScript
 */
function pythonTypeToTypeScript(pythonType) {
  // Handle basic types
  for (const [pyType, tsType] of Object.entries(TYPE_MAPPINGS)) {
    if (pythonType === pyType) return tsType;
  }

  // Handle Optional[T]
  if (pythonType.startsWith('Optional[')) {
    const innerType = pythonType.slice(9, -1);
    return `${pythonTypeToTypeScript(innerType)} | null`;
  }

  // Handle Union[T1, T2, ...]
  if (pythonType.startsWith('Union[')) {
    const types = pythonType.slice(6, -1).split(',').map(t => pythonTypeToTypeScript(t.trim()));
    return types.join(' | ');
  }

  // Handle list[T]
  if (pythonType.startsWith('list[') || pythonType.startsWith('List[')) {
    const innerType = pythonType.match(/\[(.+)\]/)[1];
    return `${pythonTypeToTypeScript(innerType)}[]`;
  }

  // Handle dict[K, V]
  if (pythonType.startsWith('dict[') || pythonType.startsWith('Dict[')) {
    const [keyType, valueType] = pythonType.match(/\[(.+),\s*(.+)\]/);
    if (keyType && valueType) {
      return `Record<${pythonTypeToTypeScript(keyType)}, ${pythonTypeToTypeScript(valueType)}>`;
    }
    return 'Record<string, any>';
  }

  // Handle tuple[T1, T2, ...]
  if (pythonType.startsWith('tuple[') || pythonType.startsWith('Tuple[')) {
    const types = pythonType.match(/\[(.+)\]/)[1].split(',').map(t => pythonTypeToTypeScript(t.trim()));
    return `[${types.join(', ')}]`;
  }

  // Handle | syntax (Python 3.10+)
  if (pythonType.includes('|')) {
    const types = pythonType.split('|').map(t => pythonTypeToTypeScript(t.trim()));
    return types.join(' | ');
  }

  // Default: assume it's a custom type (keep as-is)
  return pythonType;
}

/**
 * Convert snake_case to camelCase
 */
function toCamelCase(snakeCase) {
  return snakeCase.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Generate TypeScript interface
 */
function generateInterface(interfaceData) {
  const { name, fields, source } = interfaceData;

  const fieldLines = fields.map(field => {
    const optional = field.optional ? '?' : '';
    const comment = field.pythonName !== field.name ? `  // Python: ${field.pythonName}` : '';
    return `  ${field.name}${optional}: ${field.type};${comment}`;
  }).join('\n');

  return `/**
 * ${name}
 * Source: ${source}
 */
export interface ${name} {
${fieldLines}
}
`;
}

/**
 * Generate Zod schema
 */
function generateZodSchema(schemaData) {
  const { name, fields, source } = schemaData;

  const fieldLines = fields.map(field => {
    let zodType = pythonTypeToZod(field.type);
    if (field.optional) {
      zodType += '.optional()';
    }
    return `  ${field.name}: ${zodType},`;
  }).join('\n');

  return `/**
 * ${name} Schema
 * Source: ${source}
 */
export const ${name}Schema = z.object({
${fieldLines}
});

export type ${name} = z.infer<typeof ${name}Schema>;
`;
}

/**
 * Convert TypeScript type to Zod schema type
 */
function pythonTypeToZod(tsType) {
  if (tsType === 'string') return 'z.string()';
  if (tsType === 'number') return 'z.number()';
  if (tsType === 'boolean') return 'z.boolean()';
  if (tsType === 'null') return 'z.null()';
  if (tsType === 'any') return 'z.any()';
  if (tsType.endsWith('[]')) {
    const innerType = tsType.slice(0, -2);
    return `z.array(${pythonTypeToZod(innerType)})`;
  }
  if (tsType.startsWith('Record<')) {
    const match = tsType.match(/Record<(\w+),\s*(.+)>/);
    if (match) {
      const [, keyType, valueType] = match;
      return `z.record(${pythonTypeToZod(keyType)}, ${pythonTypeToZod(valueType)})`;
    }
  }
  if (tsType.includes('|')) {
    const types = tsType.split('|').map(t => pythonTypeToZod(t.trim()));
    return `z.union([${types.join(', ')}])`;
  }
  // Assume custom type
  return `${tsType}Schema`;
}

/**
 * Generate Error class
 */
function generateErrorClass(errorData) {
  const { name, baseClass, source } = errorData;

  return `/**
 * ${name}
 * Source: ${source}
 */
export class ${name} extends ${baseClass === 'OMRCheckerError' ? 'OMRCheckerError' : 'Error'} {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = '${name}';
  }
}
`;
}

/**
 * Walk directory tree and scan Python files
 */
async function walkDirectory(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      // Skip __pycache__, tests, venv
      if (entry.name === '__pycache__' || entry.name === 'tests' || entry.name === 'venv') {
        continue;
      }
      await walkDirectory(fullPath);
    } else if (entry.isFile() && entry.name.endsWith('.py')) {
      await scanPythonFile(fullPath);
    }
  }
}

/**
 * Main execution
 */
async function main() {
  console.log('🔍 OMRChecker Interface Generator');
  console.log('==================================\n');

  // Ensure target directory exists
  await fs.mkdir(CONFIG.targetRoot, { recursive: true });

  console.log('📁 Scanning Python source files...');
  await walkDirectory(CONFIG.pythonSrcRoot);

  console.log(`   Found ${interfaces.length} dataclasses`);
  console.log(`   Found ${zodSchemas.length} Pydantic models`);
  console.log(`   Found ${errorClasses.length} exception classes\n`);

  // Group interfaces by category
  const coreInterfaces = interfaces.filter(i => i.source.startsWith('schemas/models/'));
  const processorInterfaces = interfaces.filter(i => i.source.includes('processors/'));
  const utilInterfaces = interfaces.filter(i => !coreInterfaces.includes(i) && !processorInterfaces.includes(i));

  // Generate core interfaces
  if (coreInterfaces.length > 0) {
    console.log('📝 Generating core interfaces...');
    const content = [
      '// Auto-generated from Python dataclasses',
      '// Do not edit manually - regenerate with 2-generate-interfaces.js\n',
      ...coreInterfaces.map(generateInterface),
    ].join('\n');

    await fs.writeFile(path.join(CONFIG.targetRoot, 'core.ts'), content);
    console.log(`   ✓ core.ts (${coreInterfaces.length} interfaces)\n`);
  }

  // Generate processor interfaces
  if (processorInterfaces.length > 0) {
    console.log('📝 Generating processor interfaces...');
    const content = [
      '// Auto-generated from Python dataclasses',
      '// Do not edit manually - regenerate with 2-generate-interfaces.js\n',
      ...processorInterfaces.map(generateInterface),
    ].join('\n');

    await fs.writeFile(path.join(CONFIG.targetRoot, 'processors.ts'), content);
    console.log(`   ✓ processors.ts (${processorInterfaces.length} interfaces)\n`);
  }

  // Generate Zod schemas
  if (zodSchemas.length > 0) {
    console.log('📝 Generating Zod schemas...');
    const content = [
      "import { z } from 'zod';\n",
      '// Auto-generated from Pydantic models',
      '// Do not edit manually - regenerate with 2-generate-interfaces.js\n',
      ...zodSchemas.map(generateZodSchema),
    ].join('\n');

    await fs.writeFile(path.join(CONFIG.targetRoot, 'schemas.ts'), content);
    console.log(`   ✓ schemas.ts (${zodSchemas.length} schemas)\n`);
  }

  // Generate error classes
  if (errorClasses.length > 0) {
    console.log('📝 Generating error classes...');

    // Base error class
    const baseErrorContent = `/**
 * Base OMRChecker Error
 * Source: src/exceptions.py
 */
export class OMRCheckerError extends Error {
  constructor(message: string, public context?: Record<string, any>) {
    super(message);
    this.name = 'OMRCheckerError';
  }
}

${errorClasses.map(generateErrorClass).join('\n')}
`;

    await fs.writeFile(path.join(CONFIG.targetRoot, 'errors.ts'), baseErrorContent);
    console.log(`   ✓ errors.ts (${errorClasses.length + 1} error classes)\n`);
  }

  // Generate index.ts
  console.log('📝 Generating index.ts...');
  const indexContent = `// Auto-generated type exports
// Do not edit manually - regenerate with 2-generate-interfaces.js

export * from './core';
export * from './processors';
export * from './schemas';
export * from './errors';
`;

  await fs.writeFile(path.join(CONFIG.targetRoot, 'index.ts'), indexContent);
  console.log('   ✓ index.ts\n');

  // Summary
  console.log('✅ Interface generation complete!');
  console.log(`\n📊 Summary:`);
  console.log(`   ${interfaces.length} TypeScript interfaces`);
  console.log(`   ${zodSchemas.length} Zod schemas`);
  console.log(`   ${errorClasses.length} error classes`);
  console.log(`\n📂 Output: ${CONFIG.targetRoot}\n`);

  console.log('Next step: Run test scaffolding generator');
  console.log('  node 3-generate-tests.js\n');
}

// Run
main().catch(console.error);
