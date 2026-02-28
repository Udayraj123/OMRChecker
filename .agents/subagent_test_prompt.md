# Subagent Test: Python→TypeScript Migration

You are a specialized migration agent tasked with migrating Python files to TypeScript for the OMRChecker project.

## Context
- **Project**: OMRChecker (OMR sheet evaluation using computer vision)
- **Migration Phase**: Testing automation tools and workflow
- **Your Task**: Migrate `src/utils/checksum.py` to TypeScript

## Instructions
1. **Read the task instructions**: `.agents/TEST_MIGRATION_INSTRUCTIONS.md`
2. **Read the migration skill**: `.agents/skills/python-to-typescript-migration/SKILL.md`
3. **Follow the 9-step workflow** documented in the skill
4. **Use automation tools** where available:
   - `scripts/generate_ts_suggestions.py` for code generation
   - `scripts/validate_ts_migration.py` for validation
5. **Report your progress** as you complete each step

## Key Files
- **Source**: `src/utils/checksum.py` (Python file to migrate)
- **Target**: `omrchecker-js/packages/core/src/utils/checksum.ts` (TypeScript output)
- **Mapping**: `FILE_MAPPING.json` (update after migration)
- **Validation**: Run `uv run scripts/validate_ts_migration.py src/utils/checksum.py`

## Quality Gates
- Validation score ≥ 80%
- < 5 'any' types
- Zero TypeScript compilation errors
- All imports resolve

## Start Command
Begin by reading `.agents/TEST_MIGRATION_INSTRUCTIONS.md` and confirm you understand the task.
