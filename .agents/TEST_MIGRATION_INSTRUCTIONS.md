# Test Migration Task: Subagent Delegation

## Objective
Test the Python→TypeScript migration workflow with a single file (`src/utils/checksum.py`) to validate our automation tools and agent skill.

## Your Task
Migrate `src/utils/checksum.py` to TypeScript following the complete workflow in `.agents/skills/python-to-typescript-migration/SKILL.md`.

## Expected Deliverables
1. TypeScript file: `omrchecker-js/packages/core/src/utils/checksum.ts`
2. Validation report showing ≥80% quality score
3. Updated FILE_MAPPING.json with new mapping entry
4. Git commit with migration changes

## Success Criteria
- ✅ Validation score ≥ 80%
- ✅ < 5 'any' types in generated TypeScript
- ✅ All imports resolve correctly
- ✅ TypeScript compiles without errors
- ✅ FILE_MAPPING.json updated with status "synced"

## Notes
- This is a simple utility file with 2 pure functions
- No dependencies on OpenCV or complex types
- Use Web Crypto API for hashing in browser environment
- Python uses hashlib, TypeScript should use SubtleCrypto

## How to Start
1. Read the migration skill: `.agents/skills/python-to-typescript-migration/SKILL.md`
2. Follow the 9-step workflow
3. Use automation scripts where possible
4. Report progress and any issues
