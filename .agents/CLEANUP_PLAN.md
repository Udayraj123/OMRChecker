# .agents Folder Cleanup Plan

**Created**: 2026-02-28  
**Reason**: Reduce context window bloat from 10.9MB to ~100KB

---

## Current State

```
.agents/ (10.9MB total)
├── migration-toolkit/       7.7MB  ⚠️ BLOAT (node_modules)
├── skills/                  3.2MB  ⚠️ BLOAT (old skill with 162 files)
├── AGENTS.md                  12KB ✅ KEEP
├── TEST_MIGRATION_*            8KB ✅ KEEP (test files)
└── polyfills/                  8KB ❓ CHECK
```

---

## Proposed Cleanup

### 1. Remove Old Migration Skill (3.2MB → 0)
**Target**: `.agents/skills/omrchecker-migration-skill/`  
**Reason**: Old migration approach (162 files, 3.2MB)  
**Replacement**: New skill at `.agents/skills/python-to-typescript-migration/` (24KB, 3 files)

**Action**:
```bash
rm -rf .agents/skills/omrchecker-migration-skill/
```

### 2. Remove Migration Toolkit node_modules (7.6MB → 0)
**Target**: `.agents/migration-toolkit/node_modules/`  
**Reason**: Dependencies bloat (7.6MB of npm packages)  
**Keep**: JavaScript/shell scripts (128KB), docs (60KB)

**Action**:
```bash
rm -rf .agents/migration-toolkit/node_modules/
rm -f .agents/migration-toolkit/package-lock.json
```

**Note**: Add to .gitignore if not already:
```
.agents/migration-toolkit/node_modules/
.agents/migration-toolkit/package-lock.json
```

### 3. Evaluate Other Skills
**Keep**:
- `.agents/skills/python-to-typescript-migration/` (24KB, NEW, production-ready)
- `.agents/skills/init-python/` (32KB, if actively used)
- `.agents/skills/repo-skill/` (check if used)

**Action**: Check if init-python and repo-skill are actively used or obsolete

### 4. Evaluate Polyfills (8KB)
**Target**: `.agents/polyfills/`  
**Action**: Check if needed for current workflow

---

## Expected Results

### Before Cleanup:
```
10.9MB total
├── 7.7MB migration-toolkit (mainly node_modules)
├── 3.2MB old omrchecker-migration-skill
├── 24KB python-to-typescript-migration (NEW)
```

### After Cleanup:
```
~100KB total
├── 60KB migration-toolkit scripts/docs (no node_modules)
├── 24KB python-to-typescript-migration
├── 12KB AGENTS.md
├── 8KB test files
```

**Context Window Reduction**: 10.9MB → 100KB = **99% reduction**

---

## Safety Checks

Before removing anything:

1. ✅ Verify new skill is committed and working
2. ✅ Check if old skill is referenced anywhere
3. ✅ Ensure migration-toolkit can reinstall node_modules if needed
4. ✅ Backup if uncertain

---

## Implementation

```bash
# 1. Check git status
git status

# 2. Remove old skill (3.2MB)
rm -rf .agents/skills/omrchecker-migration-skill/

# 3. Remove node_modules (7.6MB)
rm -rf .agents/migration-toolkit/node_modules/
rm -f .agents/migration-toolkit/package-lock.json

# 4. Update .gitignore
echo ".agents/migration-toolkit/node_modules/" >> .gitignore
echo ".agents/migration-toolkit/package-lock.json" >> .gitignore

# 5. Verify size reduction
du -sh .agents/

# 6. Commit cleanup
git add -A
git commit -m "chore: cleanup .agents folder to reduce context bloat

Removed:
- Old omrchecker-migration-skill (3.2MB, 162 files)
- migration-toolkit node_modules (7.6MB)

Kept:
- python-to-typescript-migration skill (24KB, production-ready)
- Migration toolkit scripts (60KB)
- Test files

Result: 10.9MB → ~100KB (99% reduction)
"
```

---

## Rollback Plan

If anything breaks:
```bash
# Restore from git
git checkout HEAD -- .agents/skills/omrchecker-migration-skill/

# Reinstall node_modules if needed
cd .agents/migration-toolkit/
npm install
```

---

**Status**: Ready for execution  
**Risk**: LOW (old files, new skill is working)  
**Benefit**: HIGH (99% context reduction)
