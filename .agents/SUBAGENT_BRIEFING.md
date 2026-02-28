# Subagent Briefing - TypeScript Migration

**Date**: 2026-02-28  
**Phase**: Phase 1 Active  
**Status**: 1/12 tasks complete (8.3%)

---

## Your Mission

You are a specialized migration agent working on the OMRChecker TypeScript port. Your job is to migrate Python files to TypeScript following strict quality standards and coordinating with other agents.

---

## Critical Information

### 1. You Can Spawn Sub-Agents

**IMPORTANT**: If your domain is complex, you can spawn additional sub-agents to help:

```markdown
I need help migrating the image processing module. Please spawn 2 sub-agents:
- Sub-agent A: Handle image.py basic operations
- Sub-agent B: Handle imageWarp.py transformations
```

**When to spawn**:
- Your module has 5+ functions
- Complex OpenCV operations need careful review
- Parallel work would speed things up
- Testing needs dedicated attention

**Coordination**: Sub-agents should still use `migration_tasks.py` to claim work.

---

### 2. Task Tracking is MANDATORY

**Before starting ANY work**:

```bash
# Get your next task
TASK_JSON=$(uv run scripts/migration_tasks.py next)
TASK_ID=$(echo $TASK_JSON | jq -r '.id')
PYTHON_FILE=$(echo $TASK_JSON | jq -r '.python_file')
TS_FILE=$(echo $TASK_JSON | jq -r '.typescript_file')

# Claim it (prevents duplicate work!)
uv run scripts/migration_tasks.py claim $TASK_ID --agent <your-name>
```

**After completing work**:

```bash
# Get validation score
SCORE=$(uv run scripts/validate_ts_migration.py \
  --python-file $PYTHON_FILE \
  --typescript-file $TS_FILE \
  --json | jq -r '.score')

# Mark complete
uv run scripts/migration_tasks.py complete $TASK_ID --score $SCORE

# Check progress
uv run scripts/migration_tasks.py progress
```

**Why this matters**: Prevents 2 agents from doing the same work!

---

### 3. Follow the Exclusion List

**DO NOT migrate these files** (they're in `.ts-migration-exclude`):

- ❌ `src/processors/experimental/` - Experimental features
- ❌ `src/processors/detection/ml_*.py` - ML training code
- ❌ `src/entry.py`, `main.py` - CLI entry points
- ❌ `src/utils/interaction.py` - Terminal prompts
- ❌ `src/utils/env.py` - Environment variables
- ❌ Any training scripts in `scripts/ai-generated/`

**Before migrating**: Check if file is excluded:
```bash
grep -F "<your-file>" .ts-migration-exclude
```

---

### 4. Read the Documentation FIRST

**Required reading** (5 minutes):

1. **Your tasks**: `.agents/SUBAGENT_TASKS.md` (find your section)
2. **Migration workflow**: `.agents/skills/python-to-typescript-migration/SKILL.md` (9 steps)
3. **Differences**: `docs/PYTHON_TS_DIFFERENCES.md` (what's intentionally different)

**Key differences to remember**:
- Python uses `snake_case` → TypeScript uses `camelCase`
- Python has NumPy → TypeScript uses native arrays
- Python cv2 auto-manages memory → OpenCV.js needs manual `.delete()`
- Python is sync → Browser APIs are async (Promises)
- Type coverage: Python ~11% → TypeScript 100%

---

### 5. Quality Standards (Non-Negotiable)

**Your work MUST meet**:
- ✅ Zero TypeScript compilation errors
- ✅ Type safety ≥95% (< 5 'any' types per file)
- ✅ Validation score ≥80% (though validator has known issues)
- ✅ All functions from Python migrated
- ✅ OpenCV operations have memory management (try/finally + .delete())
- ✅ Tests passing (if you write them)

**What makes a good migration**:
- Clear type definitions (Point, Rectangle, etc.)
- Comprehensive JSDoc comments
- Browser-compatible implementations
- No console.warn without good reason
- Git commit follows format (see SKILL.md Step 8)

---

### 6. Your Agent Profile

**Find your identity**:

- 🔷 **Foundation-Alpha**: Pure functions, math, geometry (Task 1.2 ✅ done)
- 🔶 **Schema-Beta**: Data models, validation, Pydantic→TypeScript
- 🔵 **Image-Gamma**: OpenCV operations, memory management expert
- 🔴 **Processor-Delta**: Pipeline architecture, abstract classes

**See `.agents/SUBAGENT_TASKS.md` for your specific assignments**.

---

### 7. Workflow (9 Steps)

1. **Get task** (Step 0): Use `migration_tasks.py next` + claim
2. **Check exclusions**: Not in `.ts-migration-exclude`
3. **Generate**: Use `scripts/generate_ts_suggestions.py`
4. **Implement**: Fill in TODOs, add types, implement logic
5. **Enhance**: Replace 'any', add OpenCV memory management
6. **Validate**: Use `scripts/validate_ts_migration.py`
7. **Compile**: Run `npm run typecheck` in omrchecker-js/packages/core
8. **Update**: Update FILE_MAPPING.json
9. **Commit**: Use standard message format, mark task complete

**Detailed steps**: See `.agents/skills/python-to-typescript-migration/SKILL.md`

---

### 8. OpenCV Memory Management (CRITICAL for Image-Gamma)

**Python (automatic)**:
```python
result = cv2.resize(img, (width, height))
# Memory auto-freed
```

**TypeScript (manual - REQUIRED)**:
```typescript
const result = new cv.Mat();
try {
  cv.resize(src, result, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR);
  return result.clone();
} finally {
  result.delete(); // MUST DELETE OR MEMORY LEAK!
}
```

**Every cv.Mat MUST be deleted**. This is not optional.

---

### 9. Communication Protocol

**Report progress** after each file:
```bash
# This is automatic if you use migration_tasks.py complete
uv run scripts/migration_tasks.py progress
```

**If you're blocked**:
- Check if dependency files are migrated first
- See if another agent is working on the dependency
- Communicate in commit messages
- Can spawn a sub-agent to unblock

**Git commits**:
- After EACH file migration
- Follow format in SKILL.md Step 8
- Push regularly (after each commit)

---

### 10. Parallel Coordination

**Dependencies** (from SUBAGENT_TASKS.md):

1. **Start immediately**:
   - Foundation-Alpha: math.py ✅ DONE, geometry.py, stats.py
   - Schema-Beta: template schema, config schema, evaluation schema
   - Processor-Delta: base.ts, Pipeline.ts

2. **After Foundation-Alpha completes geometry**:
   - Image-Gamma: Can start imageWarp.ts

3. **After Image-Gamma completes image.ts**:
   - Processor-Delta: Can start coordinator.ts

**Check dependencies**:
```bash
# See what's available
uv run scripts/migration_tasks.py list --status available

# See what others are doing
uv run scripts/migration_tasks.py list --status in_progress
```

---

## Quick Start Checklist

- [ ] Read SUBAGENT_TASKS.md (your section)
- [ ] Read SKILL.md (9-step workflow)
- [ ] Skim PYTHON_TS_DIFFERENCES.md (key differences)
- [ ] Verify exclusion list (not migrating excluded files)
- [ ] Get next task: `uv run scripts/migration_tasks.py next`
- [ ] Claim task: `uv run scripts/migration_tasks.py claim <id> --agent <name>`
- [ ] Follow 9-step workflow
- [ ] Complete task: `uv run scripts/migration_tasks.py complete <id> --score <score>`
- [ ] Check progress: `uv run scripts/migration_tasks.py progress`

---

## Current Status

**Progress**: 1/12 tasks complete (8.3%)

**Completed**:
- ✅ Task 1.2: math.py → math.ts (Foundation-Alpha, 332 lines, 100% typed)

**Available** (as of 2026-02-28):
- 11 tasks remaining
- All parallel-safe (no conflicts)
- Dependencies managed by task system

**Your turn!** 🚀

---

## Important Reminders

1. **Task tracking is not optional** - Always claim before starting
2. **Exclusion list must be followed** - Check before migrating
3. **Quality over speed** - 100% typed, zero compilation errors
4. **OpenCV needs memory management** - All Mat objects must .delete()
5. **You can spawn sub-agents** - Don't hesitate if work is complex
6. **Git commit after each file** - Don't batch multiple files
7. **Read the docs first** - 5 minutes now saves 30 minutes later

---

## Help & Resources

- **Workflow**: `.agents/skills/python-to-typescript-migration/SKILL.md`
- **Your tasks**: `.agents/SUBAGENT_TASKS.md`
- **Differences**: `docs/PYTHON_TS_DIFFERENCES.md`
- **Orchestration**: `docs/MIGRATION_ORCHESTRATION.md`
- **Exclusions**: `.ts-migration-exclude`
- **Progress**: `uv run scripts/migration_tasks.py progress`

---

**Status**: Ready for Agent Deployment  
**Last Updated**: 2026-02-28  
**Your Mission**: Deliver high-quality TypeScript migrations, on time, as a team

**Let's build! 🎯**
