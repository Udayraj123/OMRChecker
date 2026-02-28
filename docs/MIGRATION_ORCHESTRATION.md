# Migration Task Orchestration

**Status**: Production Ready  
**Created**: 2026-02-28  
**Alternative to**: beads (no dolt dependency)

---

## Overview

Lightweight JSONL-based task orchestrator for managing parallel TypeScript migration work across multiple agents. No external dependencies (no dolt, no database).

**Storage**: `.migration-tasks.jsonl` (gitignored, local only)  
**Source**: `FILE_MAPPING.json` (source of truth)

---

## Quick Start

### 1. Initialize Tasks
```bash
# Create tasks for Phase 1
uv run scripts/migration_tasks.py init --phase 1

# Result: 12 tasks created
```

### 2. Agent Workflow
```bash
# Agent 1: Get next available task
TASK=$(uv run scripts/migration_tasks.py next | jq -r '.id')
PYTHON_FILE=$(uv run scripts/migration_tasks.py next | jq -r '.python_file')

# Agent 1: Claim the task
uv run scripts/migration_tasks.py claim $TASK --agent agent-1

# Agent 1: Do the migration
uv run scripts/generate_ts_suggestions.py --file $PYTHON_FILE --output...
# ... follow migration skill workflow ...

# Agent 1: Complete the task
uv run scripts/migration_tasks.py complete $TASK --score 85

# Meanwhile, Agent 2 can work on different task in parallel
```

### 3. Monitor Progress
```bash
# View all tasks
uv run scripts/migration_tasks.py list

# View available tasks only
uv run scripts/migration_tasks.py list --status available

# Show progress dashboard
uv run scripts/migration_tasks.py progress
```

---

## Commands

### `init` - Initialize Tasks
```bash
# Initialize all unmigrated tasks from FILE_MAPPING.json
uv run scripts/migration_tasks.py init

# Initialize only Phase 1 tasks
uv run scripts/migration_tasks.py init --phase 1

# Initialize specific phase
uv run scripts/migration_tasks.py init --phase 2
```

**What it does**:
- Reads `FILE_MAPPING.json`
- Creates tasks for entries with `status != "synced"`
- Skips N/A and already-migrated files
- Stores tasks in `.migration-tasks.jsonl`

### `next` - Get Next Available Task
```bash
# Get next task (JSON output)
uv run scripts/migration_tasks.py next

# Parse with jq
TASK_ID=$(uv run scripts/migration_tasks.py next | jq -r '.id')
PY_FILE=$(uv run scripts/migration_tasks.py next | jq -r '.python_file')
TS_FILE=$(uv run scripts/migration_tasks.py next | jq -r '.typescript_file')
```

**Output**: JSON object with task details
**Ordering**: Phase (ascending), then task ID

### `claim` - Claim a Task
```bash
# Claim task for agent
uv run scripts/migration_tasks.py claim task-001 --agent agent-1

# Output:
# ✅ Task task-001 claimed by agent-1
#    Python:  src/processors/detection/processor.py
#    TypeScript: ...
```

**Effect**: 
- Updates task status to `in_progress`
- Assigns agent name
- Other agents won't see this task in `next`

### `complete` - Complete a Task
```bash
# Mark complete without score
uv run scripts/migration_tasks.py complete task-001

# Mark complete with validation score
uv run scripts/migration_tasks.py complete task-001 --score 85
```

**Effect**:
- Updates status to `completed`
- Records validation score
- Sets completion timestamp
- Unblocks dependent tasks (if any)

### `list` - List Tasks
```bash
# List all tasks
uv run scripts/migration_tasks.py list

# List available tasks only
uv run scripts/migration_tasks.py list --status available

# List completed tasks
uv run scripts/migration_tasks.py list --status completed

# List Phase 1 tasks
uv run scripts/migration_tasks.py list --phase 1

# Combine filters
uv run scripts/migration_tasks.py list --status available --phase 1
```

**Output**: Table format
```
ID           Status       Phase  Agent        File
--------------------------------------------------------------------------------
task-001     available    1      -            processor.py
task-002     in_progress  1      agent-1      FeatureBasedAlignment.py
task-003     completed    1      agent-2      phase_correlation.py
```

### `progress` - Show Progress
```bash
uv run scripts/migration_tasks.py progress
```

**Output**:
```
=== Migration Progress ===
Total Tasks: 12
Completed:     3 ( 25.0%)
In Progress:   2
Available:     7
Blocked:       0

By Phase:
  Phase 1: 12
```

---

## Integration with Agent Skill

### Updated Agent Workflow

The migration skill (`.agents/skills/python-to-typescript-migration/SKILL.md`) should be updated to:

**Step 0: Get Task** (NEW)
```bash
# Get next available task
TASK_JSON=$(uv run scripts/migration_tasks.py next)
TASK_ID=$(echo $TASK_JSON | jq -r '.id')
PYTHON_FILE=$(echo $TASK_JSON | jq -r '.python_file')
TS_FILE=$(echo $TASK_JSON | jq -r '.typescript_file')

# Claim it
uv run scripts/migration_tasks.py claim $TASK_ID --agent $(whoami)
```

**Step 1-8**: Follow existing migration workflow

**Step 9: Complete Task** (UPDATED)
```bash
# Get validation score
SCORE=$(uv run scripts/validate_ts_migration.py \
  --python-file $PYTHON_FILE \
  --typescript-file $TS_FILE \
  --json | jq -r '.score')

# Mark complete
uv run scripts/migration_tasks.py complete $TASK_ID --score $SCORE

# Update FILE_MAPPING.json (as before)
# ...

# Commit (as before)
git commit -m "feat(ts-migrate): complete $TASK_ID"
```

---

## Benefits Over Beads

| Feature | Beads (with dolt) | migration_tasks.py |
|---------|-------------------|-------------------|
| **Dependencies** | dolt, dolt-server | None (pure Python) |
| **Storage** | Dolt database | .jsonl file |
| **Setup** | Complex (dolt init) | One command |
| **Performance** | Slow (DB overhead) | Fast (file I/O) |
| **Debugging** | opaque | Plain text JSONL |
| **Git-friendly** | Requires sync | Just a file |
| **Parallel Safety** | Atomic | File-based (good enough) |

---

## File Format

### .migration-tasks.jsonl
Each line is a JSON task object:

```json
{
  "id": "task-001",
  "python_file": "src/utils/math.py",
  "typescript_file": "omrchecker-js/packages/core/src/utils/math.ts",
  "phase": 1,
  "status": "completed",
  "dependencies": [],
  "agent": "agent-1",
  "score": 85,
  "created_at": "2026-02-28T10:00:00Z",
  "updated_at": "2026-02-28T10:15:00Z",
  "completed_at": "2026-02-28T10:15:00Z"
}
```

**Status values**:
- `available` - Ready to be claimed
- `in_progress` - Claimed by an agent
- `completed` - Finished
- `blocked` - Waiting on dependencies

---

## Parallel Agent Coordination

### Scenario: 4 Parallel Agents

**Terminal 1** (Agent 1):
```bash
TASK=$(uv run scripts/migration_tasks.py next | jq -r '.id')
uv run scripts/migration_tasks.py claim $TASK --agent agent-1
# ... do work ...
uv run scripts/migration_tasks.py complete $TASK --score 82
```

**Terminal 2** (Agent 2):
```bash
TASK=$(uv run scripts/migration_tasks.py next | jq -r '.id')  # Gets different task!
uv run scripts/migration_tasks.py claim $TASK --agent agent-2
# ... do work ...
```

**Terminal 3** (Agent 3):
```bash
# Same pattern, gets yet another task
```

**Terminal 4** (Monitor):
```bash
watch -n 5 'uv run scripts/migration_tasks.py progress'
```

**Safety**: File-based coordination prevents duplicate claims (same as git conflict resolution)

---

## Migration from Beads (Optional)

If you want to preserve beads for other workflows:

1. **Keep both systems**: Use `migration_tasks.py` for migration, beads for issues
2. **Export from beads**: 
   ```bash
   bd export --json > migration-tasks-from-beads.json
   # Convert to migration_tasks.py format
   ```
3. **No migration needed**: Both are independent, no conflicts

---

## Troubleshooting

### "No available tasks"
```bash
# Check if tasks were initialized
uv run scripts/migration_tasks.py list

# If empty, initialize
uv run scripts/migration_tasks.py init --phase 1
```

### "Task not found"
```bash
# List all tasks to see correct IDs
uv run scripts/migration_tasks.py list
```

### "Task is not available"
```bash
# Check task status
uv run scripts/migration_tasks.py list | grep task-001

# If in_progress but abandoned, manually edit .migration-tasks.jsonl
# or reset: rm .migration-tasks.jsonl && reinitialize
```

---

## Future Enhancements

Potential improvements (not needed now):

1. **Dependency tracking**: Add `--depends-on` to create blocked tasks
2. **Agent heartbeat**: Auto-release stale in_progress tasks
3. **Web dashboard**: Simple Flask/FastAPI UI for progress
4. **Slack/Discord notifications**: Post progress updates
5. **Time tracking**: Record time spent per task

---

**Status**: ✅ Production Ready  
**Tested**: Phase 1 initialization (12 tasks)  
**Dependencies**: None (Python stdlib + jq for parsing)
