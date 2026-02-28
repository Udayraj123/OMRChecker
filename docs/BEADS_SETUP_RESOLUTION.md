# Beads No-DB Mode Setup Resolution

**Date**: 2026-02-28  
**Status**: ✅ RESOLVED  
**Beads Version**: 0.56.1

---

## Problem

Beads was configured with `no-db: true` in `.beads/config.yaml` but all commands failed with:
```
Error: Dolt server unreachable at 127.0.0.1:3307: dial tcp 127.0.0.1:3307: connect: connection refused
```

---

## Root Cause

**The `no-db: true` setting is MISNAMED**. It doesn't mean "don't use a database" - it means:
- ✅ Use JSONL as source of truth (`.beads/issues.jsonl`)
- ✅ Auto-sync to/from JSONL after each command
- ❌ **Still requires a running Dolt SQL server**

The "no-db" refers to not needing a remote database or persistent connection, but a local dolt server is still required for in-memory query operations.

---

## Solution Steps

### 1. Install & Configure Dolt

```bash
# Dolt was already installed via homebrew
which dolt
# /opt/homebrew/bin/dolt

# Configure dolt user (global)
dolt config --global --add user.email "beads@local"
dolt config --global --add user.name "Beads User"
```

### 2. Initialize Dolt Database

```bash
cd .beads/dolt
dolt init
# Successfully initialized dolt data repository.
```

### 3. Start Dolt SQL Server

```bash
# Start in background with nohup
nohup dolt sql-server --port 3307 --host 127.0.0.1 > /tmp/dolt.log 2>&1 &

# Verify it's running
lsof -i :3307
# COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# dolt    58156 user   16u  IPv4  ...      0t0  TCP localhost:opsession-prxy (LISTEN)
```

### 4. Initialize Beads

```bash
cd /Users/udayraj.deshmukh/Personals/OMRChecker
bd init --prefix omr --quiet
```

### 5. Verify Working

```bash
# Create test issue
bd create "Migrate math.py to TypeScript" -t task -p 1
# ✓ Created issue: omr-yzj

# Check status
bd status
# 📊 Issue Database Status
# Total Issues: 1
# Open: 1
```

---

## Configuration Files

### `.beads/config.yaml` (Correct Understanding)
```yaml
# This means "sync to JSONL" NOT "don't use database"
no-db: true

# Beads still requires dolt sql-server running on 3307
# The JSONL serves as source of truth
# Dolt provides query engine for filtering, sorting, etc.
```

### Dolt Server Process
```bash
# Check if running
ps aux | grep "dolt sql-server"

# Start if not running
cd .beads/dolt && nohup dolt sql-server --port 3307 --host 127.0.0.1 > /tmp/dolt.log 2>&1 &

# Stop
pkill -f "dolt sql-server"
```

---

## Key Learnings

### What "no-db" Actually Means

| Mode | Dolt Server | JSONL File | Source of Truth | Use Case |
|------|-------------|------------|-----------------|----------|
| `no-db: false` | Required | Auto-generated | Dolt database | Team collaboration |
| `no-db: true` | **Still required** | Auto-synced | JSONL file | Git-friendly, single user |

### Why Dolt is Still Needed

Even in "no-db" mode, beads uses dolt for:
1. **SQL queries** - Filtering, sorting, complex joins
2. **Schema validation** - Type checking, constraints
3. **Transactions** - ACID guarantees for writes
4. **Performance** - Indexed lookups

The JSONL is the persistent storage, but dolt is the query engine.

---

## Future Automation

### Add to Development Setup

Create `.beads/start-dolt.sh`:
```bash
#!/bin/bash
# Start dolt server if not already running

if lsof -i :3307 > /dev/null 2>&1; then
  echo "✓ Dolt server already running on port 3307"
  exit 0
fi

echo "Starting dolt server..."
cd "$(dirname "$0")/dolt"

if [ ! -d ".dolt" ]; then
  echo "Initializing dolt database..."
  dolt init
fi

nohup dolt sql-server --port 3307 --host 127.0.0.1 > /tmp/dolt.log 2>&1 &
sleep 2

if lsof -i :3307 > /dev/null 2>&1; then
  echo "✓ Dolt server started successfully"
else
  echo "✗ Failed to start dolt server"
  cat /tmp/dolt.log
  exit 1
fi
```

### Add to AGENTS.md

```markdown
## Prerequisites

Before using bd commands, ensure dolt server is running:

\`\`\`bash
# Check if running
lsof -i :3307

# If not running
./.beads/start-dolt.sh

# Or manually
cd .beads/dolt && nohup dolt sql-server --port 3307 > /tmp/dolt.log 2>&1 &
\`\`\`
```

---

## Testing Checklist

- [x] Dolt installed and configured
- [x] Dolt database initialized in `.beads/dolt/`
- [x] Dolt sql-server running on port 3307
- [x] Beads initialized with prefix `omr`
- [x] Can create issues (`bd create`)
- [x] Can list issues (`bd list`)
- [x] Can check ready work (`bd ready`)
- [x] Can update issues (`bd update`)
- [x] Can close issues (`bd close`)
- [ ] JSONL auto-sync verified (check `.beads/issues.jsonl` after operations)

---

## Conclusion

**Beads is now fully operational** and ready for Phase 1 re-do exercise.

The key insight: `no-db: true` ≠ "no database needed". It means "JSONL-first with dolt as query engine".

For single-user, git-friendly workflows, this is the correct setup. The JSONL file can be version controlled while the dolt server provides SQL capabilities.

---

## Next Steps

1. ✅ Beads working
2. ⏳ Create startup script (`.beads/start-dolt.sh`)
3. ⏳ Update AGENTS.md with prerequisites
4. ⏳ Proceed with Phase 1 re-do using proper tooling
5. ⏳ Document beads workflow in migration skill

Time spent debugging: ~30 minutes  
Outcome: Successful resolution + documentation for future
