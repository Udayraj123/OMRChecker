# Beads Workflow for Agent Collaboration

**Date**: 2026-03-01  
**Purpose**: Define how agents use beads (bd) for task delegation and sub-agent spawning

---

## Overview

When a lead agent needs to spawn sub-agents to work in parallel or delegate specialized tasks, beads provides a structured way to:
1. Create tasks with clear assignments
2. Track ownership and status
3. Maintain dependencies and blockers
4. Enable parallel work without conflicts

---

## Core Principles

### 1. Task Assignment Pattern
Every beads issue should have a clear **assignee** indicated in the title:
```
[Agent-Role]: Task description
```

**Examples**:
- `Foundation-Alpha: Migrate math.py to math.ts`
- `Schema-Beta: Validate template.json with Zod`
- `Test-Author-Math: Write unit tests for math.ts`

### 2. Issue Description Structure
```markdown
Agent: [Role-Name]
Responsibilities:
- [Specific task 1]
- [Specific task 2]

Files:
- [Input files]
- [Output files]

Success Criteria:
- [Criterion 1]
- [Criterion 2]

Estimated Time: [X hours/minutes]
```

### 3. Dependency Tracking
```bash
# Task A blocks Task B (B depends on A)
bd create "Task B" --deps blocks:task-a-id

# Task discovered during work on parent
bd create "Found Bug" --deps discovered-from:parent-id
```

---

## Workflow: Lead Agent Spawning Sub-Agents

### Step 1: Plan and Break Down Work

Lead agent analyzes work and identifies:
- Independent tasks (can run in parallel)
- Dependent tasks (sequential)
- Specialized roles needed

**Example Breakdown**:
```
Phase 1: Infrastructure Setup
  └─> Phase 2: Parallel Migration
       ├─> Foundation-Alpha: Utils (4 files)
       ├─> Schema-Beta: Schemas (3 files)
       └─> Processor-Delta: Processors (3 files)
```

### Step 2: Create Beads Issues for Each Sub-Agent

**Format**:
```bash
bd --actor "[Agent-Role]" create "[Agent-Role]: [Task Summary]" \
  --description="[Detailed instructions with Agent field]" \
  -t task \
  -p [priority] \
  --deps [dependencies if any] \
  --json
```

**IMPORTANT**: Use `--actor` flag to set the agent role as the creator. This ensures:
- `created_by` field shows agent name (e.g., "Browser-Alpha")
- `owner` field shows agent identifier for accountability
- Clear audit trail of which agent created which task

**Concrete Example**:
```bash
bd --actor "Foundation-Alpha" create "Foundation-Alpha: Migrate math.py to math.ts" \
  --description="Agent: Foundation-Alpha
Responsibilities:
- Migrate src/utils/math.py to TypeScript
- Maintain all 19 methods
- Add type annotations
- Test compilation

Files:
- Input: src/utils/math.py (332 lines)
- Output: omrchecker-js/packages/core/src/utils/math.ts

Success Criteria:
- TypeScript compiles without errors
- All 19 methods migrated
- Type coverage 100%
- No any types

Estimated Time: 30-45 minutes" \
  -t task \
  -p 1 \
  --deps blocked-by:omr-setup-123 \
  --json
```

**Note**: Also fixed dependency from `blocks` to `blocked-by` (see BEADS_DEPENDENCY_GUIDE.md).

### Step 3: Document Sub-Agent Roles

Create a **SUBAGENT_BRIEFING.md** or similar:
```markdown
## Sub-Agent Roles

### Foundation-Alpha
**Specialty**: Utility functions, pure JavaScript/TypeScript
**Tasks**: math.ts, geometry.ts, stats.ts, checksum.ts
**Skills**: Type inference, algorithm translation

### Schema-Beta
**Specialty**: Data modeling, validation
**Tasks**: Pydantic to Zod, schema validation
**Skills**: Type-safe schemas, runtime validation

### Processor-Delta
**Specialty**: Image processing, pipeline orchestration
**Tasks**: Processor classes, pipeline coordination
**Skills**: Async/await, state management
```

### Step 4: Spawn Sub-Agents (Conceptual)

When spawning sub-agents (if supported by your environment):

**Pass to each sub-agent**:
1. Issue ID
2. Role name
3. Briefing document
4. Relevant context files

**Example spawn command** (conceptual):
```bash
spawn-subagent \
  --role "Foundation-Alpha" \
  --issue "omr-123" \
  --context "SUBAGENT_BRIEFING.md,src/utils/math.py" \
  --prompt "You are Foundation-Alpha. Claim your assigned issue (omr-123) and complete the migration task."
```

### Step 5: Sub-Agent Workflow

Each sub-agent follows this workflow:

**Claim Task**:
```bash
bd --actor "[Agent-Role]" update omr-123 --status in_progress --json
```

**IMPORTANT**: Always use `--actor` flag with the agent role name for all bd commands:
- Issue creation: `bd --actor "Browser-Alpha" create ...`
- Updates: `bd --actor "Browser-Alpha" update ...`
- Closing: `bd --actor "Browser-Alpha" close ...`

This ensures proper attribution in the audit trail.

**Work on Task**:
- Follow instructions in issue description
- Complete all success criteria
- Run validation/tests

**Commit Work**:
```bash
git add [files]
git commit -m "[category]: [summary] (omr-123)

[Agent-Role]: [Brief description]
- [Key change 1]
- [Key change 2]

Issue: omr-123"
```

**Close Task**:
```bash
bd --actor "[Agent-Role]" close omr-123 --reason "Migration complete. All success criteria met." --json
```

### Step 6: Lead Agent Monitors Progress

```bash
# Check status of all tasks
bd status --json

# Check ready tasks (unblocked)
bd ready --json

# Check specific task
bd show omr-123 --json

# List tasks by status
bd list --status in_progress --json
bd list --status open --json
```

### Step 7: Handle Discovered Work

If a sub-agent discovers new work:

```bash
# Create linked issue with agent attribution
bd --actor "Foundation-Alpha" create "Found issue in math.py: Division by zero not handled" \
  --description="Discovered during migration of math.py by Foundation-Alpha.
  
Line 45 in distance() function can divide by zero if points are identical.

Suggested fix: Add zero-check before division." \
  -t bug \
  -p 1 \
  --deps discovered-from:omr-123 \
  --json
```

---

## Issue Metadata Fields

### Required Fields
- **title**: Must include agent role `[Agent-Role]: Task`
- **description**: Must include `Agent:` field and success criteria
- **type**: `task`, `bug`, `feature`, `epic`, `chore`
- **priority**: `0` (critical), `1` (high), `2` (medium), `3` (low), `4` (backlog)

### Optional But Recommended
- **deps**: `blocks:issue-id` or `blocked-by:issue-id` or `discovered-from:issue-id`
- **estimate**: Time estimate in description

### Naming Convention for Agent Roles

**Pattern**: `[Domain]-[Greek Letter]`

**Examples**:
- **Foundation-Alpha**: Core utilities, foundational code
- **Schema-Beta**: Data models, validation
- **Image-Gamma**: Image processing, computer vision
- **Processor-Delta**: Business logic, pipelines
- **Test-Epsilon**: Testing, quality assurance
- **Integration-Zeta**: Integration tests, E2E
- **Memory-Eta**: Memory management, leak detection
- **Docs-Theta**: Documentation, guides

---

## Example: Complete Workflow

### Scenario: Migrate 10 Python files to TypeScript

**Lead Agent Creates Plan**:
```
Phase 1: Setup (omr-001) - Lead agent
Phase 2: Migrate utilities (4 files) - Parallel
  ├─> omr-002: Foundation-Alpha: math.py
  ├─> omr-003: Foundation-Alpha: geometry.py
  ├─> omr-004: Foundation-Alpha: stats.py
  └─> omr-005: Foundation-Alpha: checksum.py
Phase 3: Migrate schemas (3 files) - Parallel
  ├─> omr-006: Schema-Beta: template.py
  ├─> omr-007: Schema-Beta: config.py
  └─> omr-008: Schema-Beta: evaluation.py
Phase 4: Migrate processors (3 files) - Parallel
  ├─> omr-009: Processor-Delta: base.py
  ├─> omr-010: Processor-Delta: pipeline.py
  └─> omr-011: Processor-Delta: coordinator.py
```

**Lead Agent Commands**:
```bash
# Create Phase 1 (done by lead agent)
bd --actor "Lead-Agent" create "Lead: Setup TypeScript project structure" -t task -p 0

# Create Phase 2 (parallel work) with agent attribution
bd --actor "Foundation-Alpha" create "Foundation-Alpha: Migrate math.py to math.ts" \
  --description="..." -t task -p 1 --deps blocked-by:omr-001

bd --actor "Foundation-Alpha" create "Foundation-Alpha: Migrate geometry.py to geometry.ts" \
  --description="..." -t task -p 1 --deps blocked-by:omr-001

# ... and so on
```

**Note**: Lead agent creates tasks with agent-specific `--actor` flags, so each issue shows the correct agent as creator.

**Sub-Agent (Foundation-Alpha) Commands**:
```bash
# Claim first task
bd --actor "Foundation-Alpha" update omr-002 --status in_progress

# Do work...
# Commit...

# Close task
bd --actor "Foundation-Alpha" close omr-002 --reason "Migration complete. 19/19 methods migrated, 100% type coverage."

# Claim next task
bd --actor "Foundation-Alpha" update omr-003 --status in_progress

# ... continue
```

**Lead Agent Monitors**:
```bash
# Check progress
bd status --json
# Output: 1 closed, 3 in_progress, 6 open

# See what's ready
bd ready --json
# Output: omr-006, omr-007, omr-008 (Phase 3 tasks now unblocked)
```

---

## Best Practices

### DO

✅ **Use clear agent role prefixes**
```
Foundation-Alpha: Migrate math.py to math.ts
```

✅ **Include detailed descriptions**
```
Agent: Foundation-Alpha
Responsibilities: [clear list]
Success Criteria: [measurable outcomes]
```

✅ **Track dependencies explicitly**
```bash
--deps blocked-by:parent-id  # Current task depends on parent
```

✅ **Use standardized commit messages**
```
feat(ts-migrate): migrate math.py to math.ts (omr-123)

Foundation-Alpha: Core mathematical utilities
- 19 methods migrated
- 100% type coverage

Issue: omr-123
```

✅ **Report status at key milestones**
```bash
bd --actor "[Agent-Role]" update omr-123 --status in_progress  # When starting
bd --actor "[Agent-Role]" close omr-123 --reason "..."         # When done
```

✅ **Always use --actor flag for attribution**
```bash
bd --actor "Browser-Alpha" create "Task" ...  # Sets created_by field
bd --actor "Browser-Alpha" update ...          # Sets actor in audit trail
```

### DON'T

❌ **Don't create vague tasks**
```
Foundation-Alpha: Do some migration
```

❌ **Don't skip agent assignment**
```
Migrate math.py  # Missing agent role!
```

❌ **Don't forget dependencies**
```bash
# If Task B needs Task A, specify it!
--deps blocks:task-a-id
```

❌ **Don't use generic commit messages**
```
Updated files  # BAD
```

❌ **Don't leave tasks in limbo**
```bash
# Always close or update status
bd close omr-123 --reason "..."
```

---

## Query Patterns

### For Lead Agent

**Find available work**:
```bash
bd ready --json
```

**Check overall status**:
```bash
bd status --json
```

**See what's blocked**:
```bash
bd list --status blocked --json
```

**Find tasks by agent**:
```bash
bd list --json | jq '.[] | select(.title | startswith("Foundation-Alpha"))'
```

### For Sub-Agent

**Find my assigned tasks**:
```bash
# Assuming agent role is in title
bd list --status open --json | grep "Foundation-Alpha"
```

**Claim my next task**:
```bash
bd ready --json | grep "Foundation-Alpha" | head -1
# Get the ID, then:
bd update [id] --status in_progress
```

**Check if I have blockers**:
```bash
bd show [my-task-id] --json | jq '.dependencies'
```

---

## Integration with Warp/Agents

### In AGENTS.md

Add section:
```markdown
## Sub-Agent Task Assignment

When spawning sub-agents, use beads for task tracking:

1. Create issues with agent role prefix: `[Role]: Task`
2. Include detailed description with Agent field
3. Set dependencies with --deps
4. Sub-agents claim with `bd update --status in_progress`
5. Sub-agents close with `bd close --reason`

See docs/BEADS_AGENT_WORKFLOW.md for details.
```

### In Sub-Agent Prompts

```
You are [Agent-Role]. Your assigned task is [task-id].

1. Claim: bd update [task-id] --status in_progress
2. Read: bd show [task-id] --json
3. Work: Follow instructions in description
4. Commit: Include issue ID in commit message
5. Close: bd close [task-id] --reason "..."

Report any discovered issues with:
bd create "Issue" --deps discovered-from:[task-id]
```

---

## Troubleshooting

### Issue: Sub-agent can't find their tasks

**Solution**: Ensure task title includes agent role:
```bash
bd list --json | grep "Foundation-Alpha"
```

### Issue: Dependency confusion

**Solution**: Use `bd show [id] --json` to see full dependency graph:
```json
{
  "dependencies": [
    {"depends_on_id": "omr-001", "type": "blocks"}
  ]
}
```

### Issue: Parallel work conflicts

**Solution**: Ensure tasks are truly independent:
- Different files
- No shared state
- Clear boundaries

### Issue: Lost task ownership

**Solution**: Check status and reclaim if needed:
```bash
bd show omr-123 --json
bd update omr-123 --status in_progress
```

---

## Future Enhancements

### Potential bd Features

1. **Assignee field**: `bd create --assignee "Foundation-Alpha"`
2. **Filter by assignee**: `bd list --assignee "Foundation-Alpha"`
3. **Agent metadata**: Store agent capabilities in issue
4. **Workload balancing**: Auto-assign based on load
5. **Agent completion stats**: Track agent performance

### Integration Ideas

1. **Auto-spawn**: Lead agent spawns sub-agents automatically
2. **Progress dashboard**: Real-time view of all sub-agents
3. **Conflict detection**: Warn if multiple agents touch same file
4. **Auto-merge**: Merge completed work automatically

---

## Commit Message Standard

### Overview

Beads currently does **not** automatically update issue status from commit messages. However, following a consistent commit message standard enables:
- Easy discovery of commits related to specific issues
- Manual verification of work completion
- Git blame/log integration with issue tracking
- Future automation possibilities

### Standard Format

**Structure**:
```
<type>(<scope>): <summary> (<issue-id>)

<body>

<footer>
```

**Required Fields**:
- `type`: Commit type (feat, fix, test, docs, chore, refactor, perf, style, ci)
- `summary`: One-line description (50-72 chars)
- `issue-id`: Beads issue ID in parentheses at end of subject

**Optional Fields**:
- `scope`: Component/module affected
- `body`: Detailed explanation (if needed)
- `footer`: Additional metadata (Issue:, Co-Authored-By:, Breaking-Change:)

### Commit Types

**Feature & Fix**:
- `feat`: New feature or functionality
- `fix`: Bug fix
- `perf`: Performance improvement

**Code Quality**:
- `refactor`: Code restructuring without behavior change
- `style`: Formatting, whitespace, etc. (no logic change)

**Testing & Documentation**:
- `test`: Adding or updating tests
- `docs`: Documentation changes

**Infrastructure**:
- `chore`: Maintenance, dependencies, configuration
- `ci`: CI/CD pipeline changes

### Examples

**Simple commit**:
```bash
git commit -m "feat(auth): add OAuth2 login support (omr-abc)"
```

**Detailed commit**:
```bash
git commit -m "feat(ts-migrate): migrate math.py to math.ts (omr-z8z)

Foundation-Alpha: Core mathematical utilities
- 19 methods migrated with full type coverage
- Distance calculations (haversine, manhattan, euclidean)
- Angle conversions and normalizations
- Point transformations

Success Criteria Met:
- TypeScript compiles without errors
- 100% type coverage, zero 'any' types
- All 19 methods migrated

Issue: omr-z8z
Co-Authored-By: Oz <oz-agent@warp.dev>"
```

**Multiple issues** (if commit addresses multiple):
```bash
git commit -m "test: add browser tests for drawing and checksum (omr-ntx, omr-4yj)

Added 30 browser tests:
- drawing.ts: OpenCV.js cv.Mat operations (20 tests)
- checksum.ts: Web Crypto API operations (10 tests)

Issues: omr-ntx, omr-4yj"
```

**Discovered work** (commit that discovers new issues):
```bash
git commit -m "fix(pipeline): handle empty image arrays (omr-def)

Fixed crash when processor receives empty image array.
Discovered during work on omr-abc.

Created follow-up issue omr-ghi for refactoring error handling.

Issue: omr-def
Discovered-From: omr-abc
Created: omr-ghi"
```

### Issue References

**In Commit Subject** (REQUIRED):
```
feat(auth): add login endpoint (omr-123)
                                 ^^^^^^^^ Issue ID in parentheses
```

**In Commit Footer** (OPTIONAL but recommended for complex commits):
```
Issue: omr-123
```

**Multiple Issues**:
```
Subject: feat(test): add unit tests for math and geometry (omr-z8z, omr-73f)

Footer:
Issues: omr-z8z, omr-73f
```

### Action Keywords (Manual Workflow)

While beads doesn't auto-update from commits, use these keywords for **human clarity**:

**When starting work**:
- `start`, `begin`, `wip` (Work in Progress)
```
wip(auth): begin OAuth implementation (omr-123)
```

**When completing work**:
- `complete`, `finish`, `done`, `implement`
```
feat(auth): complete OAuth implementation (omr-123)
```

**After commit, manually update beads**:
```bash
bd close omr-123 --reason "OAuth implementation complete (commit abc123)"
```

### Agent-Specific Patterns

**Include agent role in body**:
```
feat(schema): migrate template.py to Zod schema (omr-456)

Schema-Beta: Pydantic to Zod migration
- Template.json validation with Zod
- Type-safe schema with runtime validation
- 100% parity with Python schema

Issue: omr-456
Agent: Schema-Beta
Co-Authored-By: Oz <oz-agent@warp.dev>
```

**For discovered work**:
```
fix(processor): add null check in detector (omr-789)

Discovered missing null check during omr-456 implementation.
Created follow-up issue omr-790 for comprehensive null safety audit.

Issue: omr-789
Discovered-From: omr-456
Created: omr-790
Agent: Processor-Delta
```

### Beads Workflow Integration

**1. Claim issue**:
```bash
bd update omr-123 --status in_progress --json
```

**2. Do work + commit with issue ID**:
```bash
git add .
git commit -m "feat(auth): implement OAuth login (omr-123)

Foundation-Alpha: OAuth2 integration
- Added OAuth2 provider configuration
- Implemented token exchange flow
- Added refresh token handling

Issue: omr-123
Agent: Foundation-Alpha
Co-Authored-By: Oz <oz-agent@warp.dev>"
```

**3. Close issue after commit**:
```bash
bd close omr-123 --reason "OAuth implementation complete. See commit $(git rev-parse --short HEAD)"
```

**4. Sync**:
```bash
bd sync
```

### Finding Commits for Issues

**Search git log for issue**:
```bash
git --no-pager log --all --grep="omr-123" --oneline
```

**Show commits for issue**:
```bash
git --no-pager log --all --grep="omr-123" --format="%h %s"
```

**Check if issue was completed**:
```bash
git --no-pager log --all --grep="omr-123" --grep="complete\|finish\|done" --format="%h %s"
```

### Pre-commit Hook Integration

While beads hooks don't parse commit messages, you can add your own validation:

**Check for issue ID** (optional `.git/hooks/commit-msg` addition):
```bash
#!/bin/sh
# Check if commit message contains beads issue ID

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Skip check for merge commits
if echo "$COMMIT_MSG" | grep -q "^Merge branch"; then
    exit 0
fi

# Check for issue ID pattern: omr-xxx or bd-xxx
if ! echo "$COMMIT_MSG" | grep -qE "(omr|bd)-[a-z0-9]+"; then
    echo "Error: Commit message must include a beads issue ID (e.g., omr-abc or bd-123)"
    echo "Format: type(scope): message (issue-id)"
    exit 1
fi

exit 0
```

### Future Automation Potential

If beads adds commit message parsing in the future, this standard is compatible with common patterns:

**GitHub-style keywords** (for future compatibility):
```
Closes: omr-123
Fixes: omr-456
Resolves: omr-789
```

**Current best practice** (manual update):
```bash
# In commit message (for reference)
feat(auth): complete OAuth (omr-123)

# After commit (manual)
bd close omr-123 --reason "Complete (commit abc123)"
```

### Best Practices Summary

✅ **Always include issue ID in subject line**
```
feat(auth): add OAuth (omr-123)
```

✅ **Use consistent format**
```
<type>(<scope>): <summary> (<issue-id>)
```

✅ **Include agent role in body for multi-agent work**
```
Agent: Foundation-Alpha
```

✅ **Add co-author line for agent commits**
```
Co-Authored-By: Oz <oz-agent@warp.dev>
```

✅ **Manually update beads after commit**
```bash
bd close omr-123 --reason "Complete"
```

✅ **Link discovered work**
```
Discovered-From: omr-abc
Created: omr-def
```

❌ **Don't rely on automatic status updates** (not supported yet)
❌ **Don't skip manual bd close** (commit alone won't close issue)
❌ **Don't use vague commit messages** ("updated files" - bad)

---

## Summary

**Key Takeaways**:
1. ✅ Use `[Agent-Role]: Task` naming pattern
2. ✅ Include `Agent:` field in description
3. ✅ Set dependencies with `--deps`
4. ✅ Sub-agents claim with `bd update`
5. ✅ Always close with `bd close --reason`
6. ✅ Link discovered work with `discovered-from`

**Benefits**:
- Clear ownership
- Parallel work without conflicts
- Dependency tracking
- Progress visibility
- Discoverable work trail

---

**Status**: Ready for adoption  
**Applies to**: All multi-agent workflows in OMRChecker project
