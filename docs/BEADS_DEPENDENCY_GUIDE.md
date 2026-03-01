# Beads Dependency & Field Reference

**Date**: 2026-03-01  
**Purpose**: Clarify beads dependency semantics, owner vs assignee, and common patterns

---

## Dependency Semantics

### Core Concept

Beads uses **directional dependencies** where the relationship is always from the perspective of the **first** issue.

### The Two Ways to Create Dependencies

#### Method 1: Using `bd create` with `--deps`

```bash
bd create "Task B" --deps <type>:<task-a-id>
```

The `--deps` flag describes **Task B's relationship TO Task A**:
- `--deps blocked-by:task-a` → Task B is blocked by Task A (B depends on A)
- `--deps blocks:task-a` → Task B blocks Task A (A depends on B)

#### Method 2: Using `bd dep add`

```bash
bd dep add <task-b-id> <task-a-id>
# OR
bd dep add <task-b-id> --blocked-by <task-a-id>
# OR
bd dep add <task-b-id> --depends-on <task-a-id>
```

**Default behavior**: Task B depends on (is blocked by) Task A.

**Shorthand**:
```bash
bd dep <task-a-id> --blocks <task-b-id>
# Equivalent to: bd dep add <task-b-id> <task-a-id>
```

---

## Dependency Types

### Primary Types

**1. `blocks` (default)**
- **Meaning**: Creates a blocking dependency
- **`bd ready` behavior**: Blocked issues do NOT appear in ready list
- **Example**: "Task A must complete before Task B can start"
  ```bash
  # Task B is blocked by Task A
  bd create "Task B" --deps blocked-by:task-a
  # OR
  bd dep add task-b task-a
  ```

**2. `blocked-by`** (alias for `blocks` from the other direction)
- Same as `blocks`, just expressed differently
- Use when creating the blocked task
  ```bash
  bd create "Task B" --deps blocked-by:task-a
  ```

**3. `parent-child`**
- Hierarchical relationship (epics and subtasks)
- Child issues are part of parent epic
  ```bash
  bd create "Subtask" --deps parent-child:epic-id
  ```

**4. `discovered-from`**
- Links new work discovered during another task
- Useful for audit trail and context
  ```bash
  bd create "Bug found during Task A" --deps discovered-from:task-a
  ```

### Other Types

- `tracks` - Tracking relationship
- `related` - General relatedness
- `until` - Temporal dependency
- `caused-by` - Causation link
- `validates` - Validation relationship
- `relates-to` - Bidirectional association
- `supersedes` - Replacement relationship

---

## Common Patterns

### Pattern 1: Sequential Tasks (A → B → C)

**Goal**: Task A must complete, then B, then C.

```bash
# Create Task A (no dependencies)
bd create "Task A: Foundation" -t task -p 1

# Create Task B (depends on A)
bd create "Task B: Build on A" --deps blocked-by:task-a -t task -p 1

# Create Task C (depends on B)
bd create "Task C: Build on B" --deps blocked-by:task-b -t task -p 1
```

**Result**:
- `bd ready` shows only Task A
- After closing Task A, `bd ready` shows Task B
- After closing Task B, `bd ready` shows Task C

### Pattern 2: Parallel Tasks with Convergence (A + B → C)

**Goal**: Tasks A and B can run in parallel, but C needs both to complete.

```bash
# Create Task A and B (no dependencies)
bd create "Task A: Parallel work 1" -t task -p 1
bd create "Task B: Parallel work 2" -t task -p 1

# Create Task C (depends on both)
bd create "Task C: Needs A and B" -t task -p 1
bd dep add task-c task-a
bd dep add task-c task-b
```

**Result**:
- `bd ready` shows A and B
- Task C appears in `bd ready` only after BOTH A and B are closed

### Pattern 3: Epic with Subtasks

**Goal**: Track multiple subtasks under an epic.

```bash
# Create epic
bd create "Epic: User Authentication" -t epic -p 1

# Create subtasks
bd create "Subtask: Login UI" --deps parent-child:epic-id -t task -p 1
bd create "Subtask: OAuth backend" --deps parent-child:epic-id -t task -p 1
bd create "Subtask: Session management" --deps parent-child:epic-id -t task -p 1
```

**Result**:
- All subtasks linked to epic
- Epic cannot be closed until all children are closed

### Pattern 4: Discovered Work

**Goal**: Track work discovered during another task.

```bash
# Working on task-abc, discover a bug
bd create "Bug: Null pointer in validator" \
  --description="Found during implementation of task-abc" \
  -t bug -p 0 \
  --deps discovered-from:task-abc
```

**Result**:
- New bug is linked to original task
- Provides audit trail for how work was discovered

### Pattern 5: Sub-Agent Task Assignment

**Goal**: Lead agent creates tasks for sub-agents with dependencies.

```bash
# Create foundation tasks (can run in parallel)
bd create "Browser-Alpha: Setup OpenCV.js" \
  --description="Agent: Browser-Alpha\n..." \
  -t task -p 0

bd create "Memory-Beta: Create memory utils" \
  --description="Agent: Memory-Beta\n..." \
  -t task -p 0

# Create dependent task (needs both foundations)
bd create "Test-Gamma: Write smoke tests" \
  --description="Agent: Test-Gamma\n..." \
  -t task -p 0
bd dep add test-gamma-id browser-alpha-id
bd dep add test-gamma-id memory-beta-id
```

---

## Understanding `bd ready`

The `bd ready` command shows issues that are **unblocked** and can be worked on immediately.

### Criteria for "Ready"

An issue appears in `bd ready` if:
1. ✅ Status is `open` (not closed, not deferred)
2. ✅ All `blocks` dependencies are closed
3. ✅ Has no open blockers

### Example

```
Task A (open, no dependencies) → Shows in bd ready ✓
Task B (open, blocked-by: Task A which is open) → NOT in bd ready ✗
Task C (open, blocked-by: Task A which is closed) → Shows in bd ready ✓
```

---

## Owner vs Assignee Fields

### Owner

- **Definition**: The user who **created** the issue
- **Set**: Automatically when creating issue (from `git user.email`)
- **Immutable**: Cannot be changed after creation
- **Purpose**: Accountability and audit trail
- **Field**: `owner` (email address)

**Example**:
```json
{
  "id": "omr-abc",
  "title": "Task",
  "owner": "rhuge123@gmail.com",  // Creator
  "created_by": "udayraj123"       // Creator username
}
```

### Assignee

- **Definition**: The user **currently responsible** for the issue
- **Set**: Manually with `bd update --assignee`
- **Mutable**: Can be changed anytime
- **Purpose**: Work assignment and tracking
- **Field**: `assignee` (username or identifier)
- **Optional**: Can be null/empty

**Example**:
```json
{
  "id": "omr-abc",
  "title": "Task",
  "owner": "rhuge123@gmail.com",  // Original creator
  "assignee": "alice",             // Currently assigned to
  "created_by": "udayraj123"
}
```

### Usage Patterns

#### Pattern 1: Self-Assignment
```bash
# Create issue (you are the owner)
bd create "Fix bug" -t bug -p 1

# Claim it for yourself
bd update omr-abc --assignee "$(git config user.name)"
```

#### Pattern 2: Team Assignment
```bash
# Lead creates issue
bd create "Implement feature X" -t task -p 1
# Owner: lead@example.com

# Assign to team member
bd update omr-abc --assignee "bob"
# Now: owner=lead, assignee=bob
```

#### Pattern 3: Sub-Agent Assignment
```bash
# Create issue with agent role in title
bd create "Browser-Alpha: Setup OpenCV.js" \
  --description="Agent: Browser-Alpha\n..." \
  -t task -p 0

# Optionally set assignee
bd update omr-abc --assignee "Browser-Alpha"
```

#### Pattern 4: Filter by Assignee
```bash
# List Alice's assigned tasks
bd list --assignee alice --status open --json

# List unassigned tasks
bd list --json | jq '.[] | select(.assignee == null)'
```

### Owner vs Assignee: Key Differences

| Aspect | Owner | Assignee |
|--------|-------|----------|
| **Set By** | Automatic (creator) | Manual (`--assignee` flag) |
| **Mutable** | No | Yes |
| **Purpose** | Accountability, audit | Work assignment |
| **Required** | Always present | Optional |
| **Type** | Email address | Username/identifier |
| **Query** | Filter by owner | Filter by assignee |

---

## Troubleshooting Dependencies

### Issue: Task not showing in `bd ready` when expected

**Check**:
1. Is the task's status `open`?
2. Does it have any `blocks` dependencies that are still open?
3. Use `bd show <id> --json` to see dependencies

**Example**:
```bash
$ bd show omr-abc --json | jq '.[0].dependencies'
[
  {
    "id": "omr-xyz",
    "status": "open",    # ← Blocker is still open!
    "dependency_type": "blocks"
  }
]
```

### Issue: Accidentally created dependency in wrong direction

**Fix**: Remove and re-add
```bash
# Remove wrong dependency
bd dep remove task-b task-a

# Add correct dependency
bd dep add task-b task-a  # B depends on A
```

### Issue: Cannot close task due to "blocked by open issues"

**Cause**: The task has open dependents (tasks that depend on this one).

**Check**:
```bash
bd show <id> --json | jq '.[0].dependents'
```

**Fix**: Close or remove the blocking dependents first, or use `--force`:
```bash
bd close <id> --force --reason "Closing despite open dependents"
```

---

## Quick Reference Card

### Creating Dependencies

```bash
# Task B depends on Task A
bd create "Task B" --deps blocked-by:task-a
bd dep add task-b task-a
bd dep add task-b --blocked-by task-a

# Task A blocks Task B (same as above, different perspective)
bd dep task-a --blocks task-b

# Multiple dependencies
bd create "Task C" -t task
bd dep add task-c task-a
bd dep add task-c task-b  # C needs both A and B

# Parent-child
bd create "Subtask" --deps parent-child:epic-id

# Discovered work
bd create "Bug" --deps discovered-from:task-id
```

### Checking Dependencies

```bash
# Show dependencies (what this task depends on)
bd show <id> --json | jq '.[0].dependencies'

# Show dependents (what depends on this task)
bd show <id> --json | jq '.[0].dependents'

# Show full dependency tree
bd dep tree <id>

# List blocked tasks
bd blocked --json

# List ready tasks
bd ready --json
```

### Managing Assignees

```bash
# Assign task
bd update <id> --assignee "username"

# Unassign task
bd update <id> --assignee ""

# List by assignee
bd list --assignee "alice" --json

# Check who is assigned
bd show <id> --json | jq '.[0].assignee'
```

---

## Best Practices

### ✅ DO

1. **Use descriptive task titles** that indicate blocking relationships
   ```bash
   bd create "Phase 1: Foundation" -t task
   bd create "Phase 2: Build on Phase 1" --deps blocked-by:phase1-id
   ```

2. **Link discovered work** for audit trail
   ```bash
   bd create "Refactor needed" --deps discovered-from:original-task
   ```

3. **Use parent-child** for epics
   ```bash
   bd create "Epic: Feature" -t epic
   bd create "Subtask 1" --deps parent-child:epic-id
   ```

4. **Assign tasks** when delegating
   ```bash
   bd update <id> --assignee "team-member"
   ```

5. **Check `bd ready`** to find available work
   ```bash
   bd ready --json | jq '.[] | {id, title, priority}'
   ```

### ❌ DON'T

1. **Don't confuse dependency direction**
   - ❌ `bd create "Task B" --deps blocks:task-a` (Task B blocks A - wrong!)
   - ✅ `bd create "Task B" --deps blocked-by:task-a` (Task B blocked by A - correct!)

2. **Don't create circular dependencies**
   ```bash
   bd dep add task-a task-b
   bd dep add task-b task-a  # ← Creates cycle!
   ```
   Use `bd dep cycles` to detect.

3. **Don't rely on owner for assignment**
   - Owner is the creator, not necessarily the assignee
   - Use `assignee` field for work assignment

4. **Don't forget to close blockers**
   - Tasks remain blocked until their dependencies are closed
   - Check `bd blocked` to see what's stuck

---

## Summary

### Key Takeaways

1. **Dependencies are directional**: `bd dep add B A` means "B depends on A"
2. **`blocked-by` vs `blocks`**: Same relationship, opposite perspectives
3. **`bd ready`**: Shows only unblocked, open tasks
4. **Owner**: Immutable creator (email)
5. **Assignee**: Mutable current responsible party (username)
6. **Sub-agent pattern**: Use `[Agent-Role]: Task` + assignee field

### Mental Model

```
Owner = Who created it (immutable)
Assignee = Who's working on it (mutable)

blocked-by = "I need X to finish first"
blocks = "X needs me to finish first"

bd ready = "What can I work on right now?"
bd blocked = "What's stuck waiting on something?"
```

---

**Status**: Production-ready reference  
**Validated**: 2026-03-01 with test tasks
