---
name: init-python
description: Orchestrates the full agent-readiness pipeline for Python repositories. Installs agentfill, runs discovery interview, generates BUILD_CHECKLIST.md, and drives checklist-driven extraction using the extractor sub-agent.
user-invocable: false
---

# Init Python - Full Agent Readiness Pipeline

Orchestrate the complete process of making a Python repository agentic-ready.

## Prerequisites

- Target directory must contain Python project markers (`setup.py`, `pyproject.toml`, or `requirements.txt`)
- No existing swe-agent output in `.claude/skills/` (use `/agent-ready:migrate` for that)

## Phase 0: Detection & Setup

<step number="1" required="true">

### 0.1 Detect Existing State

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/detect-existing.sh .
```

Parse the JSON output:
- If `swe_agent_dir` is not null → tell user to run `/agent-ready:migrate` and stop
- If `checklist_exists` is true → tell user to run `/agent-ready:resume` and stop
- If `is_python_project` is false → tell user this requires a Python repository and stop

<validation_gate name="python-project-check" blocking="true">
BLOCK if `is_python_project` is false.
Required state: At least one of `setup.py`, `pyproject.toml`, or `requirements.txt` exists in the target directory.
Action on failure: Inform the user this plugin requires a Python repository and halt.
</validation_gate>

</step>

<step number="2" required="true" depends_on="1">

### 0.2 Install Agentfill

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/install-agentfill.sh .
```

If agentfill is not already installed, the installer creates:
- `.agents/` directory structure
- `.agents/polyfills/agentsmd/` hook scripts
- `.claude/skills -> ../.agents/skills/` symlink
- `.cursor/skills -> ../.agents/skills/` symlink (if Cursor installed)
- `.gemini/skills -> ../.agents/skills/` symlink (if Gemini installed)

<validation_gate name="agentfill-verify" blocking="true">
BLOCK if `.agents/` directory does not exist or symlinks are broken.
Required state: `.agents/` directory exists AND `.claude/skills` is a symlink pointing to `../.agents/skills/`.
Verify with:
```bash
test -d .agents && test -L .claude/skills && readlink .claude/skills | grep -q '.agents/skills'
```
Action on failure: Re-run the install script. If it fails again, report the error and halt.

This symlink is CRITICAL. It means skills installed to `.claude/skills/` (via npx skills -a claude-code)
actually land in `.agents/skills/`. Cursor and Gemini see the same skills through their own symlinks
to `.agents/skills/`. This is why we ONLY install to claude-code — the symlinks share everything.
</validation_gate>

</step>

<step number="3" required="true" depends_on="2">

### 0.3 Initialize Directory Structure

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/init-directories.sh .
```

This script creates:
- `.agents/skills/repo-skill/` skeleton (core/, modules/domain/, modules/technical/, modules/integration/)
- `.claude/` directories (skills/, agents/, rules/, hooks/, commands/, output-styles/)
- `.claude/settings.json` (empty `{}` if not created by agentfill)
- `.claude/.gitignore` (ignores settings.local.json, agent-memory-local/)
- `.claude/rules/README.md` (explains rules directory)

Never overwrites existing files. Verifies `.claude/skills` symlink is intact.

<validation_gate name="dirs-initialized" blocking="true">
BLOCK if `.agents/skills/repo-skill/` or `.claude/` directories do not exist.
Required state: Both directory trees exist, `.claude/skills` symlink is intact.
Action on failure: Re-run the script. If it fails, report the error and halt.
</validation_gate>

</step>

<step number="4" required="true" depends_on="3">

### 0.4 Install Upstream Skills

Install developer skills from `razorpay/agent-skills` into `.agents/skills/`.
Skills are fetched via GitHub API and copied directly — no third-party CLI tools.

**Step 1: Fetch available skills from upstream**

Run the fetch script to discover all available skills with their descriptions and categories:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/fetch-skill-list.sh
```

This uses the GitHub API (`gh api`) to scan the `razorpay/agent-skills` repo and returns
a JSON array of available skills:

```json
[
  {"name": "code-security", "description": "...", "category": "Security", "path": "security/skills/code-security"},
  {"name": "devstack", "description": "...", "category": "Infrastructure", "path": "infrastructure/skills/devstack"},
  ...
]
```

Parse this JSON output. Each skill has: `name`, `description`, `category`, and `path` (repo path).

**Step 2: Install mandatory skills**

Install the 4 mandatory skills by passing their repo paths to the install script:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/install-skills.sh . \
  security/skills/code-security \
  infrastructure/skills/devstack \
  observability/skills/log-volume-optimizer \
  development/skills/tech-spec-reviewer
```

This sparse-clones the repo, copies ONLY these 4 skill directories into `.agents/skills/`,
and cleans up. Skills are installed directly to `.agents/skills/` — no symlink issues,
no third-party tools, no risk of installing to other agent directories.

**Step 3: Present ALL available optional skills to user, category by category**

From the JSON output in Step 1, build the complete list of optional skills:
1. Remove the 4 mandatory skills already installed (code-security, devstack, log-volume-optimizer, tech-spec-reviewer)
2. Remove Razorpay-internal skills not useful for general development: merchant-pricing-plan, merchant-alerts, payment-monitor, terminal-rejection-analyzer, data-point-discovery, canary-config-analyzer, api-council-doc-review, website-verification-pdf-generator, website-verification-webscraper, derived-api-evals, private-note-evals, cell-readiness, rca-validator, rca-reviewer
3. Group remaining skills by their `category` field

For EACH category, present a formatted table of ALL skills in that category and ask
the user to select. Since AskUserQuestion supports max 4 options, use this approach:

**If a category has 4 or fewer skills**, use AskUserQuestion directly:
```
AskUserQuestion:
  question: "[Category]: Which skills to install? (multiSelect)"
  header: "[Category]"
  multiSelect: true
  options: [one option per skill with name as label, description as description]
```

**If a category has more than 4 skills**, first display the full list as a formatted table:

```
Available [Category] skills:

| # | Skill | Description |
|---|-------|-------------|
| 1 | skill-name-1 | First 100 chars of description |
| 2 | skill-name-2 | First 100 chars of description |
| 3 | skill-name-3 | First 100 chars of description |
| ... | ... | ... |
| N | skill-name-N | First 100 chars of description |
```

Then ask the user to type which ones they want (by number or name):

```
AskUserQuestion:
  question: "[Category]: Enter the numbers or names of skills to install (or 'none' to skip)"
  header: "[Category]"
  multiSelect: false
  options:
    - label: "All"
      description: "Install all [N] skills in this category"
    - label: "None"
      description: "Skip this category"
    - label: "Select specific"
      description: "I'll type the numbers or names"
```

If user selects "Select specific", ask them to type the skill numbers or names
as a follow-up question. Parse their response to identify selected skills.

**Process EVERY category in order.** Do not skip categories. Do not show only a subset.
The categories from the upstream repo are: Development, Security, Observability,
Testing, Infrastructure, Data, Finance, Helpers, Integrations.

Use the ACTUAL names and descriptions from the JSON — never use hardcoded or example text.

**Step 4: Install user-selected optional skills**

Collect ALL selected skill names across all categories. Look up their `path` values
from the Step 1 JSON. Install them in a single command:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/install-skills.sh . \
  [path-for-selected-1] \
  [path-for-selected-2] \
  [path-for-selected-3]
```

If the user selected nothing, skip this step.

**Step 5: Verify installation**

```bash
ls .agents/skills/ | grep -v repo-skill
```

Verify ONLY the expected skills were installed (4 mandatory + user-selected optional).
List them to the user for confirmation.

<validation_gate name="skills-installed-check" blocking="true">
BLOCK if `.agents/skills/` contains only `repo-skill` (no upstream skills installed).
Required state: At least the 4 mandatory skills are present in `.agents/skills/`.
Action on failure: Re-run the mandatory install script. If it fails, check `gh` CLI auth.
</validation_gate>

</step>

## Phase 1: Discovery Interview

<step number="5" required="true" depends_on="4">

### 1.1 Ask Discovery Questions

Ask questions one at a time using AskUserQuestion to understand the Python codebase.

Collect answers about:
- **Service purpose and architecture**
  - What is the main purpose of this service/application?
  - Is this a web service (API), CLI tool, library, data processing pipeline, or other?
  - What framework are you using? (FastAPI, Django, Flask, Click, Typer, none)

- **Domain entities**
  - What are the core business entities? (e.g., User, Order, Payment, Document)
  - Which Python modules contain the domain models?
  - Are you using Pydantic, dataclasses, attrs, or plain classes for models?

- **Integration points**
  - What external services does this integrate with?
  - What databases are used? (PostgreSQL, MySQL, MongoDB, Redis)
  - What message queues or event systems? (Kafka, RabbitMQ, Celery, SQS)
  - What APIs does this expose? (REST, GraphQL, gRPC)
  - What APIs does this consume?

- **Critical flows**
  - What are the main workflows or pipelines?
  - What are the entry points? (CLI commands, API endpoints, scheduled tasks)
  - What are the most complex or critical operations?

- **Technical patterns**
  - What ORM are you using? (SQLAlchemy, Django ORM, Peewee, none)
  - What's your testing approach? (pytest, unittest, integration tests)
  - Are you using dependency injection? (dependency-injector, FastAPI Depends)
  - What's your configuration approach? (env vars, YAML, TOML, dynaconf)
  - Are you using type hints extensively?
  - Are you using async/await patterns?

</step>

<step number="6" required="true" depends_on="4">

### 1.2 Explore Codebase via Agent Team

Create an exploration team to analyze the codebase in parallel. Spawn specialized
explorer agents that each focus on a different aspect of the codebase:

```
TeamCreate:
  team_name: "explore-{service-name}"
  description: "Explore {service-name} Python codebase for agent-readiness discovery"
```

Create tasks for each exploration dimension:

```
TaskCreate: "Discover domain entities — Find all Python packages, identify business entities from module structure, model classes, and type definitions"
TaskCreate: "Discover API surface — Find all HTTP routes, CLI commands, entry points, decorators, and handlers"
TaskCreate: "Discover integrations — Find all external service calls, database usage, message queue patterns, and third-party API clients"
TaskCreate: "Discover technical patterns — Find ORM usage, caching patterns, async patterns, testing structure, and configuration management"
```

Spawn 4 explorer agents in a single message. Each agent reports its findings:

```
Task:
  subagent_type: "Explore"
  name: "entity-explorer"
  team_name: "explore-{service-name}"
  prompt: |
    Explore this Python codebase to discover domain entities.

    1. Find all Python packages: find . -name "__init__.py" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./env/*" -not -path "./.git/*" | sed 's|/__init__.py||' | sort -u
    2. Look at package directory structure for entity boundaries (models/, entities/, domain/)
    3. Search for class definitions (dataclasses, Pydantic models, attrs classes)
       - grep -r "@dataclass" --include="*.py"
       - grep -r "class.*BaseModel" --include="*.py"  # Pydantic
       - grep -r "class.*Model" --include="*.py"  # ORM models
    4. Identify the main domain entities (e.g., Template, OMRResponse, Field, BubbleField)
    5. For each entity, note: package path, key files, main class names

    Report findings as a structured list of entities with their package paths and key files.

Task:
  subagent_type: "Explore"
  name: "api-explorer"
  team_name: "explore-{service-name}"
  prompt: |
    Explore this Python codebase to discover the API surface.

    1. Search for HTTP route decorators:
       - grep -r "@app.route\|@router.get\|@router.post\|@api_view" --include="*.py"
       - FastAPI: @app.get, @app.post, @router.get, @router.post
       - Flask: @app.route, @blueprint.route
       - Django: urlpatterns, path(), re_path()
    2. Search for CLI commands:
       - grep -r "@click.command\|@app.command\|if __name__ == .__main__." --include="*.py"
       - Click: @click.command
       - Typer: @app.command
       - argparse: ArgumentParser()
    3. Search for entry points in setup.py or pyproject.toml
    4. Identify handlers, views, and their file locations
    5. Check for middleware, decorators, authentication patterns

    Report: list of endpoints/commands/entry points with handler file locations.

Task:
  subagent_type: "Explore"
  name: "integration-explorer"
  team_name: "explore-{service-name}"
  prompt: |
    Explore this Python codebase to discover service integrations.

    1. Search for HTTP client usage:
       - grep -r "requests\.\|httpx\.\|aiohttp\." --include="*.py"
    2. Search for database connections:
       - grep -r "create_engine\|sessionmaker\|MongoClient\|redis.Redis" --include="*.py"
    3. Search for message queue patterns:
       - grep -r "@celery\|@task\|kafka\|pika\|aiokafka" --include="*.py"
    4. Search for third-party API clients:
       - Look in requirements.txt or pyproject.toml for client libraries
       - grep -r "import.*client\|from.*client import" --include="*.py"
    5. Identify which external services are called and from where

    Report: list of integrations (service name, direction, transport, file locations).

Task:
  subagent_type: "Explore"
  name: "tech-explorer"
  team_name: "explore-{service-name}"
  prompt: |
    Explore this Python codebase to discover technical infrastructure patterns.

    1. Database/ORM:
       - grep -r "from sqlalchemy\|from django.db\|from peewee import" --include="*.py"
       - Look for models/, migrations/, alembic/
    2. Cache:
       - grep -r "redis\|cache\|@cached\|@lru_cache" --include="*.py"
    3. Async patterns:
       - grep -r "async def\|await\|asyncio\|aiohttp" --include="*.py"
    4. Configuration:
       - Look for config.py, settings.py, .env files
       - grep -r "os.getenv\|pydantic.BaseSettings\|dynaconf" --include="*.py"
    5. Framework detection:
       - Check requirements.txt or pyproject.toml for: fastapi, django, flask, starlette
    6. Testing:
       - Look for tests/, test_*.py, conftest.py
       - grep -r "import pytest\|import unittest\|from unittest import" --include="*.py"
    7. Type hints:
       - grep -r "from typing import\|-> " --include="*.py" | wc -l

    Report: infrastructure stack summary with file locations for each component.
```

Wait for all 4 explorers to complete and collect their findings.

Shut down the exploration team:
```
SendMessage: type: "shutdown_request" to each explorer
TeamDelete: (after all confirm shutdown)
```

</step>

<step number="7" required="true" depends_on="5,6" blocking="true">

### 1.3 Generate BUILD_CHECKLIST.md

Using the interview answers and scan results, generate `BUILD_CHECKLIST.md` following the template in `references/checklist-template.md`.

Place it at: `.agents/skills/repo-skill/BUILD_CHECKLIST.md`

For a Python project, the checklist should include:

**Core Documentation:**
- [ ] `core/boundaries.md` - Service boundaries and responsibilities
- [ ] `core/quick-ref.md` - Common operations quick reference

**Domain Entities** (one per entity discovered):
- [ ] `domain/{entity}/concept.md` - Mental model for {entity}
- [ ] `domain/{entity}/flows.md` - All code paths for {entity}
- [ ] `domain/{entity}/decisions.md` - Architectural decisions for {entity}
- [ ] `domain/{entity}/constraints.md` - Business rules and validations for {entity}
- [ ] `domain/{entity}/integration.md` - How {entity} relates to other entities

**Technical Modules:**
- [ ] `technical/database/schema.md` - Database schema, models, queries
- [ ] `technical/caching/strategy.md` - Caching patterns and invalidation
- [ ] `technical/async/patterns.md` - Async/await usage and event loops
- [ ] `technical/configuration/management.md` - Config loading and env vars
- [ ] `technical/testing/approach.md` - Test structure and fixtures

**Integration Documentation:**
- [ ] `integration/apis/{api_name}.md` - REST/GraphQL/gRPC API documentation
- [ ] `integration/cli/commands.md` - CLI commands and entry points
- [ ] `integration/events/published.md` - Events/messages we publish
- [ ] `integration/events/consumed.md` - Events/messages we consume
- [ ] `integration/external/{service}/outbound.md` - How we call external {service}
- [ ] `integration/external/{service}/inbound.md` - How external {service} calls us

Show the checklist to the user and ask for confirmation before proceeding.

<validation_gate name="checklist-created" blocking="true">
BLOCK if `BUILD_CHECKLIST.md` was not created at `.agents/skills/repo-skill/BUILD_CHECKLIST.md`.
Required state: File exists and contains at least one `- [ ]` task entry.
Action on failure: Re-generate the checklist from interview and scan data.
</validation_gate>

</step>

## Phase 2: Checklist-Driven Extraction via Agent Teams

<step number="8" required="true" depends_on="7">

### 2.1 Create Extraction Team

Create an Agent Team to coordinate parallel extraction across domain entities:

```
TeamCreate:
  team_name: "extraction-{service-name}"
  description: "Extract domain knowledge from {service-name} Python codebase"
```

### 2.2 Create Tasks from BUILD_CHECKLIST.md

Read BUILD_CHECKLIST.md and create a task (TaskCreate) for each pending `- [ ]` item. Group tasks into waves with dependencies:

**Wave 1 — Core (no dependencies, run first):**
```
TaskCreate: "Extract core/boundaries.md — Service boundaries and responsibilities"
TaskCreate: "Extract core/quick-ref.md — Common operations quick reference"
```

**Wave 2 — Domain Concepts (depends on Wave 1):**
```
TaskCreate: "Extract domain/{entity}/concept.md — Mental model for {entity}"
  → blockedBy: [core tasks]
```
Create one task per entity. All concept tasks can run in parallel.

**Wave 3 — Domain Flows, Decisions, Constraints (depends on Wave 2 concepts):**
```
TaskCreate: "Extract domain/{entity}/flows.md — All code paths for {entity}"
  → blockedBy: [concept task for same entity]
TaskCreate: "Extract domain/{entity}/decisions.md — Architectural decisions for {entity}"
  → blockedBy: [concept task for same entity]
TaskCreate: "Extract domain/{entity}/constraints.md — Business rules for {entity}"
  → blockedBy: [concept task for same entity]
TaskCreate: "Extract domain/{entity}/integration.md — Entity relationships for {entity}"
  → blockedBy: [concept task for same entity]
```

**Wave 4 — Technical (no dependency on domain, can start with Wave 2):**
```
TaskCreate: "Extract technical/database/schema.md — Database schema and ORM models"
TaskCreate: "Extract technical/caching/strategy.md — Cache strategy and patterns"
TaskCreate: "Extract technical/async/patterns.md — Async/await usage patterns"
TaskCreate: "Extract technical/configuration/management.md — Config and env management"
TaskCreate: "Extract technical/testing/approach.md — Testing structure and fixtures"
```

**Wave 5 — Integration (depends on domain flows):**
```
TaskCreate: "Extract integration/apis/{api}.md — API endpoint documentation"
  → blockedBy: [relevant domain flow tasks]
TaskCreate: "Extract integration/cli/commands.md — CLI command documentation"
TaskCreate: "Extract integration/events/published.md — Events we publish"
TaskCreate: "Extract integration/events/consumed.md — Events we consume"
TaskCreate: "Extract integration/external/{service}/outbound.md — How we call {service}"
TaskCreate: "Extract integration/external/{service}/inbound.md — How {service} calls us"
```

### 2.3 Spawn Extraction Teammates

Spawn 3-5 teammates (depending on codebase size). Each teammate self-claims tasks from the shared task list:

```
Task:
  subagent_type: "general-purpose"
  name: "extractor-1"
  team_name: "extraction-{service-name}"
  mode: "bypassPermissions"
  prompt: |
    You are an extraction teammate for a Python codebase. Your job:

    1. Check TaskList for available (pending, unblocked, unowned) tasks
    2. Claim a task with TaskUpdate (set owner to your name, status to in_progress)
    3. Read the task description to understand what to extract
    4. Read the swe-repo-builder skill for the extraction pattern matching the task type:
       - concept → references/patterns/domain-patterns.md
       - flows → references/patterns/flow-patterns.md
       - decisions → references/patterns/decision-patterns.md
       - technical → use swe-technical-builder skill references/categories.md
       - integration → references/workflow.md (integration section)
    5. Scan the relevant Python source code using Grep, Glob, Read
       - Look for class definitions, decorators, function signatures
       - Trace imports to understand dependencies
       - Read docstrings and type hints for context
       - Follow async/await patterns
    6. Write the output file to .agents/skills/repo-skill/modules/{path}
    7. Mark task completed with TaskUpdate
    8. Check TaskList for next available task
    9. Repeat until no unblocked tasks remain

    Python-specific guidance:
    - Use file:line references, never duplicate code
    - Document ALL code paths including error handlers and edge cases
    - Capture decorators, context managers, and middleware patterns
    - Note async vs sync functions
    - Document Pydantic models, dataclasses, and validators
    - Stay within 20-40 pages of source context per task
    - If stuck, send a message to team-lead describing the issue
    - When no more tasks are available, go idle

Task:
  subagent_type: "general-purpose"
  name: "extractor-2"
  team_name: "extraction-{service-name}"
  mode: "bypassPermissions"
  prompt: [same as above]

Task:
  subagent_type: "general-purpose"
  name: "extractor-3"
  team_name: "extraction-{service-name}"
  mode: "bypassPermissions"
  prompt: [same as above]
```

Spawn all teammates in a single message so they start working in parallel.

### 2.4 Monitor and Coordinate

As team lead, monitor extraction progress:

1. Wait for teammate messages (delivered automatically)
2. When a teammate reports an issue, provide guidance
3. Periodically check TaskList for progress
4. When teammates find cross-entity references, relay information between them via SendMessage

### 2.5 Verify and Finalize Extraction

When all tasks are complete:

1. Check TaskList — all tasks should be `completed`
2. Verify every output file exists and is non-empty:

```bash
find .agents/skills/repo-skill/modules -name "*.md" -type f | sort
```

3. Update BUILD_CHECKLIST.md — mark all extracted items as `[x]`

<validation_gate name="extraction-complete" blocking="true">
BLOCK if any task in the team task list is still pending or in_progress.
Required state: All extraction tasks completed, all output files exist.
Action on failure: Check which tasks failed, reassign or investigate.
</validation_gate>

### 2.6 Shut Down Extraction Team

Send shutdown requests to all teammates, then delete the team:

```
SendMessage: type: "shutdown_request" to each teammate
TeamDelete: (after all teammates confirm shutdown)
```

</step>

## Phase 3: Finalization

<step number="9" required="true" depends_on="8">

### 3.1 Generate Nested AGENTS.md Files

For each domain entity (Python module/package), create `[entity-package-dir]/AGENTS.md` with:
- Entity-specific context
- Key files and entry points (classes, functions)
- Common operations for this entity
- Links to the main class definitions

Use the nested AGENTS.md template from `references/agents-md-template.md`.

Example locations:
- `src/processors/detection/AGENTS.md`
- `src/processors/evaluation/AGENTS.md`
- `src/schemas/AGENTS.md`

</step>

<step number="10" required="true" depends_on="9">

### 3.2 Generate Root AGENTS.md (MUST BE LAST)

Invoke the `agents-md` skill to generate the root AGENTS.md. This skill:
1. Discovers all installed skills in `.agents/skills/`
2. Reads each skill's SKILL.md description
3. Synthesizes project context from extracted domain knowledge
4. Generates the Skills Index with explicit trigger instructions

The Skills Index is the most important section. Based on Vercel's research, agents invoke skills
56% more reliably when AGENTS.md contains explicit trigger instructions vs relying on skill auto-discovery.

This MUST be the final generation step because it reads all extracted content and installed skills
to produce an accurate, complete index.

If an existing AGENTS.md is present, back it up to `AGENTS.md.bak` before overwriting.

</step>

<step number="11" required="true" depends_on="10">

### 3.3 Update SKILL.md

Create `.agents/skills/repo-skill/SKILL.md` with progressive disclosure rules:
- `always-load`: core/boundaries.md, core/quick-ref.md
- `on-mention`: domain entity modules (load when entity name mentioned)
- `on-file-change`: technical modules (load when related files modified)
- `on-import`: Load module docs when Python imports are detected in user messages

Example progressive disclosure rules for Python:
```yaml
progressive_disclosure:
  always_load:
    - core/boundaries.md
    - core/quick-ref.md

  on_mention:
    template: domain/template/
    detection: domain/detection/
    evaluation: domain/evaluation/

  on_file_change:
    "**/*.py": technical/
    "tests/**": technical/testing/approach.md
    "alembic/**": technical/database/schema.md

  on_import:
    "from src.processors.detection": domain/detection/
    "from src.schemas": domain/schemas/
```

</step>

<step number="12" required="true" depends_on="11">

### 3.4 Write Version File

Create `.agents/.agent-ready-version`:
```json
{
  "plugin_version": "1.0.0",
  "swe_agent_template_version": "1.1.0",
  "agentfill_version": "0.4.0",
  "skills_source": "razorpay/agent-skills",
  "skills_managed_by": "install-skills.sh (sparse git clone)",
  "last_extraction": "[today's date]",
  "last_skills_update": "[today's date]",
  "last_update_check": "[today's date]",
  "language": "python",
  "python_version": "[detected from pyproject.toml or runtime]",
  "framework": "[detected framework: fastapi, django, flask, etc.]"
}
```

Note: Installed skills are tracked by the skills CLI in `.skill-lock.json`, not in this file.

</step>

<step number="13" required="true" depends_on="12">

### 3.5 Report Completion

Show summary of what was generated:
- Number of domain entities documented (Python packages/modules)
- Number of technical modules
- Number of integration docs (APIs, CLI commands, events)
- Total files created
- Agents wired (Claude, Cursor, Gemini)
- Python-specific artifacts (models, decorators, async patterns documented)

</step>

## Rules

**MUST:**
- Verify Python project markers (`setup.py`, `pyproject.toml`, or `requirements.txt`) before proceeding past Phase 0.1
- Run agentfill install before creating the skeleton structure
- Always use `install-skills.sh` to install skills — it copies directly to `.agents/skills/`
- Never use `npx skills` — it installs to all detected agent directories uncontrollably
- Ask user explicitly which optional skills they want before installing any
- Ask discovery questions one at a time via AskUserQuestion
- Mark checklist tasks complete after each extraction
- Verify each output file was written before marking its checklist task complete
- Generate root AGENTS.md as the FINAL generation step
- Back up existing AGENTS.md to `AGENTS.md.bak` before overwriting
- Document Python-specific patterns: decorators, context managers, async/await, type hints
- Include Pydantic models, dataclasses, and validators in entity documentation
- Document entry points: CLI commands, API routes, scheduled tasks, event handlers

**MUST NOT:**
- Use `npx skills` for installation (installs to all detected agents uncontrollably)
- Install skills to `.claude/skills/` or `.cursor/skills/` directly (use `.agents/skills/` only)
- Install skills the user did not explicitly select
- Skip validation gates
- Proceed to Phase 2 without user confirming the checklist
- Load more than 20-40 pages of context per extraction task
- Overwrite existing AGENTS.md without backing up first
- Proceed to the next parallel batch if any output file from the current batch is missing
- Spawn extractors for dependent tasks (e.g., flows before concepts) in the same parallel batch
- Skip documentation of async patterns, decorators, or middleware

**Python-Specific Guidelines:**
- Prioritize documenting class hierarchies and inheritance patterns
- Capture all decorators and their effects (@dataclass, @cached, @app.route, etc.)
- Document async/await usage and event loop patterns
- Note type hints and Pydantic validators
- Document middleware, context managers, and magic methods
- Include pytest fixtures and test patterns
- Document ORM models, database sessions, and query patterns
- Capture configuration loading and environment variable usage
