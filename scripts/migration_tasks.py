#!/usr/bin/env python3
"""
Lightweight task orchestrator for TypeScript migration.
No beads/dolt dependency - pure JSONL storage.

Usage:
    # Initialize tasks from FILE_MAPPING.json
    uv run scripts/migration_tasks.py init --phase 1

    # Get next available task
    uv run scripts/migration_tasks.py next

    # Claim a task
    uv run scripts/migration_tasks.py claim <task_id> --agent agent-1

    # Complete a task
    uv run scripts/migration_tasks.py complete <task_id> --score 85

    # List tasks
    uv run scripts/migration_tasks.py list --status available

    # Show progress
    uv run scripts/migration_tasks.py progress
"""

import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Literal

TASKS_FILE = Path(".migration-tasks.jsonl")

TaskStatus = Literal["available", "in_progress", "completed", "blocked"]

def load_tasks():
    """Load all tasks from JSONL file"""
    if not TASKS_FILE.exists():
        return []
    tasks = []
    with open(TASKS_FILE) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks

def save_task(task):
    """Append or update a task in JSONL file"""
    tasks = load_tasks()
    # Remove old version if exists
    tasks = [t for t in tasks if t['id'] != task['id']]
    tasks.append(task)
    
    # Rewrite file
    with open(TASKS_FILE, 'w') as f:
        for t in tasks:
            f.write(json.dumps(t) + '\n')

def create_task(task_id, python_file, typescript_file, phase, dependencies=None):
    """Create a new migration task"""
    return {
        'id': task_id,
        'python_file': python_file,
        'typescript_file': typescript_file,
        'phase': phase,
        'status': 'blocked' if dependencies else 'available',
        'dependencies': dependencies or [],
        'agent': None,
        'score': None,
        'created_at': datetime.now(UTC).isoformat(),
        'updated_at': datetime.now(UTC).isoformat(),
        'completed_at': None,
    }

def init_tasks(phase=None):
    """Initialize tasks from FILE_MAPPING.json"""
    with open('FILE_MAPPING.json') as f:
        data = json.load(f)
    
    task_id = 1
    for mapping in data['mappings']:
        # Skip if not in requested phase
        if phase and mapping.get('phase') != phase:
            continue
        
        # Skip if already synced
        if mapping.get('status') == 'synced':
            continue
        
        # Skip N/A entries
        if mapping.get('python', '').startswith('N/A'):
            continue
        
        python_file = mapping.get('python')
        typescript_file = mapping.get('typescript')
        map_phase = mapping.get('phase', 'unknown')
        
        if python_file and typescript_file:
            task = create_task(
                f'task-{task_id:03d}',
                python_file,
                typescript_file,
                map_phase
            )
            save_task(task)
            task_id += 1
    
    print(f"✅ Created {task_id-1} tasks")

def get_next_task():
    """Get next available task (not blocked, not claimed)"""
    tasks = load_tasks()
    available = [t for t in tasks if t['status'] == 'available']
    
    if not available:
        print("No available tasks")
        return None
    
    # Sort by phase, then by ID
    available.sort(key=lambda t: (t.get('phase', 999), t['id']))
    task = available[0]
    
    # Output JSON for easy parsing
    print(json.dumps(task, indent=2))
    return task

def claim_task(task_id, agent):
    """Claim a task for an agent"""
    tasks = load_tasks()
    task = next((t for t in tasks if t['id'] == task_id), None)
    
    if not task:
        print(f"❌ Task {task_id} not found")
        sys.exit(1)
    
    if task['status'] != 'available':
        print(f"❌ Task {task_id} is not available (status: {task['status']})")
        sys.exit(1)
    
    task['status'] = 'in_progress'
    task['agent'] = agent
    task['updated_at'] = datetime.now(UTC).isoformat()
    save_task(task)
    
    print(f"✅ Task {task_id} claimed by {agent}")
    print(f"   Python:  {task['python_file']}")
    print(f"   TypeScript: {task['typescript_file']}")

def complete_task(task_id, score=None):
    """Mark task as completed"""
    tasks = load_tasks()
    task = next((t for t in tasks if t['id'] == task_id), None)
    
    if not task:
        print(f"❌ Task {task_id} not found")
        sys.exit(1)
    
    task['status'] = 'completed'
    task['score'] = score
    task['completed_at'] = datetime.now(UTC).isoformat()
    task['updated_at'] = datetime.now(UTC).isoformat()
    save_task(task)
    
    # Unblock dependent tasks
    for t in tasks:
        if task_id in t.get('dependencies', []):
            t['dependencies'].remove(task_id)
            if not t['dependencies'] and t['status'] == 'blocked':
                t['status'] = 'available'
                save_task(t)
    
    print(f"✅ Task {task_id} completed" + (f" (score: {score}%)" if score else ""))

def list_tasks(status=None, phase=None):
    """List tasks with optional filters"""
    tasks = load_tasks()
    
    if status:
        tasks = [t for t in tasks if t['status'] == status]
    if phase:
        tasks = [t for t in tasks if t.get('phase') == phase]
    
    if not tasks:
        print("No tasks found")
        return
    
    # Sort by phase, then status, then ID
    tasks.sort(key=lambda t: (t.get('phase', 999), t['status'], t['id']))
    
    print(f"{'ID':<12} {'Status':<12} {'Phase':<6} {'Agent':<12} {'File'}")
    print("-" * 80)
    for task in tasks:
        agent = task.get('agent') or '-'
        phase_str = str(task.get('phase', '?'))
        file_name = Path(task['python_file']).name
        print(f"{task['id']:<12} {task['status']:<12} {phase_str:<6} {agent:<12} {file_name}")

def show_progress():
    """Show migration progress statistics"""
    tasks = load_tasks()
    
    if not tasks:
        print("No tasks found. Run 'init' first.")
        return
    
    total = len(tasks)
    by_status = {}
    by_phase = {}
    
    for task in tasks:
        status = task['status']
        phase = task.get('phase', 'unknown')
        
        by_status[status] = by_status.get(status, 0) + 1
        by_phase[phase] = by_phase.get(phase, 0) + 1
    
    completed = by_status.get('completed', 0)
    in_progress = by_status.get('in_progress', 0)
    available = by_status.get('available', 0)
    blocked = by_status.get('blocked', 0)
    
    pct_complete = (completed / total * 100) if total > 0 else 0
    
    print("=== Migration Progress ===")
    print(f"Total Tasks: {total}")
    print(f"Completed:   {completed:3d} ({pct_complete:5.1f}%)")
    print(f"In Progress: {in_progress:3d}")
    print(f"Available:   {available:3d}")
    print(f"Blocked:     {blocked:3d}")
    print()
    print("By Phase:")
    for phase in sorted(by_phase.keys()):
        print(f"  Phase {phase}: {by_phase[phase]}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migration task orchestrator")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init
    init_parser = subparsers.add_parser('init', help='Initialize tasks from FILE_MAPPING.json')
    init_parser.add_argument('--phase', type=int, help='Only create tasks for specific phase')
    
    # next
    subparsers.add_parser('next', help='Get next available task')
    
    # claim
    claim_parser = subparsers.add_parser('claim', help='Claim a task')
    claim_parser.add_argument('task_id', help='Task ID to claim')
    claim_parser.add_argument('--agent', required=True, help='Agent name')
    
    # complete
    complete_parser = subparsers.add_parser('complete', help='Mark task as completed')
    complete_parser.add_argument('task_id', help='Task ID to complete')
    complete_parser.add_argument('--score', type=int, help='Validation score (0-100)')
    
    # list
    list_parser = subparsers.add_parser('list', help='List tasks')
    list_parser.add_argument('--status', choices=['available', 'in_progress', 'completed', 'blocked'])
    list_parser.add_argument('--phase', type=int, help='Filter by phase')
    
    # progress
    subparsers.add_parser('progress', help='Show migration progress')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'init':
        init_tasks(args.phase)
    elif args.command == 'next':
        get_next_task()
    elif args.command == 'claim':
        claim_task(args.task_id, args.agent)
    elif args.command == 'complete':
        complete_task(args.task_id, args.score)
    elif args.command == 'list':
        list_tasks(args.status, args.phase)
    elif args.command == 'progress':
        show_progress()

if __name__ == '__main__':
    main()
