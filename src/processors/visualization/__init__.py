"""Workflow visualization module for OMRChecker.

This module provides tools for visualizing and tracking the OMR processing pipeline,
capturing intermediate images and generating interactive HTML visualizations.

Main components:
- WorkflowSession: Data model for workflow execution
- WorkflowTracker: Tracks and captures processor states
- HTMLExporter: Generates interactive HTML visualizations
- track_workflow: High-level function to track a complete workflow
- export_to_html: High-level function to export visualization
"""

from src.processors.visualization.html_exporter import (
    HTMLExporter,
    export_to_html,
    replay_from_json,
)
from src.processors.visualization.workflow_session import (
    ImageEncoder,
    ProcessorState,
    WorkflowGraph,
    WorkflowSession,
)
from src.processors.visualization.workflow_tracker import (
    WorkflowTracker,
    track_workflow,
)

__all__ = [
    "HTMLExporter",
    "ImageEncoder",
    "ProcessorState",
    "WorkflowGraph",
    "WorkflowSession",
    "WorkflowTracker",
    "export_to_html",
    "replay_from_json",
    "track_workflow",
]
