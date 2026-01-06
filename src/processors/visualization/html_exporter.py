"""HTML exporter for workflow visualization.

This module generates interactive HTML visualizations from WorkflowSession data,
embedding all images and creating a standalone, shareable file.
"""

import json
import webbrowser
from pathlib import Path

from src.processors.visualization.workflow_session import WorkflowSession
from src.utils.logger import logger


class HTMLExporter:
    """Exports workflow sessions to interactive HTML visualizations.

    This class takes a WorkflowSession and generates a standalone HTML file
    with an embedded flowchart, image viewer, and playback controls.

    Attributes:
        template_path: Path to the HTML template file
    """

    def __init__(self, template_path: Path | str | None = None) -> None:
        """Initialize the HTML exporter.

        Args:
            template_path: Path to custom HTML template (uses default if None)
        """
        if template_path is None:
            # Use default template from the package
            template_path = Path(__file__).parent / "templates" / "viewer.html"

        self.template_path = Path(template_path)

        if not self.template_path.exists():
            msg = f"Template file not found: {self.template_path}"
            raise FileNotFoundError(msg)

    def export(
        self,
        session: WorkflowSession,
        output_path: Path | str,
        title: str | None = None,
        open_browser: bool = False,
    ) -> Path:
        """Export a workflow session to an HTML file.

        Args:
            session: WorkflowSession to export
            output_path: Path where to save the HTML file
            title: Custom title for the visualization (uses session_id if None)
            open_browser: Whether to automatically open in browser

        Returns:
            Path to the generated HTML file
        """
        output_path = Path(output_path)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate title
        if title is None:
            title = f"OMR Workflow Visualization - {session.session_id}"

        # Read template
        template_content = self.template_path.read_text()

        # Prepare session data as JSON
        session_json = json.dumps(session.to_dict())

        # Format duration for display
        duration_str = "N/A"
        if session.total_duration_ms is not None:
            duration_str = f"{session.total_duration_ms:.2f}"

        # Substitute template variables
        html_content = template_content.replace("{{ title }}", title)
        html_content = html_content.replace("{{ session_id }}", session.session_id)
        html_content = html_content.replace("{{ file_path }}", session.file_path)
        html_content = html_content.replace("{{ total_duration_ms }}", duration_str)
        html_content = html_content.replace("{{ session_data }}", session_json)

        # Write output file
        output_path.write_text(html_content)

        logger.info(f"Exported workflow visualization to: {output_path}")

        # Open in browser if requested
        if open_browser:
            webbrowser.open(f"file://{output_path.absolute()}")
            logger.info("Opened visualization in browser")

        return output_path

    def export_with_json(
        self,
        session: WorkflowSession,
        output_dir: Path | str,
        title: str | None = None,
        open_browser: bool = False,
    ) -> tuple[Path, Path]:
        """Export workflow session to both HTML and JSON files.

        This creates:
        - An HTML visualization file
        - A JSON data file for later replay

        Args:
            session: WorkflowSession to export
            output_dir: Directory where to save files
            title: Custom title for the visualization
            open_browser: Whether to automatically open HTML in browser

        Returns:
            Tuple of (html_path, json_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate file names
        base_name = session.session_id
        html_path = output_dir / f"{base_name}.html"
        json_path = output_dir / "sessions" / f"{base_name}.json"

        # Export HTML
        html_path = self.export(session, html_path, title, open_browser)

        # Export JSON
        session.save_to_file(json_path)
        logger.info(f"Exported session JSON to: {json_path}")

        return html_path, json_path


def export_to_html(
    session: WorkflowSession,
    output_dir: Path | str,
    title: str | None = None,
    open_browser: bool = True,
    export_json: bool = True,
) -> Path:
    """High-level function to export a workflow session to HTML.

    This is a convenience function that creates an HTMLExporter and
    exports the session.

    Args:
        session: WorkflowSession to export
        output_dir: Directory where to save the visualization
        title: Custom title for the visualization
        open_browser: Whether to automatically open in browser
        export_json: Whether to also export JSON data file

    Returns:
        Path to the generated HTML file

    Example:
        >>> from src.processors.visualization.workflow_tracker import track_workflow
        >>> session = track_workflow(
        ...     file_path="inputs/sample1/sample1.jpg",
        ...     template_path="inputs/sample1/template.json"
        ... )
        >>> html_path = export_to_html(session, "outputs/visualization")
        >>> print(f"Visualization saved to: {html_path}")
    """
    exporter = HTMLExporter()

    if export_json:
        html_path, json_path = exporter.export_with_json(
            session, output_dir, title, open_browser
        )
        logger.info(f"Exported visualization: HTML={html_path}, JSON={json_path}")
        return html_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{session.session_id}.html"
    return exporter.export(session, html_path, title, open_browser)


def replay_from_json(
    json_path: Path | str,
    output_dir: Path | str | None = None,
    title: str | None = None,
    open_browser: bool = True,
) -> Path:
    """Replay a workflow visualization from a saved JSON file.

    This loads a previously saved WorkflowSession JSON and generates
    a new HTML visualization from it.

    Args:
        json_path: Path to the session JSON file
        output_dir: Directory where to save the HTML (defaults to JSON directory)
        title: Custom title for the visualization
        open_browser: Whether to automatically open in browser

    Returns:
        Path to the generated HTML file

    Example:
        >>> html_path = replay_from_json(
        ...     "outputs/visualization/sessions/session_20240106_123456_abcd1234.json"
        ... )
    """
    json_path = Path(json_path)

    if not json_path.exists():
        msg = f"JSON file not found: {json_path}"
        raise FileNotFoundError(msg)

    # Load session from JSON
    session = WorkflowSession.load_from_file(json_path)
    logger.info(f"Loaded session from: {json_path}")

    # Determine output directory
    if output_dir is None:
        output_dir = json_path.parent.parent  # Go up from 'sessions' directory

    # Export to HTML
    return export_to_html(
        session, output_dir, title=title, open_browser=open_browser, export_json=False
    )
