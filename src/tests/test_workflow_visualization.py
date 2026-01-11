"""Tests for workflow visualization components."""

import json
from datetime import UTC, datetime
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.processors.base import ProcessingContext
from src.processors.visualization import (
    HTMLExporter,
    ImageEncoder,
    ProcessorState,
    WorkflowGraph,
    WorkflowSession,
    WorkflowTracker,
    export_to_html,
    replay_from_json,
    track_workflow,
)

# Fixtures


@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_colored_image():
    """Create a sample colored image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_context(sample_image, sample_colored_image):
    """Create a sample ProcessingContext."""
    mock_template = Mock()
    mock_template.template_name = "test_template"

    return ProcessingContext(
        file_path="test_input.jpg",
        gray_image=sample_image,
        colored_image=sample_colored_image,
        template=mock_template,
    )


@pytest.fixture
def sample_processor_state():
    """Create a sample ProcessorState."""
    return ProcessorState(
        name="TestProcessor",
        order=0,
        timestamp=datetime.now(UTC).isoformat(),
        duration_ms=100.5,
        image_shape=(100, 100),
        gray_image_base64="base64encodedimage",
        metadata={"test_key": "test_value"},
        success=True,
    )


@pytest.fixture
def sample_session(sample_processor_state):
    """Create a sample WorkflowSession."""
    session = WorkflowSession(
        session_id="test_session",
        file_path="test_input.jpg",
        template_name="test_template",
        start_time=datetime.now(UTC).isoformat(),
    )
    session.add_processor_state(sample_processor_state)
    session.finalize(datetime.now(UTC).isoformat(), 150.0)
    return session


# Tests for ImageEncoder


class TestImageEncoder:
    """Tests for ImageEncoder utility class."""

    def test_encode_image(self, sample_image):
        """Test encoding an image to base64."""
        encoded = ImageEncoder.encode_image(sample_image, max_width=50, quality=85)

        assert encoded is not None
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_encode_none_image(self):
        """Test encoding None returns None."""
        encoded = ImageEncoder.encode_image(None)
        assert encoded is None

    def test_encode_with_resize(self, sample_image):
        """Test encoding with resizing."""
        # Create a wider image
        wide_image = np.random.randint(0, 255, (100, 1000), dtype=np.uint8)
        encoded = ImageEncoder.encode_image(wide_image, max_width=200)

        assert encoded is not None
        # Decode and check dimensions
        decoded = ImageEncoder.decode_image(encoded)
        assert decoded.shape[1] <= 200  # Width should be <= max_width

    def test_decode_image(self, sample_image):
        """Test decoding a base64 image."""
        encoded = ImageEncoder.encode_image(sample_image)
        decoded = ImageEncoder.decode_image(encoded)

        assert decoded is not None
        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == sample_image.shape

    def test_get_data_uri(self):
        """Test generating data URI."""
        base64_str = "abc123"
        uri = ImageEncoder.get_data_uri(base64_str)

        assert uri == "data:image/jpeg;base64,abc123"


# Tests for ProcessorState


class TestProcessorState:
    """Tests for ProcessorState data model."""

    def test_creation(self, sample_processor_state):
        """Test creating a ProcessorState."""
        assert sample_processor_state.name == "TestProcessor"
        assert sample_processor_state.order == 0
        assert sample_processor_state.duration_ms == 100.5
        assert sample_processor_state.success is True

    def test_to_dict(self, sample_processor_state):
        """Test converting to dictionary."""
        data = sample_processor_state.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "TestProcessor"
        assert data["order"] == 0
        assert data["duration_ms"] == 100.5
        assert data["metadata"]["test_key"] == "test_value"

    def test_error_state(self):
        """Test creating a failed processor state."""
        state = ProcessorState(
            name="FailedProcessor",
            order=1,
            timestamp=datetime.now(UTC).isoformat(),
            duration_ms=50.0,
            image_shape=(100, 100),
            success=False,
            error_message="Test error",
        )

        assert state.success is False
        assert state.error_message == "Test error"


# Tests for WorkflowGraph


class TestWorkflowGraph:
    """Tests for WorkflowGraph data model."""

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = WorkflowGraph()
        graph.add_node("node1", "Node 1", {"type": "processor"})

        assert len(graph.nodes) == 1
        assert graph.nodes[0]["id"] == "node1"
        assert graph.nodes[0]["label"] == "Node 1"
        assert graph.nodes[0]["metadata"]["type"] == "processor"

    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = WorkflowGraph()
        graph.add_edge("node1", "node2", "processes")

        assert len(graph.edges) == 1
        assert graph.edges[0]["from"] == "node1"
        assert graph.edges[0]["to"] == "node2"
        assert graph.edges[0]["label"] == "processes"

    def test_to_dict(self):
        """Test converting graph to dictionary."""
        graph = WorkflowGraph()
        graph.add_node("node1", "Node 1")
        graph.add_edge("node1", "node2")

        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1


# Tests for WorkflowSession


class TestWorkflowSession:
    """Tests for WorkflowSession data model."""

    def test_creation(self):
        """Test creating a WorkflowSession."""
        session = WorkflowSession(
            session_id="test_session",
            file_path="test.jpg",
            template_name="template1",
            start_time=datetime.now(UTC).isoformat(),
        )

        assert session.session_id == "test_session"
        assert session.file_path == "test.jpg"
        assert session.template_name == "template1"
        assert len(session.processor_states) == 0

    def test_add_processor_state(self, sample_session, sample_processor_state):
        """Test adding processor states."""
        initial_count = len(sample_session.processor_states)

        new_state = ProcessorState(
            name="NewProcessor",
            order=1,
            timestamp=datetime.now(UTC).isoformat(),
            duration_ms=75.0,
            image_shape=(100, 100),
        )
        sample_session.add_processor_state(new_state)

        assert len(sample_session.processor_states) == initial_count + 1

    def test_finalize(self):
        """Test finalizing a session."""
        session = WorkflowSession(
            session_id="test",
            file_path="test.jpg",
            template_name="template",
            start_time=datetime.now(UTC).isoformat(),
        )

        end_time = datetime.now(UTC).isoformat()
        session.finalize(end_time, 200.0)

        assert session.end_time == end_time
        assert session.total_duration_ms == 200.0

    def test_to_dict(self, sample_session):
        """Test converting session to dictionary."""
        data = sample_session.to_dict()

        assert isinstance(data, dict)
        assert "session_id" in data
        assert "processor_states" in data
        assert "graph" in data
        assert isinstance(data["processor_states"], list)

    def test_to_json(self, sample_session):
        """Test converting session to JSON."""
        json_str = sample_session.to_json()

        assert isinstance(json_str, str)
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["session_id"] == "test_session"

    def test_save_and_load(self, sample_session, tmp_path):
        """Test saving and loading session from file."""
        file_path = tmp_path / "session.json"

        # Save
        sample_session.save_to_file(file_path)
        assert file_path.exists()

        # Load
        loaded_session = WorkflowSession.load_from_file(file_path)
        assert loaded_session.session_id == sample_session.session_id
        assert loaded_session.file_path == sample_session.file_path
        assert len(loaded_session.processor_states) == len(
            sample_session.processor_states
        )

    def test_from_dict(self, sample_session):
        """Test creating session from dictionary."""
        data = sample_session.to_dict()
        new_session = WorkflowSession.from_dict(data)

        assert new_session.session_id == sample_session.session_id
        assert new_session.file_path == sample_session.file_path
        assert len(new_session.processor_states) == len(sample_session.processor_states)


# Tests for WorkflowTracker


class TestWorkflowTracker:
    """Tests for WorkflowTracker."""

    def test_initialization(self):
        """Test initializing a WorkflowTracker."""
        tracker = WorkflowTracker(
            file_path="test.jpg",
            template_name="template1",
            capture_processors=["Processor1", "Processor2"],
        )

        assert tracker.session.file_path == "test.jpg"
        assert tracker.session.template_name == "template1"
        assert tracker.capture_processors == ["Processor1", "Processor2"]

    def test_should_capture_all(self):
        """Test capturing all processors."""
        tracker = WorkflowTracker("test.jpg", capture_processors=["all"])

        assert tracker.should_capture("AnyProcessor") is True
        assert tracker.should_capture("AnotherProcessor") is True

    def test_should_capture_specific(self):
        """Test capturing specific processors."""
        tracker = WorkflowTracker(
            "test.jpg", capture_processors=["Processor1", "Processor2"]
        )

        assert tracker.should_capture("Processor1") is True
        assert tracker.should_capture("Processor2") is True
        assert tracker.should_capture("Processor3") is False

    def test_capture_state(self, sample_context):
        """Test capturing processor state."""
        tracker = WorkflowTracker("test.jpg")

        tracker.start_processor("TestProcessor")
        tracker.capture_state("TestProcessor", sample_context)

        assert len(tracker.session.processor_states) == 1
        state = tracker.session.processor_states[0]
        assert state.name == "TestProcessor"
        assert state.success is True

    def test_capture_error_state(self, sample_context):
        """Test capturing failed processor state."""
        tracker = WorkflowTracker("test.jpg")

        tracker.start_processor("FailedProcessor")
        tracker.capture_state(
            "FailedProcessor", sample_context, success=False, error_message="Test error"
        )

        state = tracker.session.processor_states[0]
        assert state.success is False
        assert state.error_message == "Test error"

    def test_build_graph(self):
        """Test building workflow graph."""
        tracker = WorkflowTracker("test.jpg")
        processor_names = ["Processor1", "Processor2", "Processor3"]

        tracker.build_graph(processor_names)

        graph = tracker.session.graph
        # Should have: input + 3 processors + output = 5 nodes
        assert len(graph.nodes) == 5
        # Should have: input->p1, p1->p2, p2->p3, p3->output = 4 edges
        assert len(graph.edges) == 4

    def test_finalize(self):
        """Test finalizing tracker."""
        tracker = WorkflowTracker("test.jpg")

        session = tracker.finalize()

        assert session.end_time is not None
        assert session.total_duration_ms is not None
        assert session.total_duration_ms >= 0


# Tests for HTMLExporter


class TestHTMLExporter:
    """Tests for HTMLExporter."""

    def test_initialization(self):
        """Test initializing HTMLExporter."""
        exporter = HTMLExporter()
        assert exporter.template_path.exists()

    def test_export(self, sample_session, tmp_path):
        """Test exporting session to HTML."""
        exporter = HTMLExporter()
        output_path = tmp_path / "visualization.html"

        result_path = exporter.export(
            sample_session, output_path, title="Test Visualization", open_browser=False
        )

        assert result_path.exists()
        content = result_path.read_text()
        assert "Test Visualization" in content
        assert "test_session" in content
        assert "vis-network" in content  # Should include vis.js reference

    def test_export_with_json(self, sample_session, tmp_path):
        """Test exporting with JSON data file."""
        exporter = HTMLExporter()

        html_path, json_path = exporter.export_with_json(
            sample_session, tmp_path, title="Test Visualization", open_browser=False
        )

        assert html_path.exists()
        assert json_path.exists()

        # Check JSON content
        loaded_session = WorkflowSession.load_from_file(json_path)
        assert loaded_session.session_id == sample_session.session_id


# Tests for high-level functions


class TestHighLevelFunctions:
    """Tests for high-level utility functions."""

    def test_export_to_html(self, sample_session, tmp_path):
        """Test export_to_html function."""
        html_path = export_to_html(
            sample_session,
            tmp_path,
            title="Test Export",
            open_browser=False,
            export_json=True,
        )

        assert html_path.exists()
        assert (tmp_path / "sessions" / f"{sample_session.session_id}.json").exists()

    def test_replay_from_json(self, sample_session, tmp_path):
        """Test replaying from JSON file."""
        # First save a session
        json_path = tmp_path / "session.json"
        sample_session.save_to_file(json_path)

        # Replay
        html_path = replay_from_json(json_path, output_dir=tmp_path, open_browser=False)

        assert html_path.exists()
        content = html_path.read_text()
        assert sample_session.session_id in content

    @patch("src.processors.pipeline.ProcessingPipeline")
    @patch("src.processors.template.template.Template")
    @patch("src.processors.visualization.workflow_tracker.ImageUtils")
    def test_track_workflow(  # noqa: PLR0913
        self,
        mock_image_utils,
        mock_template,
        mock_pipeline,
        sample_image,
        tmp_path,
    ):
        """Test track_workflow function."""
        # Setup mocks
        mock_image_utils.read_image_util.return_value = (sample_image, None)

        mock_template_instance = Mock()
        mock_template_instance.template_name = "test_template"
        mock_template.return_value = mock_template_instance

        mock_pipeline_instance = Mock()
        mock_pipeline_instance.get_processor_names.return_value = ["TestProcessor"]
        mock_pipeline_instance.processors = []
        mock_pipeline.return_value = mock_pipeline_instance

        # Create dummy files
        template_path = tmp_path / "template.json"
        template_path.write_text("{}")
        input_path = tmp_path / "input.jpg"
        cv2.imwrite(str(input_path), sample_image)

        # Track workflow
        session = track_workflow(
            file_path=input_path,
            template_path=template_path,
            capture_processors=["all"],
        )

        assert session is not None
        assert isinstance(session, WorkflowSession)
        assert session.template_name == "template"  # From template_path stem


# Integration Tests


class TestIntegration:
    """Integration tests for the full workflow."""

    @pytest.mark.integration
    def test_full_workflow(self, sample_image, tmp_path):
        """Test complete workflow from tracking to HTML export."""
        # Create a mock workflow session
        session = WorkflowSession(
            session_id="integration_test",
            file_path="test.jpg",
            template_name="test_template",
            start_time=datetime.now(UTC).isoformat(),
        )

        # Add some states
        for i in range(3):
            encoded_image = ImageEncoder.encode_image(sample_image)
            state = ProcessorState(
                name=f"Processor{i}",
                order=i,
                timestamp=datetime.now(UTC).isoformat(),
                duration_ms=50.0 * (i + 1),
                image_shape=sample_image.shape,
                gray_image_base64=encoded_image,
                success=True,
            )
            session.add_processor_state(state)

        # Build graph
        session.graph.add_node("input", "Input")
        for i in range(3):
            session.graph.add_node(f"p{i}", f"Processor{i}")
        session.graph.add_node("output", "Output")

        for i in range(4):
            from_node = "input" if i == 0 else f"p{i - 1}"
            to_node = "output" if i == 3 else f"p{i}"
            session.graph.add_edge(from_node, to_node)

        session.finalize(datetime.now(UTC).isoformat(), 200.0)

        # Export to HTML
        html_path = export_to_html(
            session, tmp_path, open_browser=False, export_json=True
        )

        # Verify outputs
        assert html_path.exists()
        assert (tmp_path / "sessions" / f"{session.session_id}.json").exists()

        # Verify HTML content
        content = html_path.read_text()
        assert "Processor0" in content
        assert "Processor1" in content
        assert "Processor2" in content
        assert session.session_id in content
