"""Data models for workflow visualization and tracking."""

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike


@dataclass
class ProcessorState:
    """Represents the state of a processor at a specific point in execution.

    Attributes:
        name: Human-readable name of the processor
        order: Execution order (0-indexed)
        timestamp: ISO format timestamp when processor executed
        duration_ms: Time taken to execute in milliseconds
        image_shape: Shape of the output image (height, width, channels)
        gray_image_base64: Base64-encoded JPEG of grayscale output image
        colored_image_base64: Base64-encoded JPEG of colored output image (optional)
        metadata: Additional processor-specific metadata
        success: Whether the processor executed successfully
        error_message: Error message if processor failed
    """

    name: str
    order: int
    timestamp: str
    duration_ms: float
    image_shape: tuple[int, ...]
    gray_image_base64: str | None = None
    colored_image_base64: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class WorkflowGraph:
    """Represents the processor workflow as a graph structure.

    Attributes:
        nodes: List of node definitions with id, label, and metadata
        edges: List of edge definitions connecting nodes
    """

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def add_node(
        self, node_id: str, label: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a node to the graph.

        Args:
            node_id: Unique identifier for the node
            label: Display label for the node
            metadata: Additional node metadata
        """
        self.nodes.append({"id": node_id, "label": label, "metadata": metadata or {}})

    def add_edge(self, from_id: str, to_id: str, label: str | None = None) -> None:
        """Add an edge to the graph.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            label: Optional edge label
        """
        edge = {"from": from_id, "to": to_id}
        if label:
            edge["label"] = label
        self.edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"nodes": self.nodes, "edges": self.edges}


@dataclass
class WorkflowSession:
    """Complete workflow execution session data.

    This encapsulates all data needed to visualize and replay a workflow execution,
    including processor states, images, timing information, and graph structure.

    Attributes:
        session_id: Unique identifier for this session
        file_path: Path to the input file being processed
        template_name: Name of the template used
        start_time: ISO format timestamp when session started
        end_time: ISO format timestamp when session ended
        total_duration_ms: Total execution time in milliseconds
        processor_states: List of processor states in execution order
        graph: Workflow graph structure
        config: Configuration used for this session
        metadata: Additional session metadata
    """

    session_id: str
    file_path: str
    template_name: str
    start_time: str
    end_time: str | None = None
    total_duration_ms: float | None = None
    processor_states: list[ProcessorState] = field(default_factory=list)
    graph: WorkflowGraph = field(default_factory=WorkflowGraph)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_processor_state(self, state: ProcessorState) -> None:
        """Add a processor state to the session.

        Args:
            state: ProcessorState to add
        """
        self.processor_states.append(state)

    def finalize(self, end_time: str, total_duration_ms: float) -> None:
        """Finalize the session with end time and total duration.

        Args:
            end_time: ISO format timestamp when session ended
            total_duration_ms: Total execution time in milliseconds
        """
        self.end_time = end_time
        self.total_duration_ms = total_duration_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "file_path": self.file_path,
            "template_name": self.template_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "processor_states": [state.to_dict() for state in self.processor_states],
            "graph": self.graph.to_dict(),
            "config": self.config,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: Number of spaces for JSON indentation

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_file(self, file_path: Path | str) -> None:
        """Save session to a JSON file.

        Args:
            file_path: Path where to save the JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowSession":
        """Create WorkflowSession from dictionary.

        Args:
            data: Dictionary containing session data

        Returns:
            WorkflowSession instance
        """
        # Reconstruct processor states
        processor_states = [
            ProcessorState(**state_data)
            for state_data in data.get("processor_states", [])
        ]

        # Reconstruct graph
        graph_data = data.get("graph", {})
        graph = WorkflowGraph(
            nodes=graph_data.get("nodes", []), edges=graph_data.get("edges", [])
        )

        return cls(
            session_id=data["session_id"],
            file_path=data["file_path"],
            template_name=data["template_name"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            total_duration_ms=data.get("total_duration_ms"),
            processor_states=processor_states,
            graph=graph,
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowSession":
        """Create WorkflowSession from JSON string.

        Args:
            json_str: JSON string containing session data

        Returns:
            WorkflowSession instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> "WorkflowSession":
        """Load session from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            WorkflowSession instance
        """
        file_path = Path(file_path)
        json_str = file_path.read_text()
        return cls.from_json(json_str)


class ImageEncoder:
    """Utility class for encoding images to base64."""

    @staticmethod
    def encode_image(
        image: MatLike, max_width: int | None = 800, quality: int = 85
    ) -> str | None:
        """Encode an image to base64-encoded JPEG.

        Args:
            image: Input image (numpy array)
            max_width: Maximum width for resizing (maintains aspect ratio)
            quality: JPEG quality (0-100)

        Returns:
            Base64-encoded JPEG string or None if image is None
        """
        if image is None:
            return None

        # Resize if needed
        if max_width is not None:
            h, w = image.shape[:2]
            if w > max_width:
                scale = max_width / w
                new_width = max_width
                new_height = int(h * scale)
                image = cv2.resize(image, (new_width, new_height))

        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", image, encode_param)

        # Convert to base64
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def decode_image(base64_str: str) -> MatLike:
        """Decode a base64-encoded JPEG to image.

        Args:
            base64_str: Base64-encoded JPEG string

        Returns:
            Decoded image as numpy array
        """
        # Decode base64
        img_data = base64.b64decode(base64_str)

        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)

        # Decode image
        return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def get_data_uri(base64_str: str, mime_type: str = "image/jpeg") -> str:
        """Convert base64 string to data URI for HTML embedding.

        Args:
            base64_str: Base64-encoded image string
            mime_type: MIME type of the image

        Returns:
            Data URI string
        """
        return f"data:{mime_type};base64,{base64_str}"
