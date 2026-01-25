"""Tests for dataclass serialization utilities."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.utils.serialization import dataclass_to_dict


class Color(Enum):
    """Test enum."""

    RED = "red"
    BLUE = "blue"


@dataclass
class NestedConfig:
    """Test nested dataclass."""

    value: int = 10
    name: str = "test"


@dataclass
class ComplexConfig:
    """Test complex dataclass with various types."""

    path: Path
    nested: NestedConfig
    numbers: list[int] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    color: Color = Color.RED
    flag: bool = True


def test_simple_dataclass_serialization():
    """Test serialization of a simple dataclass."""
    config = NestedConfig(value=42, name="hello")
    result = dataclass_to_dict(config)

    assert result == {"value": 42, "name": "hello"}
    assert isinstance(result, dict)


def test_nested_dataclass_serialization():
    """Test serialization of nested dataclasses."""
    config = ComplexConfig(
        path=Path("/tmp/test.txt"), nested=NestedConfig(value=100, name="nested")
    )
    result = dataclass_to_dict(config)

    assert result["path"] == "/tmp/test.txt"
    assert result["nested"] == {"value": 100, "name": "nested"}
    assert result["flag"] is True


def test_path_serialization():
    """Test that Path objects are converted to strings."""
    config = ComplexConfig(path=Path("/home/user/file.json"), nested=NestedConfig())
    result = dataclass_to_dict(config)

    assert result["path"] == "/home/user/file.json"
    assert isinstance(result["path"], str)


def test_enum_serialization():
    """Test that Enum values are converted properly."""
    config = ComplexConfig(path=Path("/tmp"), nested=NestedConfig(), color=Color.BLUE)
    result = dataclass_to_dict(config)

    assert result["color"] == "blue"


def test_list_and_dict_serialization():
    """Test serialization of lists and dictionaries."""
    config = ComplexConfig(
        path=Path("/tmp"),
        nested=NestedConfig(),
        numbers=[1, 2, 3, 4, 5],
        mapping={"key1": "value1", "key2": "value2"},
    )
    result = dataclass_to_dict(config)

    assert result["numbers"] == [1, 2, 3, 4, 5]
    assert result["mapping"] == {"key1": "value1", "key2": "value2"}


def test_nested_list_of_dataclasses():
    """Test serialization of lists containing dataclasses."""

    @dataclass
    class Container:
        items: list[NestedConfig]

    container = Container(
        items=[
            NestedConfig(value=1, name="first"),
            NestedConfig(value=2, name="second"),
        ]
    )
    result = dataclass_to_dict(container)

    assert result["items"] == [
        {"value": 1, "name": "first"},
        {"value": 2, "name": "second"},
    ]


def test_primitive_types():
    """Test that primitive types pass through correctly."""
    assert dataclass_to_dict("hello") == "hello"
    assert dataclass_to_dict(42) == 42
    assert dataclass_to_dict(3.14) == 3.14
    assert dataclass_to_dict(True) is True
    assert dataclass_to_dict(None) is None


def test_dict_with_nested_dataclasses():
    """Test serialization of dictionaries containing dataclasses."""

    @dataclass
    class Wrapper:
        configs: dict[str, NestedConfig]

    wrapper = Wrapper(
        configs={
            "config1": NestedConfig(value=10, name="first"),
            "config2": NestedConfig(value=20, name="second"),
        }
    )
    result = dataclass_to_dict(wrapper)

    assert result["configs"]["config1"] == {"value": 10, "name": "first"}
    assert result["configs"]["config2"] == {"value": 20, "name": "second"}
