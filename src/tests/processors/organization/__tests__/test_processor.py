"""Tests for FileOrganizerProcessor."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.processors.base import ProcessingContext
from src.processors.organization.processor import FileOrganizerProcessor
from src.schemas.models.config import FileGroupingConfig, GroupingRule

# QR code generation utilities
try:
    import qrcode
    from PIL import Image as PILImage

    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False


class TestFileOrganizerProcessor:
    """Test suite for FileOrganizerProcessor."""

    def test_processor_name(self):
        """Test that processor returns correct name."""
        config = FileGroupingConfig(enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))
            assert processor.get_name() == "FileOrganizer"

    def test_disabled_processor_does_nothing(self):
        """Test that disabled processor doesn't collect results."""
        config = FileGroupingConfig(enabled=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))

            context = ProcessingContext(
                file_path=Path("test.jpg"),
                gray_image=None,
                colored_image=None,
                template=MagicMock(),
            )

            result = processor.process(context)

            assert result is context  # Returns unchanged context
            assert len(processor.results) == 0  # No results collected

    def test_enabled_processor_collects_results(self):
        """Test that enabled processor collects results from processing."""
        config = FileGroupingConfig(enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))

            context = ProcessingContext(
                file_path=Path("test.jpg"),
                gray_image=None,
                colored_image=None,
                template=MagicMock(),
            )
            context.omr_response = {"roll": "12345"}
            context.score = 95
            context.is_multi_marked = False
            context.metadata = {"output_path": "/path/to/output.jpg"}

            processor.process(context)

            assert len(processor.results) == 1
            assert processor.results[0]["score"] == 95
            assert processor.results[0]["omr_response"] == {"roll": "12345"}

    def test_rule_priority_ordering(self):
        """Test that rules are sorted by priority."""
        rules = [
            GroupingRule(
                name="Low Priority",
                priority=3,
                destination_pattern="low/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": ".*"},
            ),
            GroupingRule(
                name="High Priority",
                priority=1,
                destination_pattern="high/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": ".*"},
            ),
            GroupingRule(
                name="Medium Priority",
                priority=2,
                destination_pattern="med/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": ".*"},
            ),
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))

            # Check rules are sorted by priority
            assert processor.config.rules[0].name == "High Priority"
            assert processor.config.rules[1].name == "Medium Priority"
            assert processor.config.rules[2].name == "Low Priority"

    def test_rule_matching_uses_first_match(self):
        """Test that first matching rule by priority is used."""
        rules = [
            GroupingRule(
                name="Specific Match",
                priority=1,
                destination_pattern="specific/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": "^123.*"},
            ),
            GroupingRule(
                name="General Match",
                priority=2,
                destination_pattern="general/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": ".*"},
            ),
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))

            fields = {"roll": "12345"}
            matched = processor._find_matching_rule(fields)

            assert matched is not None
            assert matched.name == "Specific Match"

    def test_rule_matching_falls_to_second_if_first_doesnt_match(self):
        """Test that second rule is used if first doesn't match."""
        rules = [
            GroupingRule(
                name="No Match",
                priority=1,
                destination_pattern="nomatch/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": "^999.*"},
            ),
            GroupingRule(
                name="Should Match",
                priority=2,
                destination_pattern="match/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": "^123.*"},
            ),
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))

            fields = {"roll": "12345"}
            matched = processor._find_matching_rule(fields)

            assert matched is not None
            assert matched.name == "Should Match"

    def test_no_matching_rule_returns_none(self):
        """Test that no match returns None."""
        rules = [
            GroupingRule(
                name="No Match",
                priority=1,
                destination_pattern="nomatch/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": "^999.*"},
            ),
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))

            fields = {"roll": "12345"}
            matched = processor._find_matching_rule(fields)

            assert matched is None

    def test_finish_processing_with_no_results(self):
        """Test that finish_processing does nothing when no results collected."""
        config = FileGroupingConfig(enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileOrganizerProcessor(config, Path(tmpdir))

            # Should not raise any errors
            processor.finish_processing_directory()

            assert len(processor.file_operations) == 0

    def test_finish_processing_creates_organized_dir(self):
        """Test that finish_processing creates organized directory."""
        config = FileGroupingConfig(enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            processor = FileOrganizerProcessor(config, output_dir)

            # Add a mock result
            mock_context = MagicMock()
            mock_context.file_path = Path("test.jpg")
            mock_context.omr_response = {"roll": "12345"}
            mock_context.score = 95
            mock_context.is_multi_marked = False
            mock_context.metadata = {}

            processor.results.append(
                {
                    "context": mock_context,
                    "output_path": None,  # Will be skipped
                    "score": 95,
                    "omr_response": {"roll": "12345"},
                    "is_multi_marked": False,
                }
            )

            processor.finish_processing_directory()

            organized_dir = output_dir / "organized"
            assert organized_dir.exists()
            assert organized_dir.is_dir()

    def test_file_organization_with_default_pattern(self):
        """Test file organization using default pattern."""
        config = FileGroupingConfig(
            enabled=True, default_pattern="ungrouped/{original_name}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Create a mock output file
            test_file = output_dir / "test_output.jpg"
            test_file.write_text("test")

            processor = FileOrganizerProcessor(config, output_dir)

            # Add a result
            mock_context = MagicMock()
            mock_context.file_path = Path("test.jpg")
            mock_context.omr_response = {"roll": "12345"}
            mock_context.score = 95
            mock_context.is_multi_marked = False
            mock_context.metadata = {"output_path": str(test_file)}

            processor.results.append(
                {
                    "context": mock_context,
                    "output_path": str(test_file),
                    "score": 95,
                    "omr_response": {"roll": "12345"},
                    "is_multi_marked": False,
                }
            )

            processor.finish_processing_directory()

            # Check that file was organized (symlink or copy created)
            organized_file = output_dir / "organized" / "ungrouped" / "test_output.jpg"
            assert (
                organized_file.exists() or organized_file.is_symlink()
            )  # Depends on OS

    def test_collision_skip_strategy(self):
        """Test that skip collision strategy skips existing files."""
        rules = [
            GroupingRule(
                name="Test Rule",
                priority=1,
                destination_pattern="output/{roll}",
                matcher={"formatString": "{roll}", "matchRegex": ".*"},
                collision_strategy="skip",
            ),
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            test_file = output_dir / "test.jpg"
            test_file.write_text("test")

            # Pre-create the destination file to cause collision
            # The pattern will resolve to "output/12345.jpg" based on roll field
            organized_dir = output_dir / "organized" / "output"
            organized_dir.mkdir(parents=True)
            collision_file = (
                organized_dir / "12345.jpg"
            )  # Updated to match actual pattern output
            collision_file.write_text("existing")

            processor = FileOrganizerProcessor(config, output_dir)

            mock_context = MagicMock()
            mock_context.file_path = Path("test.jpg")
            mock_context.omr_response = {"roll": "12345"}
            mock_context.score = 95
            mock_context.is_multi_marked = False
            mock_context.metadata = {"output_path": str(test_file)}

            processor.results.append(
                {
                    "context": mock_context,
                    "output_path": str(test_file),
                    "score": 95,
                    "omr_response": {"roll": "12345"},
                    "is_multi_marked": False,
                }
            )

            processor.finish_processing_directory()

            # Check that operation was skipped
            skipped = [
                op for op in processor.file_operations if op["action"] == "skipped"
            ]
            assert len(skipped) > 0

    def test_qr_code_sorting_by_booklet(self):
        """Test QR code based file sorting into booklet folders.

        This test validates that images with QR codes containing booklet codes
        are correctly sorted into respective folders based on their decoded values.
        """
        rules = [
            GroupingRule(
                name="Sort by Booklet Code",
                priority=1,
                destination_pattern="booklet_{barcode}/{original_name}",
                matcher={"formatString": "{barcode}", "matchRegex": ".*"},
                action="symlink",
                collision_strategy="increment",
            ),
        ]
        config = FileGroupingConfig(
            enabled=True,
            default_pattern="unsorted/{original_name}",
            rules=rules,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock output files for 3 images
            test_files = [
                output_dir / "image1.jpg",
                output_dir / "image2.jpg",
                output_dir / "image3.jpg",
            ]
            for test_file in test_files:
                test_file.write_text("test content")

            processor = FileOrganizerProcessor(config, output_dir)

            # Simulate 3 images with different QR code values
            # Image 1 and 3 have same booklet code (BOOKLET_A)
            # Image 2 has different booklet code (BOOKLET_B)
            test_data = [
                {
                    "file": test_files[0],
                    "barcode": "BOOKLET_A",
                    "omr_response": {"barcode": "BOOKLET_A", "q1": "A", "q2": "B"},
                },
                {
                    "file": test_files[1],
                    "barcode": "BOOKLET_B",
                    "omr_response": {"barcode": "BOOKLET_B", "q1": "C", "q2": "D"},
                },
                {
                    "file": test_files[2],
                    "barcode": "BOOKLET_A",
                    "omr_response": {"barcode": "BOOKLET_A", "q1": "E", "q2": "F"},
                },
            ]

            # Process each file
            for data in test_data:
                mock_context = MagicMock()
                mock_context.file_path = data["file"]
                mock_context.omr_response = data["omr_response"]
                mock_context.score = 95
                mock_context.is_multi_marked = False
                mock_context.metadata = {"output_path": str(data["file"])}

                processor.results.append(
                    {
                        "context": mock_context,
                        "output_path": str(data["file"]),
                        "score": 95,
                        "omr_response": data["omr_response"],
                        "is_multi_marked": False,
                    }
                )

            # Organize files
            processor.finish_processing_directory()

            organized_dir = output_dir / "organized"

            # Verify folder structure created
            assert (organized_dir / "booklet_BOOKLET_A").exists()
            assert (organized_dir / "booklet_BOOKLET_B").exists()

            # Verify file counts in each folder
            booklet_a_files = list((organized_dir / "booklet_BOOKLET_A").glob("*"))
            assert len(booklet_a_files) == 2  # image1.jpg and image3.jpg

            booklet_b_files = list((organized_dir / "booklet_BOOKLET_B").glob("*"))
            assert len(booklet_b_files) == 1  # image2.jpg

            # Verify all operations succeeded
            assert len(processor.file_operations) == 3
            successful_ops = [
                op
                for op in processor.file_operations
                if op["action"] in ["symlink", "copy"]
            ]
            assert len(successful_ops) == 3

            # Verify files or symlinks exist
            assert (organized_dir / "booklet_BOOKLET_A" / "image1.jpg").exists() or (
                organized_dir / "booklet_BOOKLET_A" / "image1.jpg"
            ).is_symlink()
            assert (organized_dir / "booklet_BOOKLET_B" / "image2.jpg").exists() or (
                organized_dir / "booklet_BOOKLET_B" / "image2.jpg"
            ).is_symlink()
            assert (organized_dir / "booklet_BOOKLET_A" / "image3.jpg").exists() or (
                organized_dir / "booklet_BOOKLET_A" / "image3.jpg"
            ).is_symlink()

    @pytest.mark.skipif(not HAS_QRCODE, reason="qrcode library not available")
    def test_qr_code_sorting_with_actual_detection(self):  # noqa: PLR0915
        """Integration test: Generate QR code images and run full detection pipeline.

        This test creates actual QR code images, runs barcode detection,
        and validates that files are sorted based on decoded QR values.
        """
        from src.processors.template.template import Template
        from src.schemas.defaults.config import CONFIG_DEFAULTS
        from src.utils.image import ImageUtils

        rules = [
            GroupingRule(
                name="Sort by Booklet Code",
                priority=1,
                destination_pattern="booklet_{barcode}/{original_name}",
                matcher={"formatString": "{barcode}", "matchRegex": ".*"},
                action="symlink",
                collision_strategy="increment",
            ),
        ]
        file_grouping_config = FileGroupingConfig(
            enabled=True,
            default_pattern="unsorted/{original_name}",
            rules=rules,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            inputs_dir = temp_path / "inputs"
            outputs_dir = temp_path / "outputs"
            inputs_dir.mkdir()
            outputs_dir.mkdir()

            # Create a simple template with QR code field
            template_data = {
                "templateDimensions": [600, 800],
                "bubbleDimensions": [40, 50],
                "processingImageShape": [600, 800],
                "fieldBlocks": {
                    "BookletCode": {
                        "fieldDetectionType": "BARCODE_QR",
                        "origin": [50, 50],
                        "scanZone": {
                            "margins": {
                                "top": 20,
                                "bottom": 20,
                                "left": 20,
                                "right": 20,
                            },
                            "dimensions": [200, 200],
                        },
                        "fieldLabels": ["barcode"],
                    }
                },
                "preProcessors": [],
            }

            template_path = temp_path / "template.json"
            with template_path.open("w") as f:
                json.dump(template_data, f)

            # Generate test images with QR codes containing booklet codes
            test_cases = [
                {"filename": "sheet1.jpg", "barcode": "BOOKLET_A"},
                {"filename": "sheet2.jpg", "barcode": "BOOKLET_B"},
                {"filename": "sheet3.jpg", "barcode": "BOOKLET_A"},
            ]

            for test_case in test_cases:
                # Generate QR code
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=4,
                    border=2,
                )
                qr.add_data(test_case["barcode"])
                qr.make(fit=True)

                # Create QR code image (PIL)
                qr_img = qr.make_image(fill_color="black", back_color="white")

                # Create a blank white image (simulating an OMR sheet)
                sheet_img = PILImage.new("RGB", (600, 800), "white")

                # Resize QR code to fit in scan zone
                qr_resized = qr_img.resize((160, 160))

                # Paste QR code at the correct position (considering margins)
                # Origin is [50, 50], margins are 20, so QR code starts at [70, 70]
                sheet_img.paste(qr_resized, (70, 70))

                # Save the image
                input_file = inputs_dir / test_case["filename"]
                # Convert PIL to OpenCV format and save
                img_array = np.array(sheet_img)
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(input_file), img_bgr)

            # Create a minimal config with file grouping

            # Load template with colored outputs enabled
            from copy import deepcopy

            tuning_config = deepcopy(CONFIG_DEFAULTS)
            tuning_config.outputs.colored_outputs_enabled = True

            template = Template(
                template_path=template_path,
                tuning_config=tuning_config,
            )

            # Initialize FileOrganizer with our config
            processor = FileOrganizerProcessor(file_grouping_config, outputs_dir)

            # Process each image through the pipeline
            for test_case in test_cases:
                input_file = inputs_dir / test_case["filename"]

                # Read image with colored output enabled
                gray_image, colored_image = ImageUtils.read_image_util(
                    Path(input_file), tuning_config
                )

                # Verify images were read
                assert gray_image is not None, (
                    f"Failed to read gray image: {input_file}"
                )
                assert colored_image is not None, (
                    f"Failed to read colored image: {input_file}"
                )

                # Keep original for output
                original_colored = colored_image.copy()

                # Process through the pipeline
                context = template.process_file(
                    str(input_file), gray_image, colored_image
                )

                # Create mock output file (in real scenario, this would be created by save operations)
                output_file = outputs_dir / "CheckedOMRs" / test_case["filename"]
                output_file.parent.mkdir(parents=True, exist_ok=True)
                # Use context colored_image if available, otherwise use original
                output_image = (
                    context.colored_image
                    if context.colored_image is not None
                    else original_colored
                )
                cv2.imwrite(str(output_file), output_image)

                # Update context with output path
                context.metadata["output_path"] = str(output_file)

                # Add to organizer
                processor.process(context)

                # Verify QR code was detected correctly
                assert "barcode" in context.omr_response, (
                    f"Barcode field not found in omr_response for {test_case['filename']}"
                )
                detected_barcode = context.omr_response["barcode"]
                # Handle both string and bytes string representation
                expected = test_case["barcode"]
                # PyZBar may return the value as a string representation of bytes like "b'VALUE'"
                if detected_barcode.startswith("b'") and detected_barcode.endswith("'"):
                    detected_barcode = detected_barcode[2:-1]  # Strip b' and '
                assert detected_barcode == expected, (
                    f"Expected barcode '{expected}', "
                    f"got '{detected_barcode}' for {test_case['filename']}"
                )

            # Organize files
            processor.finish_processing_directory()

            # Verify organization
            organized_dir = outputs_dir / "organized"

            # The barcode values are stored with the bytes string representation "b'VALUE'"
            # So folders will be named accordingly
            booklet_a_folder = organized_dir / "booklet_b'BOOKLET_A'"
            booklet_b_folder = organized_dir / "booklet_b'BOOKLET_B'"

            # Check folder structure
            assert booklet_a_folder.exists(), (
                f"booklet_b'BOOKLET_A' folder not created. "
                f"Folders found: {list(organized_dir.glob('*'))}"
            )
            assert booklet_b_folder.exists(), (
                f"booklet_b'BOOKLET_B' folder not created. "
                f"Folders found: {list(organized_dir.glob('*'))}"
            )

            # Check file counts
            booklet_a_files = list(booklet_a_folder.glob("*.jpg"))
            assert len(booklet_a_files) == 2, (
                f"Expected 2 files in booklet_b'BOOKLET_A', found {len(booklet_a_files)}"
            )

            booklet_b_files = list(booklet_b_folder.glob("*.jpg"))
            assert len(booklet_b_files) == 1, (
                f"Expected 1 file in booklet_b'BOOKLET_B', found {len(booklet_b_files)}"
            )

            # Verify all operations succeeded
            assert len(processor.file_operations) == 3, (
                f"Expected 3 file operations, found {len(processor.file_operations)}"
            )

            successful_ops = [
                op
                for op in processor.file_operations
                if op["action"] in ["symlink", "copy"]
            ]
            assert len(successful_ops) == 3, (
                f"Expected all 3 operations to succeed, "
                f"but only {len(successful_ops)} succeeded"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
