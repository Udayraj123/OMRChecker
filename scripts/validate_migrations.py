"""Validation script to check that migrated JSONs pass schema validation."""

import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.validations import (
    validate_config_json,
    validate_template_json,
    validate_evaluation_json,
)
from src.utils.exceptions import (
    ConfigValidationError,
    TemplateValidationError,
    EvaluationValidationError,
)


def validate_sample_files():
    """Validate all sample JSON files against updated schemas."""
    base_dir = Path(__file__).parent.parent
    samples_dir = base_dir / "samples"

    errors = []
    success_count = 0

    # Find and validate all config.json files
    for config_path in samples_dir.rglob("config.json"):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            validate_config_json(config_data, config_path)
            print(f"✓ Config: {config_path.relative_to(base_dir)}")
            success_count += 1
        except (ConfigValidationError, Exception) as e:
            error_msg = f"✗ Config: {config_path.relative_to(base_dir)}: {e}"
            print(error_msg)
            errors.append(error_msg)

    # Find and validate all template.json files
    for template_path in samples_dir.rglob("template*.json"):
        try:
            with open(template_path, "r") as f:
                template_data = json.load(f)
            validate_template_json(template_data, template_path)
            print(f"✓ Template: {template_path.relative_to(base_dir)}")
            success_count += 1
        except (TemplateValidationError, Exception) as e:
            error_msg = f"✗ Template: {template_path.relative_to(base_dir)}: {e}"
            print(error_msg)
            errors.append(error_msg)

    # Find and validate all evaluation.json files
    for eval_path in samples_dir.rglob("evaluation.json"):
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            validate_evaluation_json(eval_data, eval_path)
            print(f"✓ Evaluation: {eval_path.relative_to(base_dir)}")
            success_count += 1
        except (EvaluationValidationError, Exception) as e:
            error_msg = f"✗ Evaluation: {eval_path.relative_to(base_dir)}: {e}"
            print(error_msg)
            errors.append(error_msg)

    print("=" * 80)
    if errors:
        print(f"Validation failed: {len(errors)} errors, {success_count} successes")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return 1
    else:
        print(
            f"✓ All validations passed! {success_count} files validated successfully."
        )
        return 0


if __name__ == "__main__":
    exit(validate_sample_files())
