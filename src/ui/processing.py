"""Pipeline wrapper: run preprocessors exactly as main.py does."""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .common import LOG

# ---------------------------------------------------------------------------
# Try importing main pipeline modules; record error for UI display
# ---------------------------------------------------------------------------
try:
    from src.defaults.config import CONFIG_DEFAULTS  # type: ignore
    from src.template import Template  # type: ignore
    from src.utils.image import ImageUtils  # type: ignore
    from src.utils.parsing import open_config_with_defaults  # type: ignore

    PIPELINE_IMPORT_ERROR: Optional[str] = None
except Exception as e:
    PIPELINE_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    LOG.error(
        f"Qt editor: pipeline import failed — {PIPELINE_IMPORT_ERROR}\n"
        "Tip: run from repo root with dependencies installed."
    )
    CONFIG_DEFAULTS = None  # type: ignore
    open_config_with_defaults = None  # type: ignore
    Template = None  # type: ignore
    ImageUtils = None  # type: ignore


def run_preprocessors_for_editor(template_path: Path, image_path: Path) -> np.ndarray:
    """
    Run the full preprocessing pipeline exactly like main.py and return the
    processed (cropped + aligned) grayscale image.

    Raises RuntimeError with a descriptive message on failure.
    """
    if Template is None or CONFIG_DEFAULTS is None:
        raise RuntimeError(
            "Pipeline modules are not available.\n"
            + (f"Details: {PIPELINE_IMPORT_ERROR}" if PIPELINE_IMPORT_ERROR else "")
        )

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    template_dir = template_path.parent
    cfg = CONFIG_DEFAULTS
    cfg_path = template_dir / "config.json"
    if open_config_with_defaults is not None and cfg_path.exists():
        cfg = open_config_with_defaults(cfg_path)

    tmpl = Template(template_path, cfg)
    processed = tmpl.image_instance_ops.apply_preprocessors(str(image_path), img, tmpl)
    if processed is None:
        raise RuntimeError(
            "Preprocessors returned None — cropping/alignment failed.\n"
            "Check that preProcessors in template.json are correct and the image is valid."
        )

    # Resize to template page dimensions for overlay consistency
    try:
        pw, ph = tmpl.page_dimensions
        if int(pw) > 0 and int(ph) > 0 and ImageUtils is not None:
            processed = ImageUtils.resize_util(processed, int(pw), int(ph))
    except Exception:
        pass

    return processed


def run_preprocessors_stepwise(template_path: Path, image_path: Path) -> list:
    """
    Run preprocessors one step at a time. Returns list of (step_name, ndarray).
    Falls back to [('original', img)] if pipeline unavailable.
    """
    import copy as _copy
    import json

    if Template is None:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return [("original", img)]
        return []

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    steps = [("original", img.copy())]

    # Load template JSON to iterate preprocessors manually
    try:
        with open(template_path, encoding="utf-8") as f:
            tmpl_data = json.load(f)
        processors = tmpl_data.get("preProcessors", [])
    except Exception:
        processors = []

    # For each processor, build a truncated template and run
    for i, proc in enumerate(processors):
        partial_data = _copy.deepcopy(tmpl_data)
        partial_data["preProcessors"] = processors[: i + 1]

        import os
        import tempfile

        # Write the temp file into the original template's directory so that
        # relative asset paths (e.g. omr_marker.jpg) resolve correctly.
        with tempfile.NamedTemporaryFile(
            dir=template_path.parent,
            suffix=".json",
            delete=False,
            mode="w",
            encoding="utf-8",
            prefix=".omr_preview_",
        ) as tmp:
            json.dump(partial_data, tmp, indent=2)
            tmp_path = Path(tmp.name)

        try:
            template_dir = template_path.parent
            cfg = CONFIG_DEFAULTS
            cfg_path = template_dir / "config.json"
            if open_config_with_defaults is not None and cfg_path.exists():
                cfg = open_config_with_defaults(cfg_path)
            tmpl = Template(tmp_path, cfg)
            result = tmpl.image_instance_ops.apply_preprocessors(
                str(image_path), img.copy(), tmpl
            )
            if result is not None:
                name = proc.get("name", f"step_{i + 1}")
                steps.append((name, result.copy()))
        except Exception as e:
            LOG.warning(f"Stepwise preview step {i + 1} failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return steps
