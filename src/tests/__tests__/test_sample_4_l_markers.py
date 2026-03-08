import pytest


@pytest.mark.sample_4_l_markers
def test_sample_4_l_markers_template_loads():
    """Verify the L-markers sample template loads and validates without errors."""
    import json
    from pathlib import Path

    template_path = Path("samples/4-l-markers/template.json")
    assert template_path.exists(), "Sample template not found"

    # Template should parse without raising
    # (full pipeline test requires an actual image in inputs/)
    data = json.loads(template_path.read_text())
    assert data["preProcessors"][0]["options"]["type"] == "L_MARKERS"
