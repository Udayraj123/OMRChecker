import importlib

import screeninfo

import src.utils.interaction as interaction


def test_import_survives_missing_display():
    def no_display():
        raise RuntimeError("no display")

    original = screeninfo.get_monitors
    screeninfo.get_monitors = no_display
    try:
        reloaded = importlib.reload(interaction)
        assert reloaded.monitor_window is None
        metrics = reloaded.ImageMetrics()
        assert metrics.window_width == 1920
        assert metrics.window_height == 1080
    finally:
        screeninfo.get_monitors = original
        importlib.reload(interaction)
