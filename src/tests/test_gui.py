import os
import tkinter as tk

import numpy as np

from main import entry_point_for_args
from src.utils.interaction import InteractionUtils


def _find_widget(root, widget_type, text=None):
    """Find the first widget of the requested type."""
    for child in root.winfo_children():
        if isinstance(child, widget_type):
            if text is None or child.cget("text") == text:
                return child

        result = _find_widget(child, widget_type, text)
        if result is not None:
            return result

    return None


def test_show_scrollable_gui_is_interactive(monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    original_mainloop = tk.Tk.mainloop
    interaction_completed = False

    def test_mainloop(root):
        nonlocal interaction_completed

        def interact_with_gui():
            nonlocal interaction_completed

            plus_button = _find_widget(root, tk.Button, "+")
            assert plus_button is not None

            plus_button.invoke()

            zoom_label = _find_widget(root, tk.Label, "110%")
            interaction_completed = zoom_label is not None

            root.destroy()

        root.after(100, interact_with_gui)
        root.after(5000, root.destroy)

        original_mainloop(root)

    monkeypatch.setattr(tk.Tk, "mainloop", test_mainloop)

    InteractionUtils.show_scrollable(
        "GUI Test",
        image,
    )

    assert interaction_completed



