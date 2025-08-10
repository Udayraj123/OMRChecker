from dataclasses import dataclass

import cv2
from matplotlib import pyplot as plt
from screeninfo import Monitor, get_monitors

from src.utils.constants import WAIT_KEYS
from src.utils.drawing import DrawingUtils
from src.utils.env import env
from src.utils.image import ImageUtils
from src.utils.logger import logger


@dataclass
class ImageMetrics:
    # TODO: fix window metrics doesn't account for the doc/taskbar on macos
    if env.OMR_CHECKER_CONTAINER:
        monitor_window = Monitor(0, 0, 1000, 1000, 100, 100, "FakeMonitor", False)  # NOQA: FBT003
    else:
        monitor_window = get_monitors()[0]
    window_width, window_height = monitor_window.width, monitor_window.height
    # for positioning image windows
    window_x, window_y = 0, 0

    def set_window_pos(self, pos) -> None:
        self.window_x, self.window_y = pos

    def update_for_shape(self, h, w) -> None:
        # TODO: get ppi for correct positioning?
        adjustment_ratio = 3
        h, w = h // adjustment_ratio, w // adjustment_ratio
        if self.window_x + w > self.window_width:
            self.window_x = 0
            if self.window_y + h > self.window_height:
                self.window_y = 0
            else:
                self.window_y += h
        else:
            self.window_x += w


class SelectROI:
    draw_color = (255, 0, 0)

    def __init__(self) -> None:
        # Our ROI, defined by two points
        self.rectangle_tl, self.rectangle_br = (0, 0), (0, 0)
        # True while ROI is actively being drawn by mouse
        self.drawing = False
        # True while ROI is drawn but is pending use or cancel
        self.show_drawing = False

    def on_mouse(self, event, x, y, _flags, _userdata) -> None:
        # logger.debug(event, x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click down (select first point)
            self.drawing = self.show_drawing = True
            self.rectangle_tl = self.rectangle_br = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            # Drag to second point
            if self.drawing:
                self.rectangle_br = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            # Left click up (select second point)
            self.drawing = False
            self.rectangle_br = x, y

    def run_selector_gui(self, name, image_to_show) -> None:
        # Make a copy for drawing ROIs
        image_copy = image_to_show.copy()
        cv2.imshow(name, image_copy)

        while True:
            if self.drawing:
                # Reset to source image
                image_copy = image_to_show.copy()
                tl = self.rectangle_tl

                # TODO: debug the GUI lag here -
                """
                tl, br = self.rectangle_tl, self.rectangle_br
                if tl != br:
                    tl, br = ImageUtils.clip_zone_to_image_bounds([tl, br], image_copy)
                    # Re-draw the rectangle
                    cv2.rectangle(image_copy, tl, br, SelectROI.draw_color, 2)
                """

                DrawingUtils.draw_symbol(image_copy, "x", tl, tl)
                DrawingUtils.draw_text(image_copy, str(tl), tl)

                cv2.imshow(name, image_copy)

            pressed = cv2.waitKey(1)
            if pressed in {WAIT_KEYS.ENTER, WAIT_KEYS.SPACE}:
                # Pressed Enter or Space to use ROI
                self.drawing = False
                self.show_drawing = False
                # here do something with ROI points values (rectangle_tl and rectangle_br)
            elif pressed in {ord("c"), ord("C"), WAIT_KEYS.ESCAPE}:
                # Pressed C or Esc to cancel ROI
                self.drawing = False
                self.show_drawing = False
            elif pressed in {ord("q"), ord("Q")}:
                # Pressed Q to exit
                break

    def show(self, name, image_to_show) -> None:
        # Enable status bar in the named window window
        cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)

        # Register the mouse callback
        cv2.setMouseCallback(name, self.on_mouse)

        # Run the GUI
        self.run_selector_gui(name, image_to_show)


class InteractionUtils:
    """Perform primary functions such as displaying images and reading responses."""

    image_metrics = ImageMetrics()

    select_roi = SelectROI()

    @staticmethod
    def show(
        # ruff: noqa: PLR0913
        name,
        image,
        pause=1,
        resize_to_width=False,
        resize_to_height=False,
        reset_pos=None,
        config=None,
    ) -> None:
        image_metrics = InteractionUtils.image_metrics
        if env.OMR_CHECKER_CONTAINER:
            return
        if image is None:
            logger.warning(f"'{name}' - NoneType image to show!")
            if pause:
                cv2.destroyAllWindows()
            return
        if config is not None:
            display_width, display_height = config.outputs.display_image_dimensions
            if resize_to_width:
                image_to_show = ImageUtils.resize_single(image, u_width=display_width)
            elif resize_to_height:
                image_to_show = ImageUtils.resize_single(image, u_height=display_height)
            else:
                image_to_show = image
        else:
            image_to_show = image

        if not is_window_available(name):
            cv2.namedWindow(name)

        # Show the image in the named window
        cv2.imshow(name, image_to_show)

        if reset_pos:
            image_metrics.set_window_pos(reset_pos)

        cv2.moveWindow(
            name,
            image_metrics.window_x,
            image_metrics.window_y,
        )

        h, w = image_to_show.shape[:2]

        # Set next window position
        margin = 25
        h += margin
        w += margin

        image_metrics.update_for_shape(h, w)

        if pause:
            logger.info(
                f"Showing '{name}'\n\t Press Q on image to continue. Press Ctrl + C in terminal to exit"
            )

            close_all_on_wait_key("q")
            InteractionUtils.image_metrics.window_x = 0
            InteractionUtils.image_metrics.window_y = 0

    @staticmethod
    def show_for_roi(name, image_to_show) -> None:
        InteractionUtils.select_roi.show(name, image_to_show)


class Stats:
    # TODO: Fill these for stats: multiMarkedFilesCount = 0,  errorFilesCount = 0
    files_moved = 0
    files_not_moved = 0


def close_all_on_wait_key(key="q") -> None:
    while cv2.waitKey(1) & 0xFF not in {ord(key), WAIT_KEYS.ESCAPE}:
        pass
    cv2.destroyAllWindows()
    # also close open plots!
    plt.close()


def is_window_available(name: str) -> bool:
    """Checks if a window is available."""
    try:
        cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
    except Exception as e:
        logger.debug(e)
        return False
    else:
        return True
