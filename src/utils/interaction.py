from dataclasses import dataclass

import cv2
from screeninfo import get_monitors

from src.logger import logger
from src.utils.image import ImageUtils

# Fallback when screeninfo fails or returns unreasonable values (e.g. scaled/virtual monitors)
_DEFAULT_DISPLAY_WIDTH = 1920
_DEFAULT_DISPLAY_HEIGHT = 1080
_MAX_REASONABLE_WIDTH = 5120
_MAX_REASONABLE_HEIGHT = 2880


def _get_display_size():
    """Returns (width, height) for the primary monitor, with fallback if screeninfo fails or is wrong."""
    try:
        monitors = get_monitors()
        if not monitors:
            return _DEFAULT_DISPLAY_WIDTH, _DEFAULT_DISPLAY_HEIGHT
        m = monitors[0]
        w, h = int(getattr(m, "width", 0)), int(getattr(m, "height", 0))
        if w <= 0 or h <= 0 or w > _MAX_REASONABLE_WIDTH or h > _MAX_REASONABLE_HEIGHT:
            return _DEFAULT_DISPLAY_WIDTH, _DEFAULT_DISPLAY_HEIGHT
        return w, h
    except Exception:
        return _DEFAULT_DISPLAY_WIDTH, _DEFAULT_DISPLAY_HEIGHT


_display_width, _display_height = _get_display_size()

_DISPLAY_MARGIN = 100


def get_max_display_dimensions():
    """Returns (max_width, max_height) to use when resizing images for display (capped to screen)."""
    return (
        max(320, _display_width - _DISPLAY_MARGIN),
        max(320, _display_height - _DISPLAY_MARGIN),
    )


@dataclass
class ImageMetrics:
    # TODO: Move TEXT_SIZE, etc here and find a better class name
    window_width, window_height = _display_width, _display_height
    # for positioning image windows
    window_x, window_y = 0, 0
    reset_pos = [0, 0]


class InteractionUtils:
    """Perform primary functions such as displaying images and reading responses"""

    image_metrics = ImageMetrics()

    @staticmethod
    def show(name, origin, pause=1, resize=False, reset_pos=None, config=None):
        image_metrics = InteractionUtils.image_metrics
        if origin is None:
            logger.info(f"'{name}' - NoneType image to show!")
            if pause:
                cv2.destroyAllWindows()
            return
        if resize:
            if not config:
                raise Exception("config not provided for resizing the image to show")
            max_display_w, max_display_h = get_max_display_dimensions()
            max_width = min(config.dimensions.display_width, max_display_w)
            max_width = max(320, max_width)
            img = ImageUtils.resize_util(origin, max_width)
            h_img, w_img = img.shape[:2]
            if h_img > max_display_h:
                img = ImageUtils.resize_util_h(img, max_display_h)
        else:
            img = origin
            h_img, w_img = img.shape[:2]
            max_display_w, max_display_h = get_max_display_dimensions()
            if w_img > max_display_w or h_img > max_display_h:
                scale = min(max_display_w / w_img, max_display_h / h_img)
                if scale < 1:
                    new_w = max(320, int(w_img * scale))
                    img = ImageUtils.resize_util(origin, new_w)

        if not is_window_available(name):
            cv2.namedWindow(name)

        cv2.imshow(name, img)

        if reset_pos:
            image_metrics.window_x = reset_pos[0]
            image_metrics.window_y = reset_pos[1]

        cv2.moveWindow(
            name,
            image_metrics.window_x,
            image_metrics.window_y,
        )

        h, w = img.shape[:2]

        # Set next window position
        margin = 25
        w += margin
        h += margin

        w, h = w // 2, h // 2
        if image_metrics.window_x + w > image_metrics.window_width:
            image_metrics.window_x = 0
            if image_metrics.window_y + h > image_metrics.window_height:
                image_metrics.window_y = 0
            else:
                image_metrics.window_y += h
        else:
            image_metrics.window_x += w

        if pause:
            logger.info(
                f"Showing '{name}'\n\t Press Q on image to continue. Press Ctrl + C in terminal to exit"
            )

            wait_q()
            InteractionUtils.image_metrics.window_x = 0
            InteractionUtils.image_metrics.window_y = 0


@dataclass
class Stats:
    # TODO Fill these for stats
    # Move qbox_vals here?
    # badThresholds = []
    # veryBadPoints = []
    files_moved = 0
    files_not_moved = 0


def wait_q():
    esc_key = 27
    while cv2.waitKey(1) & 0xFF not in [ord("q"), esc_key]:
        pass
    cv2.destroyAllWindows()


def is_window_available(name: str) -> bool:
    """Checks if a window is available"""
    try:
        cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        return True
    except Exception as e:
        print(e)
        return False
