from dataclasses import dataclass

import cv2
from screeninfo import get_monitors


from src.logger import logger
from src.utils.image import ImageUtils

monitor_window = get_monitors()[0]


@dataclass
class ImageMetrics:
    # TODO: Move TEXT_SIZE, etc here and find a better class name
    window_width, window_height = monitor_window.width, monitor_window.height
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
            img = ImageUtils.resize_util(origin, config.dimensions.display_width)
        else:
            img = origin

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

    @staticmethod
    def show_scrollable(name, origin, reset_pos=None):
        """
        Display an image in a scrollable Tkinter window.

        The image is displayed at its original resolution. If it is
        larger than the available screen area, horizontal and vertical
        scrollbars allow the complete image to be inspected.

        Zooming only changes the visualization but not the coordinates.
        """

        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "The scrollable layout viewer requires Tkinter. "
                "Please install the Tk package for your Python installation. "
            ) from exc

        if origin is None:
            logger.info(f"'{name}' - NoneType image to show!")
            return

        original_image = InteractionUtils._prepare_scrollable_image(
            origin, Image
        )

        root = tk.Tk()
        root.title(name)

        canvas, zoom = InteractionUtils._create_scrollable_canvas(
            root, original_image, tk
        )

        image_reference = {
            "photo": None,
            "item": None,
        }

        def update_image():
            """Resize and redraw the image according to the zoom value."""
            scale = zoom.get() / 100.0

            new_width = max(1, int(original_image.width * scale))
            new_height = max(1, int(original_image.height * scale))

            resized = original_image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

            photo = ImageTk.PhotoImage(resized)
            image_reference["photo"] = photo

            if image_reference["item"] is None:
                image_reference["item"] = canvas.create_image(
                    0,
                    0,
                    image=photo,
                    anchor=tk.NW,
                )
            else:
                canvas.itemconfigure(
                    image_reference["item"],
                    image=photo,
                )

            canvas.configure(
                scrollregion=(0, 0, new_width, new_height)
            )

            zoom_label.config(text=f"{int(zoom.get())}%")


        def set_zoom(value):
            value = max(25, min(200, value))
            zoom.set(value)
            update_image()

        def change_zoom(delta):
            set_zoom(zoom.get() + delta)

        zoom_label, coordinate_label = (
            InteractionUtils._create_scrollable_controls(
                root,
                zoom,
                update_image,
                set_zoom,
                change_zoom,
                tk,
            )
        )        

        InteractionUtils._setup_scrollable_mouse_controls(
            canvas,
            zoom,
            original_image,
            coordinate_label,
            set_zoom,
            change_zoom,
            tk,
        )

        InteractionUtils._setup_scrollable_keyboard_controls(
            root,
            set_zoom,
            change_zoom,
        )

        update_image()

        logger.info(
            f"Showing '{name}' at original resolution "
            f"{original_image.width}x{original_image.height}"
        )

        logger.info(
            "Zoom: +/- or slider, 0 = 100%, "
            "mouse wheel = scroll, Q/ESC = close"
        )

        root.mainloop()

    @staticmethod
    def _prepare_scrollable_image(origin, Image):
        """Convert an OpenCV image to a Pillow image."""
        # OpenCV uses BGR, Pillow expects RGB.
        if len(origin.shape) == 2:
            image_rgb = cv2.cvtColor(origin, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)

        return Image.fromarray(image_rgb)

    @staticmethod
    def _create_scrollable_canvas(root, original_image, tk):
        """Create the canvas, scrollbars and zoom variable."""
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Leave some room for the desktop/taskbar/window borders.
        max_width = int(screen_width * 0.90)
        max_height = int(screen_height * 0.85)

        window_width = min(original_image.width + 40, max_width)
        window_height = min(original_image.height + 80, max_height)

        root.geometry(f"{window_width}x{window_height}")

        zoom = tk.DoubleVar(value=100)

        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame)

        vertical_scrollbar = tk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=canvas.yview,
        )

        horizontal_scrollbar = tk.Scrollbar(
            frame,
            orient=tk.HORIZONTAL,
            command=canvas.xview,
        )

        canvas.configure(
            xscrollcommand=horizontal_scrollbar.set,
            yscrollcommand=vertical_scrollbar.set,
        )

        vertical_scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y,
        )

        horizontal_scrollbar.pack(
            side=tk.BOTTOM,
            fill=tk.X,
        )

        canvas.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        return canvas, zoom

    @staticmethod
    def _create_scrollable_controls(
        root,
        zoom,
        update_image,
        set_zoom,
        change_zoom,
        tk,
    ):
        """Create zoom controls and the mouse-coordinate display."""
        controls = tk.Frame(root)

        controls.pack(
            side=tk.BOTTOM,
            fill=tk.X,
            padx=5,
            pady=5,
        )

        tk.Button(
            controls,
            text="−",
            width=3,
            command=lambda: change_zoom(-10),
        ).pack(side=tk.LEFT)

        tk.Button(
            controls,
            text="100 %",
            command=lambda: set_zoom(100),
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="+",
            width=3,
            command=lambda: change_zoom(10),
        ).pack(side=tk.LEFT)

        zoom_label = tk.Label(
            controls,
            text="100%",
            width=6,
        )
        zoom_label.pack(side=tk.LEFT, padx=5)

        zoom_scale = tk.Scale(
            controls,
            from_=25,
            to=200,
            orient=tk.HORIZONTAL,
            variable=zoom,
            command=lambda value: update_image(),
            showvalue=False,
            length=250,
        )

        zoom_scale.pack(
            side=tk.LEFT,
            padx=10,
        )

        tk.Label(
            controls,
            text="Zoom",
        ).pack(side=tk.LEFT)

        coordinate_label = tk.Label(
            controls,
            text="X: --- Y: ---",
            width=20,
            anchor=tk.W,
        )

        coordinate_label.pack(
            side=tk.LEFT,
            padx=15,
        )

        return zoom_label, coordinate_label

    @staticmethod
    def _setup_scrollable_mouse_controls(
        canvas,
        zoom,
        original_image,
        coordinate_label,
        set_zoom,
        change_zoom,
        tk,
    ):
        """Configure mouse movement and cross-platform mouse-wheel controls."""

        def on_mouse_move(event):
            scale = zoom.get() / 100.0

            # Position within whole Canvas, also when scrolling.
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)

            # Calculate original coordinates.
            image_x = int(canvas_x / scale)
            image_y = int(canvas_y / scale)

            if (
                0 <= image_x < original_image.width
                and 0 <= image_y < original_image.height
            ):
                coordinate_label.config(
                    text=f"X: {image_x:4d} Y: {image_y:4d}"
                )
            else:
                coordinate_label.config(
                    text="X: --- Y: ---"
                )

        canvas.bind("<Motion>", on_mouse_move)

        def scroll_vertical(direction):
            canvas.yview_scroll(direction, "units")

        def scroll_horizontal(direction):
            canvas.xview_scroll(direction, "units")

        def on_mousewheel(event):
            """
            Windows / macOS:

            Wheel -> vertical scrolling
            Shift+Wheel -> horizontal scrolling
            """
            direction = -1 if event.delta > 0 else 1

            if event.state & 0x0001:
                scroll_horizontal(direction)
            else:
                scroll_vertical(direction)

        canvas.bind(
            "<MouseWheel>",
            on_mousewheel,
        )

        def on_linux_scroll_up(event):
            """
            Linux/X11:

            Button 4 = mouse wheel up
            """
            if event.state & 0x0001:
                scroll_horizontal(-1)
            else:
                scroll_vertical(-1)

        def on_linux_scroll_down(event):
            """
            Linux/X11:

            Button 5 = mouse wheel down
            """
            if event.state & 0x0001:
                scroll_horizontal(1)
            else:
                scroll_vertical(1)

        canvas.bind(
            "<Button-4>",
            on_linux_scroll_up,
        )

        canvas.bind(
            "<Button-5>",
            on_linux_scroll_down,
        )

    @staticmethod
    def _setup_scrollable_keyboard_controls(
        root,
        set_zoom,
        change_zoom,
    ):
        """Configure keyboard shortcuts for zooming and closing the window."""
        root.bind(
            "<plus>",
            lambda event: change_zoom(10),
        )

        root.bind(
            "<equal>",
            lambda event: change_zoom(10),
        )

        root.bind(
            "<minus>",
            lambda event: change_zoom(-10),
        )

        root.bind(
            "<Key-0>",
            lambda event: set_zoom(100),
        )

        root.bind(
            "<Escape>",
            lambda event: root.destroy(),
        )

        root.bind(
            "<Key-q>",
            lambda event: root.destroy(),
        )

        root.bind(
            "<Key-Q>",
            lambda event: root.destroy(),
        ) 

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
