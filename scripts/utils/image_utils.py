import cv2
import os

from PIL import Image


def get_size_in_kb(path):
    return os.path.getsize(path) / 1000


def get_size_reduction(old_size, new_size):
    percent = 100 * (new_size - old_size) / old_size
    return f"({percent:.2f}%)"


def convert_image(image_path):
    with Image.open(image_path) as image:
        # Note: using hardcoded -4 as we're expected to receive .png or .PNG files only
        new_image_path = f"{image_path[:-4]}.jpg"
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image.save(new_image_path, "JPEG", quality=90, optimize=True)

        return new_image_path


def get_capped_resize_dimensions(image_width, image_height, u_width, u_height):
    if u_height is None:
        if u_width is None:
            raise Exception(f"Both u_width and u_height unavailable")

        u_height = int(image_height * u_width / image_width)

    # Note: u_width takes priority over u_height in our utility (thus different from max_width semantics)
    if u_width is None:
        u_width = int(image_width * u_height / image_height)

    return u_width, u_height


def resize_util(image, u_width=None, u_height=None):
    w, h = image.size[:2]
    # Resize by width
    resized_width, resized_height = get_capped_resize_dimensions(
        w, h, u_width=u_width, u_height=None
    )

    if resized_height >= u_height:
        # Resize by height
        resized_width, resized_height = get_capped_resize_dimensions(
            w, h, u_width=None, u_height=u_height
        )

    if resized_height == h and resized_width == w:
        # No need to resize
        return image

    return image.resize((int(resized_width), int(resized_height)), Image.LANCZOS)


def resize_image_and_save(image_path, max_width, max_height):
    without_extension, extension = os.path.splitext(image_path)
    temp_image_path = f"{without_extension}-tmp{extension}"
    with Image.open(image_path) as image:
        old_image_size = image.size[:2]
        w, h = old_image_size
        resized = False

        if h > max_height or w > max_width:
            image = resize_util(image, u_width=max_width, u_height=max_height)
            resized = True
            image.save(temp_image_path)

        return resized, temp_image_path, old_image_size, image.size


def load_image(file_path):
    # Wrapper to load images from available extensions
    # Check for corrupt files
    return cv2.imread(file_path)
