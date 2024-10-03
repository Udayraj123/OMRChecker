from src.utils.logger import logger
from src.utils.math import MathUtils


class ShapeUtils:
    @staticmethod
    def compute_scan_zone_rectangle(zone_description, include_margins):
        x, y = zone_description["origin"]
        w, h = zone_description["dimensions"]
        if include_margins:
            margins = zone_description["margins"]
            x -= margins["left"]
            y -= margins["top"]
            w += margins["left"] + margins["right"]
            h += margins["top"] + margins["bottom"]
        return MathUtils.get_rectangle_points(x, y, w, h)

    @staticmethod
    def extract_image_from_zone_description(image, zone_description):
        # TODO: check bug in margins for scan zone
        zone_label = zone_description["label"]
        scan_zone_rectangle = ShapeUtils.compute_scan_zone_rectangle(
            zone_description, include_margins=True
        )
        print(
            "'zone_description",
            zone_description,
            "scan_zone_rectangle",
            scan_zone_rectangle,
        )
        return (
            ShapeUtils.extract_image_from_zone_rectangle(
                image, zone_label, scan_zone_rectangle
            ),
            scan_zone_rectangle,
        )

    @staticmethod
    def extract_image_from_zone_rectangle(image, zone_label, scan_zone_rectangle):
        # parse arguments
        h, w = image.shape[:2]
        # compute zone and clip to image dimensions
        zone_start = list(map(int, scan_zone_rectangle[0]))
        zone_end = list(map(int, scan_zone_rectangle[2]))

        if zone_start[0] < 0 or zone_start[1] < 0 or zone_end[0] > w or zone_end[1] > h:
            logger.warning(
                f"Clipping label {zone_label} with scan rectangle: {[zone_start, zone_end]} to image boundary {[w, h]}."
            )
            # zone_start, zone_end = ImageUtils.clip_zone_to_image_bounds([zone_start, zone_end], image)
            zone_start = [max(0, zone_start[0]), max(0, zone_start[1])]
            zone_end = [min(w, zone_end[0]), min(h, zone_end[1])]

        # Extract image zone
        return image[zone_start[1] : zone_end[1], zone_start[0] : zone_end[0]]
