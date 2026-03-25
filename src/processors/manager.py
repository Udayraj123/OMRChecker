from src.utils.exceptions import ConfigError
from src.processors.image.AutoRotate import AutoRotate
from src.processors.image.Contrast import Contrast
from src.processors.image.CropOnMarkers import CropOnMarkers
from src.processors.image.CropPage import CropPage
from src.processors.image.FeatureBasedAlignment import FeatureBasedAlignment
from src.processors.image.GaussianBlur import GaussianBlur
from src.processors.image.Levels import Levels
from src.processors.image.MedianBlur import MedianBlur
from src.utils.constants import SUPPORTED_PROCESSOR_NAMES

# Note: we're now hard coding the processors mapping to support working export of PyInstaller
PROCESSOR_MANAGER: dict[str, type] = {
    "AutoRotate": AutoRotate,
    "Contrast": Contrast,
    "CropOnMarkers": CropOnMarkers,
    "CropPage": CropPage,
    "FeatureBasedAlignment": FeatureBasedAlignment,
    "GaussianBlur": GaussianBlur,
    "Levels": Levels,
    "MedianBlur": MedianBlur,
    # TODO: extract AlignOnMarkers preprocess from WarpOnPoints instead, or rename CropOnMarkers to something better with croppingEnabled support?
}

if set(PROCESSOR_MANAGER.keys()) != set(SUPPORTED_PROCESSOR_NAMES):
    msg = f"Processor keys mismatch: {set(PROCESSOR_MANAGER.keys())} != {set(SUPPORTED_PROCESSOR_NAMES)}"
    raise ConfigError(
        msg,
        context={
            "registered": list(PROCESSOR_MANAGER.keys()),
            "supported": SUPPORTED_PROCESSOR_NAMES,
        },
    )
