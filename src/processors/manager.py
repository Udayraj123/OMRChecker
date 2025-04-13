from dotmap import DotMap

from src.processors.AutoRotate import AutoRotate
from src.processors.Contrast import Contrast
from src.processors.CropOnMarkers import CropOnMarkers
from src.processors.CropPage import CropPage
from src.processors.FeatureBasedAlignment import FeatureBasedAlignment
from src.processors.GaussianBlur import GaussianBlur
from src.processors.Levels import Levels
from src.processors.MedianBlur import MedianBlur
from src.utils.constants import SUPPORTED_PROCESSOR_NAMES

# Note: we're now hard coding the processors mapping to support working export of PyInstaller
PROCESSOR_MANAGER = DotMap(
    {
        "processors": {
            "AutoRotate": AutoRotate,
            "Contrast": Contrast,
            "CropOnMarkers": CropOnMarkers,
            "CropPage": CropPage,
            "FeatureBasedAlignment": FeatureBasedAlignment,
            "GaussianBlur": GaussianBlur,
            "Levels": Levels,
            "MedianBlur": MedianBlur,
            # TODO: extract AlignOnMarkers preprocess from WarpOnPoints instead, or rename CropOnMarkers to something better with enableCropping support?
        }
    },
    _dynamic=False,
)

assert set(PROCESSOR_MANAGER.processors.keys()) == set(SUPPORTED_PROCESSOR_NAMES)
