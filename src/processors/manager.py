from dotmap import DotMap

from src.processors.CropOnMarkers import CropOnMarkers
from src.processors.CropPage import CropPage
from src.processors.FeatureBasedAlignment import FeatureBasedAlignment
from src.processors.GaussianBlur import GaussianBlur
from src.processors.Levels import Levels
from src.processors.MedianBlur import MedianBlur
from src.processors.AutoAlign import AutoAlign
# Note: we're now hard coding the processors mapping to support working export of PyInstaller
PROCESSOR_MANAGER = DotMap(
    {
        "processors": {
            "CropOnMarkers": CropOnMarkers,
            # TODO: "WarpOnPoints": WarpOnPointsCommon,
            "CropPage": CropPage,
            "FeatureBasedAlignment": FeatureBasedAlignment,
            "GaussianBlur": GaussianBlur,
            "Levels": Levels,
            "MedianBlur": MedianBlur,
            "AutoAlign":AutoAlign
        }
    },
    _dynamic=False,
)
