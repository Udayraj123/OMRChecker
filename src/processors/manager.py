"""
Processor/Extension framework
Adapated from https://github.com/gdiepen/python_processor_example
"""
import importlib
import inspect
import pkgutil
import sys

from src.logger import logger

# Explicit fallback list of processor modules. Used in frozen
# (PyInstaller) environments where ``pkgutil.walk_packages`` cannot
# enumerate modules because they are stored inside the PYZ archive
# instead of on disk. Update this list whenever a new file is added to
# ``src/processors``.
_FROZEN_PROCESSOR_MODULES = [
    "src.processors.CropPage",
    "src.processors.CropOnMarkers",
    "src.processors.FeatureBasedAlignment",
    "src.processors.builtins",
]


class Processor:
    """Base class that each processor must inherit from."""

    def __init__(
        self,
        options=None,
        relative_dir=None,
        image_instance_ops=None,
    ):
        self.options = options
        self.relative_dir = relative_dir
        self.image_instance_ops = image_instance_ops
        self.tuning_config = image_instance_ops.tuning_config
        self.description = "UNKNOWN"


class ProcessorManager:
    """Upon creation, this class will read the processors package for modules
    that contain a class definition that is inheriting from the Processor class
    """

    def __init__(self, processors_dir="src.processors"):
        """Constructor that initiates the reading of all available processors
        when an instance of the ProcessorCollection object is created
        """
        self.processors_dir = processors_dir
        self.reload_processors()

    @staticmethod
    def get_name_filter(processor_name):
        def filter_function(member):
            return inspect.isclass(member) and member.__module__ == processor_name

        return filter_function

    def reload_processors(self):
        """Reset the list of all processors and initiate the walk over the main
        provided processor package to load all available processors
        """
        self.processors = {}
        self.seen_paths = []

        logger.info(f'Loading processors from "{self.processors_dir}"...')
        self.walk_package(self.processors_dir)

    def walk_package(self, package):
        """walk the supplied package to retrieve all processors"""
        imported_package = __import__(package, fromlist=["blah"])
        loaded_packages = []

        # ``pkgutil.walk_packages`` works in normal Python installs but
        # returns nothing when this code runs from inside a PyInstaller
        # bundle (modules live in the PYZ archive, not on disk). Build
        # the module name list dynamically when possible, then merge in
        # the explicit ``_FROZEN_PROCESSOR_MODULES`` list so frozen apps
        # still load every processor.
        module_names: list[str] = []
        try:
            for _, processor_name, ispkg in pkgutil.walk_packages(
                imported_package.__path__, imported_package.__name__ + "."
            ):
                if not ispkg and processor_name != __name__:
                    module_names.append(processor_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"pkgutil.walk_packages failed ({exc!r}); using frozen list")

        if getattr(sys, "frozen", False) or not module_names:
            for name in _FROZEN_PROCESSOR_MODULES:
                if name == __name__ or name in module_names:
                    continue
                module_names.append(name)

        for processor_name in module_names:
            try:
                processor_module = importlib.import_module(processor_name)
            except ImportError as exc:
                logger.warning(f"Skipping processor {processor_name!r}: {exc}")
                continue
            # https://stackoverflow.com/a/46206754/6242649
            clsmembers = inspect.getmembers(
                processor_module,
                ProcessorManager.get_name_filter(processor_name),
            )
            for _, c in clsmembers:
                # Only add classes that are a sub class of Processor, but NOT Processor itself
                if issubclass(c, Processor) & (c is not Processor):
                    self.processors[c.__name__] = c
                    if c.__name__ not in loaded_packages:
                        loaded_packages.append(c.__name__)

        logger.info(f"Loaded processors: {loaded_packages}")


# Singleton export
PROCESSOR_MANAGER = ProcessorManager()
