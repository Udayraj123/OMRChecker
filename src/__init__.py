# https://docs.python.org/3/tutorial/modules.html#:~:text=The%20__init__.py,on%20the%20module%20search%20path.

# Note: This import is added to the root __init__.py to adjust for perceived loading time
from pathlib import Path

from src.utils.env import env
from src.utils.logger import logger

# It takes a few seconds for the imports
logger.info("Loading OMRChecker modules...")

logger.info(f"VIRTUAL_ENV: {env.VIRTUAL_ENV}")

if not Path(env.VIRTUAL_ENV).exists():
    logger.warning("Your virtual Environment doesn't exist at the path!")
