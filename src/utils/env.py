import os


class Env:
    # Container for all environment variables used in OMRChecker
    CI = os.environ.get("CI")
    OMR_CHECKER_CONTAINER = os.environ.get("OMR_CHECKER_CONTAINER", CI)
    VIRTUAL_ENV = os.environ.get("VIRTUAL_ENV", "Not Present")


env = Env()
