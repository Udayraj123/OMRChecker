"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import argparse
import sys
from pathlib import Path

from src.entry import entry_point
from src.utils.logger import logger


def parse_args():
    # construct the argument parse and parse the arguments
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "-i",
        "--inputDir",
        default=["inputs"],
        # https://docs.python.org/3/library/argparse.html#nargs
        nargs="*",
        required=False,
        type=str,
        dest="input_paths",
        help="Specify an input directory.",
    )

    argparser.add_argument(
        "-d",
        "--debug",
        required=False,
        dest="debug",
        action="store_false",
        help="Enables debugging mode for showing detailed errors",
    )

    argparser.add_argument(
        "-o",
        "--outputDir",
        default="outputs",
        required=False,
        dest="output_dir",
        help="Specify an output directory.",
    )

    argparser.add_argument(
        "-m",
        "--outputMode",
        default="default",
        required=False,
        choices=["moderation", "default"],
        dest="output_mode",
        help="Specify the output mode. Supported: moderation, default",
    )

    argparser.add_argument(
        "-l",
        "--setLayout",
        required=False,
        dest="setLayout",
        action="store_true",
        help="Set up OMR template layout - modify your json file and \
        run again until the template is set.",
    )

    (
        args,
        unknown,
    ) = argparser.parse_known_args()

    args = vars(args)

    if len(unknown) > 0:
        argparser.print_help()
        raise Exception(f"\nError: Unknown arguments: {unknown}")
    return args


def entry_point_for_args(args):
    if args["debug"] is True:
        # Disable traceback limit
        sys.tracebacklimit = 0
        # TODO: set log levels
    for root in args["input_paths"]:
        try:
            entry_point(
                Path(root),
                args,
            )
        except Exception:
            if args["debug"] is True:
                logger.critical(
                    f"OMRChecker crashed. add --debug and run again to see error details"
                )
            raise


if __name__ == "__main__":
    args = parse_args()
    entry_point_for_args(args)
