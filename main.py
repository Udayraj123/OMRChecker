"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import argparse
import sys
from pathlib import Path

from src.entry import entry_point
from src.logger import logger


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
        "-a",
        "--autoAlign",
        required=False,
        dest="autoAlign",
        action="store_true",
        help="(experimental) Enables automatic template alignment - \
        use if the scans show slight misalignments.",
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

    # NEW: Add CLI arguments for configuration files (Issue #201)
    argparser.add_argument(
        "--templateFile",
        required=False,
        dest="template_file",
        help="Path to template.json file",
    )

    argparser.add_argument(
        "--configFile",
        required=False,
        dest="config_file",
        help="Path to config.json file",
    )

    argparser.add_argument(
        "--evaluationFile",
        required=False,
        dest="evaluation_file",
        help="Path to evaluation.json file",
    )

    (
        args,
        unknown,
    ) = argparser.parse_known_args()

    args = vars(args)

    if len(unknown) > 0:
        logger.warning(f"\nError: Unknown arguments: {unknown}", unknown)
        argparser.print_help()
        exit(11)
    return args


def entry_point_for_args(args):
    if args["debug"] is True:
        # Disable tracebacks
        sys.tracebacklimit = 0

    # NEW: Log configuration file sources for Issue #201
    if args.get("template_file"):
        logger.info(f"Using custom template file: {args['template_file']}")
    if args.get("config_file"):
        logger.info(f"Using custom config file: {args['config_file']}")
    if args.get("evaluation_file"):
        logger.info(f"Using custom evaluation file: {args['evaluation_file']}")

    for root in args["input_paths"]:
        entry_point(
            Path(root),
            args,
        )


if __name__ == "__main__":
    args = parse_args()
    entry_point_for_args(args)