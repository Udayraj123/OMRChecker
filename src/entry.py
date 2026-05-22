"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import os
import shutil
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time

import cv2
import pandas as pd
from rich.table import Table

from src.constants.common import (
    CONFIG_FILENAME,
    ERROR_CODES,
    EVALUATION_FILENAME,
    TEMPLATE_FILENAME,
)
from src.defaults import CONFIG_DEFAULTS
from src.evaluation import EvaluationConfig, evaluate_concatenated_response
from src.logger import console, logger
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.parsing import (
    OVERRIDE_MERGER,
    get_concatenated_response,
    open_config_with_defaults,
)
from src.utils.validations import validate_config_json

# Load processors
STATS = Stats()


def _build_tuning_config_inmemory(config_payload: dict):
    """Build a TuningConfig DotMap from a dict without touching the filesystem.

    This replicates ``open_config_with_defaults`` but accepts an already-loaded
    dict instead of a file path. Used by the in-memory web pipeline so that
    concurrent workers never race over a shared ``config.json`` file.

    Strategy: Option A — bypass disk entirely. Each worker builds its own
    DotMap from the merged payload passed in via the process-pool task payload.
    Workers are separate processes so ``_TEMPLATE_CACHE`` is naturally isolated,
    but building config in-memory also removes the shared-file race on
    ``<base_root>/config.json`` that caused intermittent ``JSONDecodeError``
    under load (all N workers were serialising distinct per-image payloads to
    the same path, then reading them back).
    """
    from copy import deepcopy

    from dotmap import DotMap

    merged = OVERRIDE_MERGER.merge(deepcopy(CONFIG_DEFAULTS), dict(config_payload or {}))
    # config_path is used only for a log message inside validate_config_json.
    validate_config_json(merged, "<in-memory>")
    return DotMap(merged, _dynamic=False)


def serialize_path(path):
    return Path(path).as_posix()


def entry_point(input_dir, args):
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    return process_dir(input_dir, curr_dir, args)


# ---------------------------------------------------------------------------
# In-memory single-image entry point.
#
# This is the high-throughput path used by the Web UI's worker pool when
# OMR_WEBUI_INMEMORY_PIPELINE=true (default). It bypasses the legacy
# "scan a directory" entry point entirely so that:
#   1. The source image is decoded ONCE into a numpy array (no staged copy
#      written to a per-image runtime directory).
#   2. Rotation, if any, is applied via cv2.rotate on the in-memory array
#      (no _rotated cache file written).
#   3. Template construction is cached per worker process, so the engine
#      doesn't re-parse template.json on every page.
#
# On a 5000-page Defender-enabled Windows box this removes ~10,000 file
# create + scan operations from the hot path while keeping the engine's
# output layout (CheckedOMRs / Manual / Results.csv) unchanged.
# ---------------------------------------------------------------------------

_TEMPLATE_CACHE: dict[str, tuple[float, "object"]] = {}


def _rotation_code_or_raise(rotation_degrees):
    """Return the OpenCV rotate code for supported right-angle rotations.

    Mirrors the legacy web runtime rotation path: unsupported non-zero values
    are programming/configuration errors and must not silently process the
    image unrotated.
    """
    try:
        degrees = int(rotation_degrees)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported rotation: {rotation_degrees}") from exc

    rotate_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotate_code = rotate_codes.get(degrees)
    if rotate_code is None:
        raise ValueError(f"Unsupported rotation: {rotation_degrees}")
    return rotate_code


def _build_or_fetch_template(template_path, tuning_config):
    """Cache ``Template`` instances per (path, mtime, processing_dims, pid).

    ``Template`` construction is non-trivial (it parses field blocks and
    instantiates preprocessors). For a 5000-page batch the same template
    is reused unchanged on every page, so caching saves real CPU.

    ``pid`` is included in the key as an explicit defensive guard even though
    workers ARE separate processes (and therefore each have their own module-level
    ``_TEMPLATE_CACHE`` dict). It makes the isolation contract self-documenting.
    """
    key = (
        str(template_path),
        os.path.getmtime(template_path),
        int(tuning_config.dimensions.processing_width),
        int(tuning_config.dimensions.processing_height),
        os.getpid(),
    )
    cached = _TEMPLATE_CACHE.get(repr(key))
    if cached is not None:
        return cached[1]
    template = Template(template_path, tuning_config)
    _TEMPLATE_CACHE[repr(key)] = (key, template)
    return template


def entry_point_for_image(
    *,
    image_path,
    output_dir,
    template_payload: dict,
    config_payload: dict,
    evaluation_path=None,
    template_dir=None,
    rotation_degrees: int = 0,
):
    """Process a single image in-memory, writing engine outputs to ``output_dir``.

    Parameters
    ----------
    image_path
        Filesystem path to the source image. Read **directly** by cv2 —
        not via any staged copy.
    output_dir
        Per-worker output directory. The engine writes ``Results/*.csv``,
        ``CheckedOMRs/<name>``, and ``Manual/{Errors,MultiMarked}*`` under
        this path (same layout as ``entry_point_for_args``).
    template_payload
        Raw template JSON object (kept for symmetry with the legacy
        directory-staged worker payload; the actual ``Template`` is
        instantiated from ``template_dir / template.json`` because the
        ``Template`` constructor needs a path to resolve asset references).
    config_payload
        Merged config dict (with dynamic ``processing_width`` /
        ``processing_height`` already injected by the caller). Consumed
        in-memory via ``_build_tuning_config_inmemory`` — no ``config.json``
        is written to disk, eliminating the shared-file race between workers.
    evaluation_path
        Optional Path to evaluation.json. If supplied, scores are computed.
    template_dir
        Directory that contains ``template.json`` and any template-relative
        asset files. Lives under the per-worker runtime base path; reused
        across every task this worker handles.
    rotation_degrees
        0, 90, 180, or 270 — applied to the decoded image in memory.
    """
    from pathlib import Path

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    template_dir = Path(template_dir) if template_dir is not None else None
    if template_dir is None:
        raise ValueError("entry_point_for_image requires a template_dir")
    template_path = template_dir / "template.json"
    if not template_path.exists():
        raise FileNotFoundError(
            f"template.json not found under {template_dir.as_posix()}; "
            "the per-worker base directory must be prepared before "
            "entry_point_for_image is called."
        )

    # Build TuningConfig in-memory — no disk write.
    #
    # The old approach wrote config_payload to template_dir/config.json and
    # then read it back. All N workers share the same template_dir (base_root),
    # so under load they raced over a single file with distinct per-image
    # payloads (processing_height/width differ per image). On Windows that
    # produced intermittent JSONDecodeError → silent preprocess_failures.
    #
    # Option A fix: _build_tuning_config_inmemory replicates the merge +
    # validate logic of open_config_with_defaults without touching the
    # filesystem, so each worker builds its own isolated DotMap.
    cfg = dict(config_payload or {})
    outputs = dict(cfg.get("outputs") or {})
    outputs["show_image_level"] = 0
    cfg["outputs"] = outputs
    tuning_config = _build_tuning_config_inmemory(cfg)
    template = _build_or_fetch_template(template_path, tuning_config)

    evaluation_config = None
    if (
        evaluation_path is not None
        and Path(evaluation_path).exists()
    ):
        evaluation_config = EvaluationConfig(
            template_dir,
            Path(evaluation_path),
            template,
            tuning_config,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = Paths(output_dir)
    setup_dirs_for_paths(paths)
    outputs_namespace = setup_outputs_for_template(paths, template)

    # Reset per-image stats. STATS is a module-level singleton and
    # ProcessPoolExecutor reuses each worker across many tasks, so without
    # this reset the files_moved / files_not_moved counters silently
    # accumulate across every image a worker processes — corrupting the
    # per-image metrics and the final "Sum Tallied!" check in print_stats.
    STATS.files_moved = 0
    STATS.files_not_moved = 0

    in_omr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if in_omr is not None and rotation_degrees:
        in_omr = cv2.rotate(in_omr, _rotation_code_or_raise(rotation_degrees))

    _process_single_omr_image(
        in_omr,
        image_path,
        image_path.name,
        template,
        tuning_config,
        evaluation_config,
        outputs_namespace,
        files_counter=1,
    )


def print_config_summary(
    curr_dir,
    omr_files,
    template,
    tuning_config,
    local_config_path,
    evaluation_config,
    args,
):
    logger.info("")
    table = Table(title="Current Configurations", show_header=False, show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Directory Path", f"{curr_dir}")
    table.add_row("Count of Images", f"{len(omr_files)}")
    table.add_row("Set Layout Mode ", "ON" if args["setLayout"] else "OFF")
    pre_processor_names = [pp.__class__.__name__ for pp in template.pre_processors]
    table.add_row(
        "Markers Detection",
        "ON" if "CropOnMarkers" in pre_processor_names else "OFF",
    )
    table.add_row("Auto Alignment", f"{tuning_config.alignment_params.auto_align}")
    table.add_row("Detected Template Path", f"{template}")
    if local_config_path:
        table.add_row("Detected Local Config", f"{local_config_path}")
    if evaluation_config:
        table.add_row("Detected Evaluation Config", f"{evaluation_config}")

    table.add_row(
        "Detected pre-processors",
        ", ".join(pre_processor_names),
    )
    console.print(table, justify="center")


def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=None,
):
    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(TEMPLATE_FILENAME)
    local_template_exists = os.path.exists(local_template_path)
    if local_template_exists:
        template = Template(
            local_template_path,
            tuning_config,
        )
    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    output_dir = Path(args["output_dir"], curr_dir.relative_to(root_dir))
    paths = Paths(output_dir)

    # look for images in current dir to process
    exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    # Exclude images (take union over all pre_processors)
    excluded_files = []
    if template:
        for pp in template.pre_processors:
            excluded_files.extend(Path(p) for p in pp.exclude_files())

    local_evaluation_path = curr_dir.joinpath(EVALUATION_FILENAME)
    if not args["setLayout"] and os.path.exists(local_evaluation_path):
        if not local_template_exists:
            logger.warning(
                f"Found an evaluation file without a parent template file: {local_evaluation_path}"
            )
        evaluation_config = EvaluationConfig(
            curr_dir,
            local_evaluation_path,
            template,
            tuning_config,
        )

        excluded_files.extend(
            Path(exclude_file) for exclude_file in evaluation_config.get_exclude_files()
        )

    omr_files = [f for f in omr_files if f not in excluded_files]

    if omr_files:
        if not template:
            logger.error(
                f"Found images, but no template in the directory tree \
                of '{curr_dir}'. \nPlace {TEMPLATE_FILENAME} in the \
                appropriate directory."
            )
            raise Exception(
                f"No template file found in the directory tree of {serialize_path(curr_dir)}"
            )

        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)

        print_config_summary(
            curr_dir,
            omr_files,
            template,
            tuning_config,
            local_config_path,
            evaluation_config,
            args,
        )
        if args["setLayout"]:
            show_template_layouts(omr_files, template, tuning_config)
        else:
            process_files(
                omr_files,
                template,
                tuning_config,
                evaluation_config,
                outputs_namespace,
            )

    elif not subdirs:
        # Each subdirectory should have images or should be non-leaf
        logger.info(
            f"No valid images or sub-folders found in {curr_dir}.\
            Empty directories not allowed."
        )

    # recursively process sub-folders
    for d in subdirs:
        process_dir(
            root_dir,
            d,
            args,
            template,
            tuning_config,
            evaluation_config,
        )


def show_template_layouts(omr_files, template, tuning_config):
    for file_path in omr_files:
        file_name = file_path.name
        file_path = str(file_path)
        in_omr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        template_layout = template.image_instance_ops.draw_template_layout(
            in_omr, template, shifted=False, border=2
        )
        InteractionUtils.show(
            f"Template Layout: {file_name}", template_layout, 1, 1, config=tuning_config
        )


def _process_single_omr_image(
    in_omr,
    file_path,
    file_name,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
    *,
    files_counter: int = 1,
):
    """Inner per-image OMR pipeline shared by ``process_files`` and the
    in-memory ``entry_point_for_image`` worker entry.

    ``in_omr`` must be either a ``numpy.ndarray`` (already loaded image)
    or ``None`` (signals a read failure upstream). Splitting this body out
    lets callers feed a pre-decoded array — avoiding the staged-PNG round
    trip that dominates Windows Defender scan cost on 5000+ page batches.
    """
    if in_omr is None:
        logger.error(
            f"({files_counter}) Could not read image: '{file_path}'"
            " — file is corrupt, empty, or not a valid image format"
        )
        new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
        outputs_namespace.OUTPUT_SET.append(
            [file_name] + outputs_namespace.empty_resp
        )
        if check_and_move(ERROR_CODES.NO_MARKER_ERR, file_path, new_file_path):
            err_line = [
                file_name,
                serialize_path(file_path),
                serialize_path(new_file_path),
                "NA",
            ] + outputs_namespace.empty_resp
            pd.DataFrame(err_line, dtype=str).T.to_csv(
                outputs_namespace.files_obj["Errors"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
        return

    logger.info("")
    logger.info(
        f"({files_counter}) Opening image: \t'{file_path}'\tResolution: {in_omr.shape}"
    )

    template.image_instance_ops.reset_all_save_img()

    template.image_instance_ops.append_save_img(1, in_omr)

    in_omr = template.image_instance_ops.apply_preprocessors(
        file_path, in_omr, template
    )

    if in_omr is None:
        new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
        outputs_namespace.OUTPUT_SET.append(
            [file_name] + outputs_namespace.empty_resp
        )
        if check_and_move(ERROR_CODES.NO_MARKER_ERR, file_path, new_file_path):
            err_line = [
                file_name,
                serialize_path(file_path),
                serialize_path(new_file_path),
                "NA",
            ] + outputs_namespace.empty_resp
            pd.DataFrame(err_line, dtype=str).T.to_csv(
                outputs_namespace.files_obj["Errors"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
        return

    file_id = str(file_name)
    save_dir = outputs_namespace.paths.save_marked_dir
    (
        response_dict,
        final_marked,
        multi_marked,
        _,
    ) = template.image_instance_ops.read_omr_response(
        template, image=in_omr, name=file_id, save_dir=save_dir
    )

    omr_response = get_concatenated_response(response_dict, template)

    if (
        evaluation_config is None
        or not evaluation_config.get_should_explain_scoring()
    ):
        logger.info(f"Read Response: \n{omr_response}")

    score = 0
    if evaluation_config is not None:
        score = evaluate_concatenated_response(
            omr_response,
            evaluation_config,
            file_path,
            outputs_namespace.paths.evaluation_dir,
        )
        logger.info(
            f"(/{files_counter}) Graded with score: {round(score, 2)}\t for file: '{file_id}'"
        )
    else:
        logger.info(f"(/{files_counter}) Processed file: '{file_id}'")

    if tuning_config.outputs.show_image_level >= 2:
        InteractionUtils.show(
            f"Final Marked Bubbles : '{file_id}'",
            ImageUtils.resize_util_h(
                final_marked, int(tuning_config.dimensions.display_height * 1.3)
            ),
            1,
            1,
            config=tuning_config,
        )

    resp_array = []
    for k in template.output_columns:
        resp_array.append(omr_response[k])

    outputs_namespace.OUTPUT_SET.append([file_name] + resp_array)

    if multi_marked == 0 or not tuning_config.outputs.filter_out_multimarked_files:
        STATS.files_not_moved += 1
        new_file_path = save_dir.joinpath(file_id)
        results_line = [
            file_name,
            serialize_path(file_path),
            serialize_path(new_file_path),
            score,
        ] + resp_array
        pd.DataFrame(results_line, dtype=str).T.to_csv(
            outputs_namespace.files_obj["Results"],
            mode="a",
            quoting=QUOTE_NONNUMERIC,
            header=False,
            index=False,
        )
    else:
        logger.info(f"[{files_counter}] Found multi-marked file: '{file_id}'")
        new_file_path = outputs_namespace.paths.multi_marked_dir.joinpath(file_name)
        if check_and_move(ERROR_CODES.MULTI_BUBBLE_WARN, file_path, new_file_path):
            mm_line = [
                file_name,
                serialize_path(file_path),
                serialize_path(new_file_path),
                "NA",
            ] + resp_array
            pd.DataFrame(mm_line, dtype=str).T.to_csv(
                outputs_namespace.files_obj["MultiMarked"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )


def process_files(
    omr_files,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
):
    start_time = int(time())
    files_counter = 0
    STATS.files_moved = 0
    STATS.files_not_moved = 0

    for file_path in omr_files:
        files_counter += 1
        file_name = file_path.name

        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        _process_single_omr_image(
            in_omr,
            file_path,
            file_name,
            template,
            tuning_config,
            evaluation_config,
            outputs_namespace,
            files_counter=files_counter,
        )
    print_stats(start_time, files_counter, tuning_config)


def check_and_move(error_code, file_path, filepath2):
    """Copy a review-needed source image into the engine's Manual folders.

    The web UI aggregates worker outputs from scratch directories, so the
    source image must remain in place. Copying gives operators a reviewable
    artifact without breaking later cleanup or retry flows.
    """
    try:
        source = Path(file_path)
        target = Path(filepath2)
        if not source.exists():
            STATS.files_not_moved += 1
            logger.warning(
                "Could not copy review file for %s: source does not exist: %s",
                error_code,
                source,
            )
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        STATS.files_moved += 1
        return True
    except Exception as exc:  # noqa: BLE001 - keep legacy CSV-writing path alive
        STATS.files_not_moved += 1
        logger.warning(
            "Could not copy review file for %s: %s: %s",
            error_code,
            type(exc).__name__,
            exc,
        )
        return True


def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
    log = logger.info
    log("")
    log(f"{'Total file(s) moved': <27}: {STATS.files_moved}")
    log(f"{'Total file(s) not moved': <27}: {STATS.files_not_moved}")
    log("--------------------------------")
    log(
        f"{'Total file(s) processed': <27}: {files_counter} ({'Sum Tallied!' if files_counter == (STATS.files_moved + STATS.files_not_moved) else 'Not Tallying!'})"
    )

    if tuning_config.outputs.show_image_level <= 0:
        log(
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking / 60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate': <27}: \t ~ {round(time_checking / files_counter, 2)} seconds/OMR"
        )
        log(
            f"{'OMR Processing Speed': <27}: \t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute"
        )
    else:
        log(f"\n{'Total script time': <27}: {time_checking} seconds")

    if tuning_config.outputs.show_image_level <= 1:
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )
