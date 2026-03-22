import cv2

from src.constants.common import ERROR_CODES
from src.evaluation import evaluate_concatenated_response
from src.utils.parsing import get_concatenated_response


def process_single_file_worker(
    file_path, template, tuning_config, evaluation_config, save_dir, evaluation_dir
):
    """
    Worker function for processing a single OMR file in parallel.
    """
    file_name = file_path.name
    file_id = str(file_name)

    in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    resolution = in_omr.shape

    template.image_instance_ops.reset_all_save_img()
    template.image_instance_ops.append_save_img(1, in_omr)

    in_omr = template.image_instance_ops.apply_preprocessors(
        file_path, in_omr, template
    )

    if in_omr is None:
        return {
            "status": "error",
            "file_name": file_name,
            "file_path": file_path,
            "error_code": ERROR_CODES.NO_MARKER_ERR,
        }

    (
        response_dict,
        final_marked,
        multi_marked,
        _,
    ) = template.image_instance_ops.read_omr_response(
        template, image=in_omr, name=file_id, save_dir=save_dir
    )

    omr_response = get_concatenated_response(response_dict, template)

    score = 0
    if evaluation_config is not None:
        score = evaluate_concatenated_response(
            omr_response,
            evaluation_config,
            file_path,
            evaluation_dir,
        )

    resp_array = []
    for k in template.output_columns:
        resp_array.append(omr_response[k])

    # Return final_marked only if it's needed for visualization to save IPC overhead
    return_final_marked = None
    if tuning_config.outputs.show_image_level >= 2:
        return_final_marked = final_marked

    return {
        "status": "success",
        "file_name": file_name,
        "file_path": file_path,
        "file_id": file_id,
        "score": score,
        "resp_array": resp_array,
        "multi_marked": multi_marked,
        "omr_response": omr_response,
        "final_marked": return_final_marked,
        "resolution": resolution,
    }
