import os
from pathlib import Path

from src import constants, core
from src.config import CONFIG_DEFAULTS as config
from src.logger import logger
from src.processors.manager import ProcessorManager
from src.template import Template
from src.utils.imgutils import ImageUtils

# import cv2


class TemplateByBarcode:
    def update_template(local_template_path, path, args, curr_dir, root_dir):
        PROCESSOR_MANAGER = ProcessorManager()
        template = Template(local_template_path, PROCESSOR_MANAGER.processors)
        paths = constants.Paths(
            Path(
                os.path.join(args["output_dir"],"CheckedOMRs",path),
                root_dir.relative_to(root_dir),
            )
        )
        core.setup_dirs(paths)
        out = core.setup_output(paths, template)
        return template, out

    def make_folders(path, data):
        for file in os.listdir(path):
            if file == str(data):
                break
        else:
            path=os.path.join(path,data)
            os.mkdir(path)

    def TemplateBarcode(in_omr, template, out, file_name, args, curr_dir, root_dir):
        save_dir = out.paths.save_marked_dir
        for i, pre_processor in enumerate(template.pre_processors):
            if template.name[i] == "CropPage":
                in_omr = pre_processor.apply_filter(in_omr, args)
                in_omr = ImageUtils.resize_util(
                    in_omr,
                    config.dimensions.processing_width,
                    config.dimensions.processing_height,
                )

        for pre_processor in template.TemplateByBarcode:
            data, input_sorting = pre_processor.apply_filter(in_omr, args)
            path = data
            path_input = out.paths.output_dir
            path_1 = str(save_dir[:-1])
            TemplateByBarcode.make_folders(path_1, data[:-1])
            data_name = f"configs/{str(data[:-1])}.json"
            local_template_path = root_dir.joinpath(data_name)
            if os.path.exists(local_template_path):
                template, out = TemplateByBarcode.update_template(
                    local_template_path, path, args, curr_dir, root_dir
                )
            else:
                logger.error(f"Unable to find the path {local_template_path}")
                return None, None
            if input_sorting:
                data_2 = f"{data[:-1]}_inputs"
                TemplateByBarcode.make_folders(path_input, data_2)
                path_input = os.path.join(path_input,data_2,file_name)
                ImageUtils.save_img(path_input, in_omr)

        return template, out
