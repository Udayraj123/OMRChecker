"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import numpy as np

from src.constants import QTYPE_DATA
from src.utils.parsing import OVERRIDE_MERGER, open_template_with_defaults


# Coordinates Part
class Pt:
    """
    Container for a Point Box on the OMR

    q_no is the point's property- question to which this point belongs to
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """

    def __init__(self, pt, q_no, q_type, val):
        self.x = round(pt[0])
        self.y = round(pt[1])
        self.q_no = q_no
        self.q_type = q_type
        self.val = val


class QBlock:
    def __init__(self, dimensions, key, orig, traverse_pts, empty_val):
        self.dimensions = tuple(round(x) for x in dimensions)
        self.key = key
        self.orig = orig
        self.traverse_pts = traverse_pts
        self.empty_val = empty_val
        self.shift = 0


class Template:
    def __init__(self, template_path, image_instance_ops, extensions):
        json_obj = open_template_with_defaults(template_path)
        self.q_blocks, self.path = [], template_path
        (
            self.dimensions,
            self.global_empty_val,
            self.bubble_dimensions,
            self.concatenations,
            self.singles,
        ) = map(
            json_obj.get,
            ["dimensions", "emptyVal", "bubbleDimensions", "concatenations", "singles"],
        )

        # Add new qTypes from template
        if "qTypes" in json_obj:
            QTYPE_DATA.update(json_obj["qTypes"])

        # load image pre_processors
        self.pre_processors = [
            extensions[pre_processor["name"]](
                options=pre_processor["options"],
                relative_dir=template_path.parent,
                image_instance_ops=image_instance_ops,
            )
            for pre_processor in json_obj.get("preProcessors", [])
        ]

        # Add options
        self.options = json_obj.get("options", {})

        # Add q_blocks
        for name, block in json_obj["qBlocks"].items():
            self.add_q_blocks(name, block)

        # TODO: also validate these
        # - concatenations and singles together should be mutually exclusive
        # - All qNos in template are unique
        # - template bubbles don't overflow the image (already in instance)

    # Expects bubble_dimensions to be set already
    def add_q_blocks(self, key, rect):
        assert self.bubble_dimensions != [-1, -1]
        # For q_type defined in q_blocks
        if "qType" in rect:
            rect.update(**QTYPE_DATA[rect["qType"]])
        else:
            rect.update(**{"vals": rect["vals"], "orient": rect["orient"]})

        self.q_blocks += gen_grid(
            self.bubble_dimensions, self.global_empty_val, key, rect
        )

    def __str__(self):
        return str(self.path)


def gen_q_block(
    bubble_dimensions,
    q_block_dims,
    key,
    orig,
    q_nos,
    gaps,
    vals,
    q_type,
    orient,
    col_orient,
    empty_val,
):
    _h, _v = (0, 1) if (orient == "H") else (1, 0)
    traverse_pts = []
    o = [float(i) for i in orig]

    if col_orient == orient:
        for (q, _) in enumerate(q_nos):
            pt = o.copy()
            pts = []
            for (v, _) in enumerate(vals):
                pts.append(Pt(pt.copy(), q_nos[q], q_type, vals[v]))
                pt[_h] += gaps[_h]
            # For diagonal endpoint of QBlock
            pt[_h] += bubble_dimensions[_h] - gaps[_h]
            pt[_v] += bubble_dimensions[_v]
            traverse_pts.append(([o.copy(), pt.copy()], pts))
            o[_v] += gaps[_v]
    else:
        for (v, _) in enumerate(vals):
            pt = o.copy()
            pts = []
            for (q, _) in enumerate(q_nos):
                pts.append(Pt(pt.copy(), q_nos[q], q_type, vals[v]))
                pt[_v] += gaps[_v]
            # For diagonal endpoint of QBlock
            pt[_v] += bubble_dimensions[_v] - gaps[_v]
            pt[_h] += bubble_dimensions[_h]
            traverse_pts.append(([o.copy(), pt.copy()], pts))
            o[_h] += gaps[_h]

    return QBlock(q_block_dims, key, orig, traverse_pts, empty_val)


def gen_grid(bubble_dimensions, global_empty_val, key, rectParams):
    rect = OVERRIDE_MERGER.merge(
        {"orient": "V", "col_orient": "V", "emptyVal": global_empty_val}, rectParams
    )
    # case mapping
    (q_type, orig, big_gaps, gaps, q_nos, vals, orient, col_orient, empty_val) = map(
        rect.get,
        [
            "qType",
            "orig",
            "bigGaps",
            "gaps",
            "qNos",
            "vals",
            "orient",
            "col_orient",
            "emptyVal",
        ],
    )

    q_blocks, orig_gap, grid_data, orig = [], [0, 0], np.array(q_nos), np.array(orig)

    num_qs_max = max([max([len(qb) for qb in row]) for row in grid_data])

    num_dims = [num_qs_max, len(vals)]

    _h, _v = (0, 1) if (orient == "H") else (1, 0)

    q_start = orig.copy()
    # Usually single row
    for row in grid_data:
        q_start[_v] = orig[_v]

        # Usually multiple qTuples
        for q_tuple in row:
            # Update num_dims and origGaps
            num_dims[0] = len(q_tuple)
            # big_gaps is independent of orientation
            orig_gap[0] = big_gaps[0] + (num_dims[_v] - 1) * gaps[_h]
            orig_gap[1] = big_gaps[1] + (num_dims[_h] - 1) * gaps[_v]
            # each q_tuple will have q_nos
            q_block_dims = [
                # width x height in pixels
                gaps[0] * (num_dims[_v] - 1) + bubble_dimensions[_h],
                gaps[1] * (num_dims[_h] - 1) + bubble_dimensions[_v],
            ]

            q_blocks.append(
                gen_q_block(
                    bubble_dimensions,
                    q_block_dims,
                    key,
                    q_start.copy(),
                    q_tuple,
                    gaps,
                    vals,
                    q_type,
                    orient,
                    col_orient,
                    empty_val,
                )
            )
            # Goes vertically down first
            q_start[_v] += orig_gap[_v]
        q_start[_h] += orig_gap[_h]
    return q_blocks
