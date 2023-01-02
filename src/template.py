"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import numpy as np

from src.constants import QTYPE_DATA
from src.logger import logger
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
        # dimensions = (width, height)
        self.dimensions = tuple(round(x) for x in dimensions)
        self.key = key
        self.orig = orig
        self.traverse_pts = traverse_pts
        self.empty_val = empty_val
        # will be set when using
        self.shift = 0


class Template:
    def __init__(self, template_path, tuning_config, extensions):
        json_obj = open_template_with_defaults(template_path)
        self.path = template_path
        self.q_blocks = []
        # TODO: ajv validation - throw exception on key not exist
        # TODO: extend DotMap here and only access keys that need extra parsing
        self.dimensions = json_obj["dimensions"]
        self.global_empty_val = json_obj["emptyVal"]
        self.bubble_dimensions = json_obj["bubbleDimensions"]
        self.concatenations = json_obj["concatenations"]
        self.singles = json_obj["singles"]

        # Add new qTypes from template
        if "qTypes" in json_obj:
            QTYPE_DATA.update(json_obj["qTypes"])

        # load image pre_processors
        self.pre_processors = [
            extensions[pre_processor["name"]](
                options=pre_processor["options"],
                relative_dir=template_path.parent,
                tuning_config=tuning_config,
            )
            for pre_processor in json_obj.get("preProcessors", [])
        ]

        # Add options
        self.options = json_obj.get("options", {})

        # Add q_blocks
        for name, block in json_obj["qBlocks"].items():
            self.add_q_blocks(name, block)

    # Expects bubble_dimensions to be set already
    def add_q_blocks(self, key, rect):
        assert self.bubble_dimensions != [-1, -1]
        # For q_type defined in q_blocks
        if "qType" in rect:
            rect.update(**QTYPE_DATA[rect["qType"]])
        else:
            rect.update(**{"vals": rect["vals"], "orient": rect["orient"]})

        # keyword arg unpacking followed by named args
        self.q_blocks += gen_grid(
            self.bubble_dimensions, self.global_empty_val, key, rect
        )
        # self.q_blocks.append(QBlock(rect.orig, calcQBlockDims(rect), maketemplate(rect)))

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
    """
    Input:
    orig - start point
    q_nos  - a tuple of q_nos
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - a 1D array of values of each alternative for a question

    Output:
    // Returns set of coordinates of a rectangular grid of points
    Returns a QBlock containing array of Qs and some metadata?!

    Ref:
        1 2 3 4
        1 2 3 4
        1 2 3 4

        (q1, q2, q3)

        00
        11
        22
        33
        44

        (q1.1,q1.2)

    """
    _h, _v = (0, 1) if (orient == "H") else (1, 0)
    # orig[0] += np.random.randint(-6,6)*2 # test random shift
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
            # TODO- make a mini object for this
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
    # Pass first three args as is. only append 'traverse_pts'
    return QBlock(q_block_dims, key, orig, traverse_pts, empty_val)


def gen_grid(bubble_dimensions, global_empty_val, key, rectParams):
    """
        Input(Directly passable from JSON parameters):
        bubble_dimensions - dimesions of single QBox
        orig- start point
        q_nos - an array of q_nos tuples(see below) that align with dimension
               of the big grid (gridDims extracted from here)
        big_gaps - (bigGapX,bigGapY) are the gaps between blocks
        gaps - (gapX,gapY) are the gaps between rows and cols in a block
        vals - a 1D array of values of each alternative for a question
        orient - The way of arranging the vals (vertical or horizontal)

        Output:
        // Returns an array of Q objects (having their points) arranged in a rectangular grid
        Returns grid of QBlock objects

                                    00    00    00    00
       Q1   1 2 3 4    1 2 3 4      11    11    11    11
       Q2   1 2 3 4    1 2 3 4      22    22    22    22         1234567
       Q3   1 2 3 4    1 2 3 4      33    33    33    33         1234567
                                    44    44    44    44
                                ,   55    55    55    55    ,    1234567
       Q7   1 2 3 4    1 2 3 4      66    66    66    66         1234567
       Q8   1 2 3 4    1 2 3 4      77    77    77    77
       Q9   1 2 3 4    1 2 3 4      88    88    88    88
                                    99    99    99    99

    TODO: Update this part, add more examples like-
        Q1  1 2 3 4

        Q2  1 2 3 4
        Q3  1 2 3 4

        Q4  1 2 3 4
        Q5  1 2 3 4

        MCQ type (orient="H")-
            [
                [(q1,q2,q3),(q4,q5,q6)]
                [(q7,q8,q9),(q10,q11,q12)]
            ]

        INT type (orient="V")-
            [
                [(q1d1,q1d2),(q2d1,q2d2),(q3d1,q3d2),(q4d1,q4d2)]
            ]

        ROLL type-
            [
                [(roll1,roll2,roll3,...,roll10)]
            ]

    """
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
            "col_orient",  # todo: consume this
            "emptyVal",
        ],
    )

    grid_data = np.array(q_nos)
    # print(grid_data.shape, grid_data)
    if (
        0 and len(grid_data.shape) != 3 or grid_data.size == 0
    ):  # product of shape is zero
        logger.error(
            "Error(gen_grid): Invalid q_nos array given:", grid_data.shape, grid_data
        )
        exit(32)

    orig = np.array(orig)

    num_qs_max = max([max([len(qb) for qb in row]) for row in grid_data])

    num_dims = [num_qs_max, len(vals)]

    q_blocks = []

    # **Simple is powerful**
    # _h and _v are named with respect to orient == "H", reverse their meaning
    # when orient = "V"
    _h, _v = (0, 1) if (orient == "H") else (1, 0)

    # print(orig, num_dims, grid_data.shape, grid_data)
    # orient is also the direction of making q_blocks

    # print(key, num_dims, orig, gaps, big_gaps, orig_gap )
    q_start = orig.copy()

    orig_gap = [0, 0]

    # Usually single row
    for row in grid_data:
        q_start[_v] = orig[_v]

        # Usually multiple qTuples
        for q_tuple in row:
            # Update num_dims and origGaps
            num_dims[0] = len(q_tuple)
            # big_gaps is indep of orientation
            orig_gap[0] = big_gaps[0] + (num_dims[_v] - 1) * gaps[_h]
            orig_gap[1] = big_gaps[1] + (num_dims[_h] - 1) * gaps[_v]
            # each q_tuple will have q_nos
            q_block_dims = [
                # width x height in pixels
                gaps[0] * (num_dims[_v] - 1) + bubble_dimensions[_h],
                gaps[1] * (num_dims[_h] - 1) + bubble_dimensions[_v],
            ]
            # WATCH FOR BLUNDER(use .copy()) - q_start was getting passed by
            # reference! (others args read-only)
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
