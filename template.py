"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""
import cv2
import json
import numpy as np
from globals import *

### Coordinates Part ###
class Pt():
    """
    Container for a Point Box on the OMR
    """
    """
    qNo is the question's property-
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """
    def __init__(self, pt, qNo, qType, val):
        self.x=pt[0]
        self.y=pt[1]
        self.qNo=qNo
        self.qType=qType
        self.val=val

class QBlock():
    def __init__(self, dims, key, orig, traverse_pts):
        # dims = (width, height)
        self.dims = dims
        self.key = key
        self.orig = orig
        self.traverse_pts = traverse_pts
        # will be set when using
        self.shift = 0

qtype_data = {
    'QTYPE_MED':{
        'vals' : ['E','H'],
        'orient':'V'
    },
    'QTYPE_ROLL':{
        'vals':range(10),
        'orient':'V'
    },
    'QTYPE_INT':{
        'vals':range(10),
        'orient':'V'
    },
    'QTYPE_MCQ4':{
        'vals' : ['A','B','C','D'],
        'orient':'H'
    },
    'QTYPE_MCQ5':{
        'vals' : ['A','B','C','D','E'],
        'orient':'H'
    },
    # Add custom question types here-
}

class Template():
    def __init__(self, jsonObj):
        self.QBlocks = []
        # throw exception on key not exist
        self.dims = jsonObj["Dimensions"]
        self.bubbleDims = jsonObj["BubbleDimensions"]
        self.concats = jsonObj["Concatenations"]
        self.singles = jsonObj["Singles"]

    # Expects bubbleDims to be set already
    def addQBlocks(self, key, rect):
        assert(self.bubbleDims != [-1, -1])
        # keyword arg unpacking followed by named args
        self.QBlocks += genGrid(self.bubbleDims, key, **rect,**qtype_data[rect['qType']])
        # self.QBlocks.append(QBlock(rect.orig, calcQBlockDims(rect), maketemplate(rect)))

def genQBlock(bubbleDims, QBlockDims, key, orig, qNos, gaps, vals, qType, orient):
    """
    Input:
    orig - start point
    qNo  - a qNos tuple
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
    H, V = (0,1) if(orient=='H') else (1,0)
    # orig[0] += np.random.randint(-6,6)*2 # test random shift
    Qs=[]
    traverse_pts = []
    o = orig.copy()
    for q in range(len(qNos)):
        pt = o.copy()
        pts = []
        for v in range(len(vals)):
            pts.append(Pt(pt.copy(),qNos[q],qType,vals[v]))
            pt[H] += gaps[H]
        pt[H] += bubbleDims[H] - gaps[H]
        pt[V] += bubbleDims[V]
        #TODO- make a mini object for this
        traverse_pts.append(([o.copy(), pt.copy()], pts))
        o[V] += gaps[V]

    # Pass first three args as is. only append 'traverse_pts'
    return QBlock(QBlockDims, key, orig, traverse_pts)

def genGrid(bubbleDims, key, qType, orig, bigGaps, gaps, qNos, vals, orient='V'):
    """
    Input(Directly passable from JSON parameters):
    bubbleDims - dimesions of single QBox
    orig- start point
    qNos - an array of qNos tuples(see below) that align with dimension of the big grid (gridDims extracted from here)
    bigGaps - (bigGapX,bigGapY) are the gaps between blocks
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
                            ,   55    55    55    55    ,    1234567                       and many more possibilities!
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

    MCQ type (orient='H')-
        [
            [(q1,q2,q3),(q4,q5,q6)]
            [(q7,q8,q9),(q10,q11,q12)]
        ]

    INT type (orient='V')-
        [
            [(q1d1,q1d2),(q2d1,q2d2),(q3d1,q3d2),(q4d1,q4d2)]
        ]

    ROLL type-
        [
            [(roll1,roll2,roll3,...,roll10)]
        ]

    """
    gridData = np.array(qNos)
    # print(gridData.shape, gridData)
    if(0 and len(gridData.shape)!=3 or gridData.size==0): # product of shape is zero
        print("Error(genGrid): Invalid qNos array given:", gridData.shape, gridData)
        exit(0)
        return []

    # ^4ENDUSER should also validate no overlap of rect points somehow?!

    """
    orient = 'H'
    numVals = 4
    [
    [["q1", "q2", "q3", "q4"], ["q5", "q6", "q7", "q8"]],
    [["q9", "q10", "q11", "q12"], ["q13", "q14", "q15", "q16"]]
    ]

    q1          q9
    q2          q10
    q3          q11
    q4          q12

    q5          q13
    q6          q14
    q7          q15
    q8          q16
    """

    orig = np.array(orig)
    numQsMax = max([max([len(qb) for qb in row]) for row in gridData])
    
    numDims = [numQsMax, len(vals)]

    QBlocks=[]
    
    # **Simple is powerful**
    # H and V are named with respect to orient == 'H', reverse their meaning when orient = 'V'
    H, V = (0,1) if(orient=='H') else (1,0)

    # print(orig, numDims, gridData.shape, gridData)
    # orient is also the direction of making QBlocks
    
    # print(key, numDims, orig, gaps, bigGaps, origGap )
    qStart = orig.copy()

    origGap = [0, 0]
    
    # Usually single row
    for row in gridData:
        qStart[V] = orig[V]
        
        # Usually multiple qTuples
        for qTuple in row:
            # Update numDims and origGaps
            numDims[0] = len(qTuple)
            # bigGaps is indep of orientation
            origGap[0] = bigGaps[0] + (numDims[V]-1)*gaps[H]
            origGap[1] = bigGaps[1] + (numDims[H]-1)*gaps[V]
            # each qTuple will have qNos
            QBlockDims = [
                # width x height in pixels
                gaps[0] * (numDims[V]-1) + bubbleDims[H],
                gaps[1] * (numDims[H]-1) + bubbleDims[V]
            ]
            # WATCH FOR BLUNDER(use .copy()) - qStart was getting passed by reference! (others args read-only)
            QBlocks.append(genQBlock(bubbleDims, QBlockDims, key, qStart.copy(),qTuple,gaps,vals,qType,orient))
            # Goes vertically down first
            qStart[V] += origGap[V]
        qStart[H] += origGap[H]
    return QBlocks

# The utility for GUI
def calcGaps(PointsX,PointsY,numsX,numsY):
    gapsX = ( abs(PointsX[0]-PointsX[1])/(numsX[0]-1),abs(PointsX[2]-PointsX[3]) )
    gapsY = ( abs(PointsY[0]-PointsY[1])/(numsY[0]-1),abs(PointsY[2]-PointsY[3]) )
    return (gapsX,gapsY)

def read_template(filename):
    with open(filename, "r") as f:
        return json.load(f)


templJSON={
'J' : read_template("inputs/Layouts/J_template.json"),
'H' : read_template("inputs/Layouts/H_template.json")
# 'H' : read_template("inputs/Layouts/H_template2.json")
}
TEMPLATES={}

for squad in templJSON.keys():
    TEMPLATES[squad] = Template(templJSON[squad])
    for k, QBlocks in templJSON[squad].items():
        if(k not in ["Dimensions","BubbleDimensions","Concatenations","Singles"]):
            # Add QBlock to array of grids
            TEMPLATES[squad].addQBlocks(k, QBlocks)