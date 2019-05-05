import cv2
import json
import numpy as np
from constants import *

def resize_util(img, u_width, u_height=None):
    if u_height == None:
        h,w=img.shape[:2]
        u_height = int(h*u_width/w)        
    return cv2.resize(img,(u_width,u_height))
### Image Template Part ###
template = cv2.imread('images/FinalCircle_hd.png',cv2.IMREAD_GRAYSCALE) #,cv2.CV_8UC1/IMREAD_COLOR/UNCHANGED 
template = resize_util(template, int(uniform_width_hd/templ_scale_fac))
template = cv2.GaussianBlur(template, (5, 5), 0)
template = cv2.normalize(template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# template_eroded_sub = template-cv2.erode(template,None)
template_eroded_sub = template - cv2.erode(template, kernel=np.ones((5,5)),iterations=5)
lontemplateinv = cv2.imread('images/lon-inv-resized.png',cv2.IMREAD_GRAYSCALE)
# lontemplateinv = imutils.rotate_bound(lontemplateinv,angle=180) 
# lontemplateinv = imutils.resize(lontemplateinv,height=int(lontemplateinv.shape[1]*0.75))
# cv2.imwrite('images/lontemplate-inv-resized.jpg',lontemplateinv)

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
    def __init__(self, dims, key, orig, colpts):
        # dims = (width, height)
        self.dims = dims
        self.key = key
        self.orig = orig
        self.colpts = colpts
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
    'QTYPE_MCQ':{
        'vals' : ['A','B','C','D'],
        'orient':'H'
    },
}

class Template():
    def __init__(self):
        self.QBlocks = []
        self.boxDims = [-1, -1]
        self.dims = [-1,-1]

    def setDims(self,dims):
        self.dims = dims

    def setBoxDims(self,dims):
        self.boxDims = dims

    # Expects boxDims to be set already
    def addQBlocks(self, key, rect):
        assert(self.boxDims != [-1, -1])
        # keyword arg unpacking followed by named args
        self.QBlocks += genGrid(self.boxDims, key, **rect,**qtype_data[rect['qType']])
        # self.QBlocks.append(QBlock(rect.orig, calcQBlockDims(rect), maketemplate(rect)))

def genQBlock(boxDims, QBlockDims, key, orig, qNos, gaps, vals, qType, orient):
    """
    Input:
    orig - start point
    qNo  - a qNos tuple
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - values of each alternative for a question

    Output:
    // Returns set of coordinates of a rectangular grid of points
    Returns a QBlock containing array of Qs and some metadata
    
        1 2 3 4
        1 2 3 4
        1 2 3 4

        (q1, q2, q3)

        00
        11
        22
        33
        44
        (q1d1,q1d2)

    """
    Qs=[]
    H, V = (0,1) if(orient=='H') else (1,0)
    # orig[0] += np.random.randint(-5,6) # test random shift
    orig[0] -= 10
    
    colpts = []
    colW, colH = (len(vals), len(qNos)) if(orient == 'H') else (len(qNos), len(vals))
    o = orig.copy()
    for i in range(colW):
        pt = o.copy()
        pts = []
        for j in range(colH):
            idx = [j, i]
            p = Pt(pt.copy(),qNos[idx[H]], qType, vals[idx[V]])
            pts.append(p)
            pt[1] += gaps[1] 
        pt[0] += boxDims[0]
        pt[1] += boxDims[1] - gaps[1]
        colpts.append(([o.copy(), pt.copy()], pts))
        o[0] += gaps[0]
    
    return QBlock(QBlockDims, key, orig, colpts)

def genGrid(boxDims, key, qType, orig, bigGaps, gaps, qNos, vals, orient='V'):
    """
    Input:
    boxDims - dimesions of single QBox
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
    gridData=np.array(qNos)
    if(len(gridData.shape)!=3 or gridData.size==0): # product of shape is zero
        print("genGrid: Invalid qNos array given", gridData)
        return []

    # ^ should also validate no overlap of rect points somehow?!

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
    gridRows, gridCols = gridData.shape[:2]
    numQsMax = max([max([len(qb) for qb in row]) for row in gridData])
    numVals = len(vals)

    QBlocks=[]
    # H and V are named with respect to orient == 'H', reverse their meaning when orient = 'V'
    H, V = (0,1) if(orient=='H') else (1,0)
    numDims = [numQsMax, numVals]
    # print(orig, numDims, gridRows,gridCols, gridData)
    # orient is also the direction of making QBlocks
    # Simple is powerful-
    origGap = [
        # bigGaps is indep of orientation
        bigGaps[0] + (numDims[V]-1)*gaps[H],
        bigGaps[1] + (numDims[H]-1)*gaps[V]
    ]
    # print(key, numDims, orig, gaps, bigGaps, origGap )
    qStart = orig.copy()

    for row in gridData:        
        qStart[V] = orig[V]
        for qTuple in row:
            QBlockDims = [
                # width x height in pixels
                gaps[0] * (numDims[V]-1) + boxDims[H],
                gaps[1] * (numDims[H]-1) + boxDims[V]
            ]
            # TONIGHT'S BLUNDER - qStart was getting passed by reference! (others args read-only)
            QBlocks.append(genQBlock(boxDims, QBlockDims, key, qStart.copy(),qTuple,gaps,vals,qType,orient))
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

def scalePts(pts,facX,facY):
    return [[int(pt[0]*facX),int(pt[1]*facY)] for pt in pts]


templJSON={
'J' : read_template("J_template.json"),
'H' : read_template("H_template.json")
}
TEMPLATES={'J': Template(),'H': Template()}


for squad in ['J','H']:
    TEMPLATES[squad].setDims(templJSON[squad]["Dimensions"])
    TEMPLATES[squad].setBoxDims(templJSON[squad]["boxDimensions"])
    # print(TEMPLATES[squad].dims)
    # print(TEMPLATES[squad].boxDims)
    for k, rect in templJSON[squad].items():
        if(k=="Dimensions" or k=="boxDimensions"):
            continue
        # rect["orig"], rect["gaps"], rect["bigGaps"] = scalePts([rect["orig"], rect["gaps"], rect["bigGaps"]], 0.54, 0.39 ) 
        # Add QBlock to array of grids
        TEMPLATES[squad].addQBlocks(k, rect)

    if(TEMPLATES[squad].dims == [-1, -1]):
        print(squad, "Invalid JSON! No reference dimensions given")
        exit(0)





"""
# mask = 255 * np.ones(pt - o,np.uint8).T
# pt = [0, 0] # relative to mask
# # print(mask.shape, gaps, vals, pt, boxDims)
# for v in vals:
#     mask[pt[1]:pt[1]+boxDims[1], pt[0]:pt[0]+boxDims[0]] = 0
#     pt[H] += gaps[H] 

# Actually need columns
# if(orient=='H'):
#     mask = 255 - mask
"""
