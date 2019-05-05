import numpy as np
from constants import *

class Pt():
    """Container for a Point Box on the OMR"""
    def __init__(self, x, y,val):
        self.x=x
        self.y=y
        self.val=val
    # overloaded
    def __init__(self, pt,val):
        self.x=pt[0]
        self.y=pt[1]
        self.val=val

class Q():
    """
    Container for a Question on the OMR
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """
    def __init__(self,qNo,qType, pts,ans=None):
        self.qNo = qNo
        self.qType = qType
        self.pts = pts
        self.ans = ans

def genRect(orig, qNos, gaps, vals, qType, orient):
    """
    Input:
    orig - start point
    qNo  - a qNos tuple
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - values of each alternative for a question

    Returns set of coordinates of a rectangular grid of points
    
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
    i0, i1 = (0,1) if(orient=='H') else (1,0)
    o=orig[:] # copy list
    for qNo in qNos:
        pt = o[:] #copy pt
        pts=[]
        for v in vals:
            pts.append(Pt(pt,v))
            pt[i0] += gaps[i0]
        o[i1] += gaps[i1]
        Qs.append( Q(qNo,qType, pts))
    return Qs

def genGrid(orig, qNos, bigGaps, gaps, vals, qType, orient='V'):
    """
    Input:
    orig- start point
    qNos - an array of qNos tuples(see below) that align with dimension of the big grid (gridDims extracted from here)
    bigGaps - (bigGapX,bigGapY) are the gaps between blocks
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - a 1D array of values of each alternative for a question
    orient - The way of arranging the vals (vertical or horizontal)

    Returns an array of Q objects (having their points) arranged in a rectangular grid

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
    npqNos=np.array(qNos)
    if(len(npqNos.shape)!=3 or npqNos.size==0): # product of shape is zero
        print("genGrid: Invalid qNos array given", npqNos)
        return []

    # ^ should also validate no overlap of rect points somehow?!
    gridHeight, gridWidth, numDigs = npqNos.shape
    numVals = len(vals)
    # print(orig, numDigs,numVals, gridWidth,gridHeight, npqNos)

    Qs=[]
    i0, i1 = (0,1) if(orient=='H') else (1,0)
    hGap, vGap = bigGaps[i1], bigGaps[i0]
    if(orient=='H'):
        hGap += (numVals-1)*gaps[i1]
        vGap += (numDigs-1)*gaps[i0]
    else:
        hGap += (numDigs-1)*gaps[i1]
        vGap += (numVals-1)*gaps[i0]
    qStart=orig[:]
    for row in npqNos:
        qStart[i1] = orig[i1]
        for qTuple in row:
            Qs += genRect(qStart,qTuple,gaps,vals,qType,orient)
            qStart[i1] += hGap
        qStart[i0] += vGap

    return Qs

# The utility for GUI            
def calcGaps(PointsX,PointsY,numsX,numsY):
    gapsX = ( abs(PointsX[0]-PointsX[1])/(numsX[0]-1),abs(PointsX[2]-PointsX[3]) )
    gapsY = ( abs(PointsY[0]-PointsY[1])/(numsY[0]-1),abs(PointsY[2]-PointsY[3]) )
    return (gapsX,gapsY)


def scalePts(pts,facX,facY):
    for pt in pts:
        pt = (pt[0]*facX,pt[1]*facY)

# Config for Manual fit - 
templJSON={}
templJSON['J']={
    'Medium' : {
    'qType' : QTYPE_MED,
    'orig' : [160,276],
    'bigGaps' : [115,51],
    'gaps' : [59,46],
    'qNos' : [[['Medium']]]
    },
    'Roll' : {
    'qType' : QTYPE_ROLL,
    'orig' : [218,276],
    'bigGaps' : [115,51],
    'gaps' : [58,46],
    'qNos' : [[['r'+str(i) for i in range(0,9)]]]
    },
    'Int1' : {
    'qType' : QTYPE_INT,
    'orig' : [903,276],
    'bigGaps' : [115,51],
    'gaps' : [59,46],
    'qNos' : [[('q'+str(i)+'.1','q'+str(i)+'.2') for i in range(5,8)]]
    },
    'Int2' : {
    'qType' : QTYPE_INT,
    'orig' : [1418,276],
    'bigGaps' : [115,51],
    'gaps' : [59,46],
    'qNos' : [[('q'+str(i)+'.1','q'+str(i)+'.2') for i in range(8,10)]]
    },
    'Mcq1' : {
    'qType' : QTYPE_MCQ,
    'orig' : [118,857],
    'bigGaps' : [115,183],
    'gaps' : [59,53],
    'qNos' : [[['q'+str(i) for i in range(1,5)],['q'+str(i) for i in range(10,14)]]]
    },
    'Mcq2' : {
    'qType' : QTYPE_MCQ,
    'orig' : [905,860],
    'bigGaps' : [115,180],
    'gaps' : [59,53],
    'qNos' : [[['q'+str(i) for i in range(14,17)]]]
    },
    'Mcq3' : {
    'qType' : QTYPE_MCQ,
    'orig' : [905,1195],
    'bigGaps' : [115,180],
    'gaps' : [59,53],
    'qNos' : [[['q'+str(i) for i in range(17,21)]]]
    }
}

templJSON['H']={
    'Int1' : {
    'qType' : QTYPE_INT,
    'orig' : [903,278],
    'bigGaps' : [128,51],
    'gaps' : [62,46],
    'qNos' : [[('q'+str(i)+'.1','q'+str(i)+'.2') for i in range(9,13)]]
    },
    'Int2' : {
    'qType' : QTYPE_INT,
    'orig' : [1655, 275],
    'bigGaps' : [128,51],
    'gaps' : [62,46],
    'qNos' : [[('q'+str(i)+'.1','q'+str(i)+'.2') for i in range(13,14)]]
    },
}
for k in ['Medium', 'Roll', 'Mcq1','Mcq2','Mcq3']:
    templJSON['H'][k]=templJSON['J'][k]

commonargs={
QTYPE_MED:{
'vals' : ['E','H'],
'orient':'V'
},
QTYPE_ROLL:{
'vals':range(10),
'orient':'V'
},
QTYPE_INT:{
'vals':range(10),
'orient':'V'
},
QTYPE_MCQ:{
'vals' : ['A','B','C','D'],
'orient':'H'
},
}
TEMPLATES={'J':[],'H':[]}
def maketemplate(rect):
    # keyword arg unpacking followed by named args
    return genGrid(**rect,**commonargs[rect['qType']])

# scale fit
for squad,templ in templJSON.items():
    for rect in templ.values():
        scalePts([rect['orig'],rect['bigGaps'],rect['gaps']],omr_templ_scale[0],omr_templ_scale[1])
        TEMPLATES[squad] += maketemplate(rect)
