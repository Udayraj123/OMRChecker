"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""

"""
Constants
"""
windowWidth = 1280
windowHeight = 720

saveMarked=1
showimglvl = 2
saveimglvl = 1
saveImgList = {}
resetpos = [0,0]
explain= 0
# autorotate=1

BATCH_NO=1000
NO_MARKER_ERR=12
badRollError=13
verifyError=14 #Goes into verifyFiles, can be ignored? -Nope, especially for kvs


#Intermediate - 
ext='.jpg'
minWhiteTHR,maxBlackTHR=255,0

stitched = 0;

# For normal images
thresholdRead =  116

# For preProcessing
GAMMA_LOW = 0.7
GAMMA_HIGH = 1.25

ERODE_SUB_OFF = 1

# For new ways of determining threshold
MIN_GAP, MIN_STD = 30, 25
MIN_JUMP = 20
JUMP_DELTA = 40
# MIN_GAP : worst case gap of black and gray

# Templ alignment parameters
ALIGN_RANGE  = range(-5,6,1) #
# ALIGN_RANGE  = [-6,-4,-2,-1,0,1,2,4,6]

# max threshold difference for template matching
thresholdVar = 0.41

# TODO: remove unnec variables here- 
thresholdCircle = 0.3 #matchTemplate returns 0 to 1
# thresholdCircle = 0.4 #matchTemplate returns 0 to 1
scaleRange=(0.35,0.95)
match_precision = 20 # > 1


# Original scan dimensions: 3543 x 2478
display_height = int(800)
display_width  = int(800)

uniform_width = int(1000 / 1.5)
uniform_height = int(1231 / 1.5)
# original dims are (3527, 2494)

## Any input images should be resized to this--
uniform_width_hd = int(uniform_width*1.5)
uniform_height_hd = int(uniform_height*1.5)

templ_scale_fac = 17
MIN_PAGE_AREA = 80000

TEXT_SIZE=1.5

OMR_INPUT_DIR ='inputs/OMR_Files/'
manualDir='outputs/Manual/'
resultDir='outputs/Results/'
errorPath=manualDir+'errorFiles/'
errorFile=manualDir+'errorFiles.csv'
verifyPath=manualDir+'verifyFiles/'
verifyFile=manualDir+'verifyFiles.csv'
badRollNosFile=manualDir+'badRollNosFiles.csv'
badRollsPath=manualDir+'badRollNosFiles/'
multiMarkedPath=manualDir+'multiMarkedFiles/'
multiMarkedFile=manualDir+'multiMarkedFiles.csv'
saveMarkedDir='outputs/checkedOMRs/' 


sheetCols=['batch','error','filename','path','roll']+['q'+str(i) for i in range(1,21)]#+['t'+str(i) for i in range(1,6)]
resultSheetCols=sheetCols+['score'] 


"""
Variables
"""
filesMoved=0
filesNotMoved=0

# for positioning image windows
windowX,windowY = 0,0 


Answers={
'J':{
'q1': ['B'],'q2':['B'],'q3':['B'],'q4': ['C'],'q5': ['0','00'],'q6': ['0','00'],'q7': ['4','04'],
'q8': ['9','09'],    'q9': ['11','11'],    'q10': ['C'],'q11': ['C'],'q12': ['B'],'q13': ['C'],
'q14': ['C'],'q15': ['B'],'q16': ['C'],'q17': ['BONUS'],'q18': ['A'],'q19': ['C'],'q20': ['B']},
'H':{
'q1': ['B'],'q2':['BONUS'],'q3':['A'],'q4': ['B'],'q5': ['A'],'q6': ['B'],'q7': ['B'],
'q8': ['C'],    'q9': ['4','04'],'q10': ['4','04'],'q11': ['5','05'],'q12': ['1','01'],'q13': ['28'],
'q14': ['C'],'q15': ['B'],'q16': ['C'],'q17': ['C'],'q18': ['C'],'q19': ['B'],'q20': ['C']},
'JK':{
'q1': ['B'],'q2':['B'],'q3':['B'],'q4': ['C'],'q5': ['0','00'],'q6': ['0','00'],'q7': ['4','04'],
'q8': ['9','09'],    'q9': ['11','11'],    'q10': ['C'],'q11': ['C'],'q12': ['B'],'q13': ['C'],
'q14': ['C'],'q15': ['B'],'q16': ['C'],'q17': ['BONUS'],'q18': ['A'],'q19': ['C'],'q20': ['B']},
'HK':{
'q1': ['B'],'q2':['BONUS'],'q3':['A'],'q4': ['B'],'q5': ['B'],'q6': ['B'],'q7': ['B'],
'q8': ['C'],    'q9': ['4','04'],'q10': ['4','04'],'q11': ['5','05'],'q12': ['1','01'],'q13': ['28'],
'q14': ['C'],'q15': ['B'],'q16': ['C'],'q17': ['C'],'q18': ['C'],'q19': ['B'],'q20': ['C']},
}

# Fibo is across the sections - Q4,5,6,7,13,
Sections = {
'J':{
'Fibo1':{'ques':[1,2,3,4],'+seq':[2,3,5,8],'-seq':[0,1,1,2]},
'Power1':{'ques':[5,6,7,8,9],'+seq':[1,2,4,8,16],'-seq':[0,0,0,0,0]},
'Fibo2':{'ques':[10,11,12,13],'+seq':[2,3,5,8],'-seq':[0,1,1,2]},
'allNone1':{'ques':[14,15,16],'marks':12},
'Boom1':{'ques':[17,18,19,20],'+seq':[3,3,3,3],'-seq':[1,1,1,1]},
},
'H':{
'Boom1':{'ques':[1,2,3,4],'+seq':[3,3,3,3],'-seq':[1,1,1,1]},
'Fibo1':{'ques':[5,6,7,8],'+seq':[2,3,5,8],'-seq':[0,1,1,2]},
'Power1':{'ques':[9,10,11,12,13],'+seq':[1,2,4,8,16],'-seq':[0,0,0,0,0]},
'allNone1':{'ques':[14,15,16],'marks':12},
'Boom2':{'ques':[17,18,19,20],'+seq':[3,3,3,3],'-seq':[1,1,1,1]},
},
}

qNos={
'J':['q'+str(i) for i in range(1,21)],
'H':['q'+str(i) for i in range(1,21)]
}
