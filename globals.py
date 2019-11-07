"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""

"""
Constants
"""
display_height = int(480)
display_width  = int(640)
windowWidth = 1280
windowHeight = 720

saveMarked = 1
showimglvl = 2
saveimglvl = 0
PRELIM_CHECKS=0
saveImgList = {}
resetpos = [0,0]
explain= 0
# autorotate=1

BATCH_NO=1000
NO_MARKER_ERR=12
MULTI_BUBBLE_WARN=15

# For preProcessing
GAMMA_LOW = 0.7
GAMMA_HIGH = 1.25

ERODE_SUB_OFF = 1

# For new ways of determining threshold
MIN_GAP, MIN_STD = 30, 25
MIN_JUMP = 18
# If only not confident, take help of globalTHR
CONFIDENT_JUMP = MIN_JUMP+15
JUMP_DELTA = 30
# MIN_GAP : worst case gap of black and gray

# Templ alignment parameters
ALIGN_RANGE  = range(-5,6,1) 
#TODO ^THIS SHOULD BE IN LAYOUT FILE AS ITS RELATED TO DIMENSIONS
# ALIGN_RANGE  = [-6,-4,-2,-1,0,1,2,4,6]

# max threshold difference for template matching
thresholdVar = 0.41

# TODO: remove unnec variables here- 
thresholdCircle = 0.3 #matchTemplate returns 0 to 1
# thresholdCircle = 0.4 #matchTemplate returns 0 to 1
markerScaleRange=(35,100)
markerScaleSteps = 10 
templ_scale_fac = 17

# Presentation variables

uniform_height = int(1231 / 1.5)
uniform_width = int(1000 / 1.5)
# Original dims are about (3527, 2494)

## Any input images should be resized to this--
uniform_width_hd = int(uniform_width*1.5)
uniform_height_hd = int(uniform_height*1.5)

TEXT_SIZE=0.95
CLR_BLACK = (50,150,150)
CLR_WHITE = (250,250,250)
CLR_GRAY = (120,120,120)
# CLR_DARK_GRAY = (190,190,190)
CLR_DARK_GRAY = (90,90,90)

MIN_PAGE_AREA = 80000

# Filepaths
class Paths:
	def __init__(self, input, output):
		self.input = input
		self.output = output
		self.omrInputDir=f'{input}/OMR_Files/'
		self.saveMarkedDir=f'{output}/CheckedOMRs/' 
		self.resultDir=f'{output}/Results/'
		self.manualDir=f'{output}/Manual/'
		self.errorsDir=f'{self.manualDir}ErrorFiles/'
		self.badRollsDir=f'{self.manualDir}BadRollNosFiles/'
		self.multiMarkedDir=f'{self.manualDir}MultiMarkedFiles/'

"""
Variables
"""
filesMoved=0
filesNotMoved=0

# for positioning image windows
windowX,windowY = 0,0 


# TODO: move to template or similar json
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

