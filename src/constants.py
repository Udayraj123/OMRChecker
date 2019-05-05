# In[5]:
kv=0

"""
Evaluate Hindi Files acc to diff template & anskey - Korba, Gwalior, Gonda _/ , Rajouri?

# ('OMR_Files_2018/Secunderabad_2060/JE/{}{}{}{}Normal/Mainpuri_JE_0045.tif', 'Secunderabad', '<<<<<<<')
# ('Bhilwara', '<<<<<<<', 'OMR_Files_2018/Bhilwara_5007/JE/Normal/Bhilawara_JE_0030.tif')
# ('najibabad', '<<<<<<<', 'OMR_Files_2018/najibabad_1147/JH/Normal/KV new tehri town_5135_JE_0003.tif')
# ('Bhiwadi', '<<<<<<<', 'OMR_Files_2018/Bhiwadi_5039/JH/Normal/amalapuram_JE_0079.tif')
# ('rewari', '<<<<<<<', 'OMR_Files_2018/rewari_1012/HH/Normal/Thrissur_HE_0001.tif')

"""
### Qtypes constants
QTYPE_INT,QTYPE_ROLL,QTYPE_MCQ,QTYPE_MED= range(4)
typeName={QTYPE_INT:"Integer",QTYPE_MCQ:"MCQ",QTYPE_MED:"MED",QTYPE_ROLL:"ROLL"}

showimglvl= 1
resetpos=[770,10]
verbose = 1 # Warning, the code may occasionally stop and showimg if verbose = 0
explain= 0
autorotate=1
saveMarked=1


#Intermediate - 
ext='.jpg'
minWhiteTHR,maxBlackTHR=255,0

stitched = 0;

# For normal images
thresholdRead_L =  60#116
# For already normalized(contrasted) images
thresholdRead_R =  60

thresholdCircle= 0.55 #matchTemplate returns 0 to 1
scaleRange=(0.75,0.95)

#Expert :p
display_height = 1000//1.3
display_width  = 1231//1.3

uniform_height = 1000 
uniform_width  = 1231
# original dims are (3527, 2494)
## Any input images should be resized to this--
uniform_height_hd = int(uniform_height*1.5)
uniform_width_hd = int(uniform_width*1.5)
circle_templ_scaledown=27

template_height,template_width = 1000, 1231 # If you change this, need to change startCoords too
omr_templ_scale = (uniform_width_hd/template_width, uniform_height_hd/template_height )
CV2_FONTSIZE=1.5

boxDimX,boxDimY=(30,30) # dims of the single square

directory ='images/OMR_Files/' if kv else 'images/OMR_Files/'
'feedsheets/errorSheet.csv'
errorpath=directory+'ErrorFiles/'
ErrorFile='feedsheets/ErrorFiles.csv'
WarningFile='feedsheets/WarningFiles.csv'
verifypath=directory+'verifyFiles/'
verifyFile='feedsheets/verifyFiles.csv'
badRollspath=directory+'badRollsFiles/'
badRollsFile='feedsheets/badRollsFiles.csv'
multiMarkedpath=directory+'multiMarkedFiles/'
multiMarkedFile='feedsheets/multiMarkedFiles.csv'
saveMarkedDir='markedOMRs/' #Should have Squad Langs inside
sheetCols=['batch','error','filename','path','roll']+['q'+str(i) for i in range(1,21)]#+['t'+str(i) for i in range(1,6)]
resultSheetCols=sheetCols+['score']

results_2018batch=1000
results_2018error=11
multiMarkedError=12
badRollError=13
verifyError=14 #Goes into verifyFiles, can be ignored? -Nope, especially for kvs

# for positioning image windows
windowX,windowY = 0,0 

#windowWidth = screen.get_width()
#windowHeight = screen.get_height()
windowWidth = 1200
windowHeight = 700



Directories = [multiMarkedpath,errorpath,verifypath,badRollspath]


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
    'JK' : {
        'Power1':{'ques':[1,2,3],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Fibo1':{'ques':[4,5,6,7],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'Power2':{'ques':[8,9,10],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Fibo2':{'ques':[11,12,13,14],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'allNone1':{'ques':[15,16],'marks':9},
        'allNone2':{'ques':[17,18],'marks':12},
        'allNone3':{'ques':[19,20],'marks':6},

    },
    'HK' : {
        'allNone1':{'ques':[1],'marks':8},
        'Power1':{'ques':[2,3,4,5],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'allNone2':{'ques':[6],'marks':12},
        'Fibo1':{'ques':[7,8,9,10,11],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'allNone3':{'ques':[12],'marks':8},
        'Power2':{'ques':[13,14,15],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Proxy1':{'ques':[16,17],'+marks':5,'-marks':3},
        'Power3':{'ques':[18,19,20],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},

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
