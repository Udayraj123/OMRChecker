

"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""

# In[62]:
import re
import os
import sys
import cv2
import glob
import argparse
from time import localtime,strftime,time
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils #thru the pip package.
# from skimage.filters import threshold_adaptive
from globals import *
from utils import *
from template import *

# Sometime later-
# from colorama import init
# init()
# from colorama import Fore, Back, Style

def move(error,filepath,filepath2,filename):
    print("Dummy Move:  "+filepath, " --> ",filepath2 + filename)
    # if(error!=NO_MARKER_ERR):
    #     print("Error-Code: "+str(error))
    global filesNotMoved
    filesNotMoved += 1
    return [BATCH_NO,error,filename,filepath2]
    global filesMoved
    # print(filepath,filepath2,filename,array)
    if(os.path.exists(filepath)):
        if(os.path.exists(filepath2+filename)):
            print('ERROR : Duplicate file at '+filepath2+filename)
            # exit(0)
            if(1):
                return None
        os.rename(filepath,filepath2+filename)
        append = [BATCH_NO,error,filename,filepath2]
        filesMoved+=1
        print("Moved.")
        return append
    else:
        print('File already moved')
        return None

def processOMR(squad, omrResp):
    # Note: This is a reference function. It is not part of the OMR checker 
    #       So its implementation is completely subjective to user's requirements.
    
    # Additional Concatenation key
    omrResp['Squad'] = squad 
    resp={}
    # symbol for absent response
    UNMARKED = '' # 'X'

    # Multi-Integer Type Qs / RollNo / Name
    for qNo, respKeys in TEMPLATES[squad].concats.items():
        resp[qNo] = ''.join([omrResp.get(k,UNMARKED) for k in respKeys])
    # Normal Questions
    for qNo in TEMPLATES[squad].singles:
        resp[qNo] = omrResp.get(qNo,UNMARKED)
    # Note: Concatenations and Singles together should be mutually exclusive 
    # and should cover all questions in the template(exhaustive)
    # ^TODO add a warning if omrResp has unused keys remaining
    # Done : write concatenation etc in generalized way from template here.
    return resp

# In[76]:

def report(Status,streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks):
    print('%s \t %s \t\t %s \t %s \t %s \t %s \t %s ' % (qNo,
          Status,str(streak), '['+scheme+'] ',(str(prevmarks)+' + '+str(currmarks)+' ='+str(marks)),str(marked),str(ans)))

# check sectionwise only.
def evaluate(resp,answers,sections,explain=False):
    marks = 0
    allans = answers.items()
    if(explain):
        print('Question\tStatus \t Streak\tSection \tMarks_Update\tMarked:\tAnswer:')
    for scheme,section in sections.items():
        sectionques = section['ques']
        prevcorrect=None
        allflag=1
        streak=0
        for q in sectionques:
            qNo='q'+str(q)
            ans=answers[qNo]
            marked = resp.get(qNo, 'X')
            firstQ = sectionques[0]
            lastQ = sectionques[len(sectionques)-1]
            unmarked = marked=='X' or marked==''
            bonus = 'BONUS' in ans
            correct = bonus or (marked in ans)
            inrange=0

# ('q13(Power2) Correct(streak0) -3 + 2 = -1', 'C', ['C'])
# ('q14(Power2) Correct(streak0) -1 + 2 = 1', 'A', ['A'])
# ('q15(Power2) Incorrect(streak0) 1 + -1 = 0', 'C', ['B'])

            if(unmarked or int(q)==firstQ):
                streak=0
            elif(prevcorrect == correct):
                streak+=1
            else:
                streak=0


            if( 'allNone' in scheme):
                #loop on all sectionques
                allflag = allflag and correct
                if(q == lastQ ):
                    #at the end check allflag
                    prevcorrect = correct
                    currmarks = section['marks'] if allflag else 0
                else:
                    currmarks = 0

            elif('Proxy' in scheme):
                a=int(ans[0])
                #proximity check
                inrange = 1 if unmarked else (float(abs(int(marked) - a))/float(a) <= 0.25)
                currmarks = section['+marks'] if correct else (0 if inrange else -section['-marks'])

            elif('Fibo' in scheme or 'Power' in scheme or 'Boom' in scheme):
                currmarks = section['+seq'][streak] if correct else (0 if unmarked else -section['-seq'][streak])
            elif('TechnoFin' in scheme):
                currmarks = 0
            else:
                print('Invalid Sections')
            prevmarks=marks
            marks += currmarks

            if(explain):
                if bonus:
                    report('BonusQ',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif correct:
                    report('Correct',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif unmarked:
                    report('Unmarked',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif inrange:
                    report('InProximity',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                else:
                    report('Incorrect',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)

            prevcorrect = correct

    return marks

# os.sep is not an issue here in iglob
allOMRs = list(glob.iglob(OMR_INPUT_DIR+'*/*/*.jpg')) + list(glob.iglob(OMR_INPUT_DIR+'*/*/*.png'))

timeNowHrs=strftime("%I%p",localtime())

def pad(val,array):
    if(len(val) < len(array)):
        for i in range(len(array)-len(val)):
            val.append('V')


def appendArr(val,array,filename):
    array.append(val)

start_time = int(time())

# construct the argument parse and parse the arguments
argparser = argparse.ArgumentParser()

argparser.add_argument("-c", "--noCrop", required=False, dest='noCrop', action='store_true', help="Disable page contour detection - the program will take image as it is.")
argparser.add_argument("-m", "--noMarkers", required=False, dest='noMarkers', action='store_true', help="Disable marker detection - if page is already cropped at marker points.")
argparser.add_argument("-a", "--noAlign", required=False, dest='noAlign', action='store_true', help="Disable automatic template alignment - if columns are not varying significantly(saves computation).")
argparser.add_argument("-l", "--setLayout", required=False, dest='setLayout', action='store_true', help="Use this option to set up OMR template layout for once.(*_template.json file)")

args = vars(argparser.parse_args())
print('\tConfig:')
print("noCrop(Scanned) Images : "+str(args["noCrop"]))
print("Flat(Scanned) papers : "+str(args["noAlign"]))
print("OMR Already Cropped(No Markers) : "+str(args["noMarkers"]))
print('')

respCols, sheetCols, resultSheetCols = {},{},{}
resultFiles,resultFileObj = {},{}
errorsArray, badRollsArray, multiMarkedArray = {},{},{}
# Loop over squads
for squad in templJSON.keys():
    # Concats + Singles includes : all template keys including RollNo if present
    respCols[squad] = list(TEMPLATES[squad].concats.keys())+TEMPLATES[squad].singles
    sheetCols[squad]=['batch','error','filename','path']+respCols[squad]
    resultSheetCols[squad]=sheetCols[squad]+['score'] 
    resultFiles[squad] = resultDir+'Results_'+squad+'_'+timeNowHrs+'.csv'
    errorsArray[squad] = [ sheetCols[squad] ]
    badRollsArray[squad] = [ sheetCols[squad] ]
    multiMarkedArray[squad] = [ sheetCols[squad] ]
    print("resultFile:", resultFiles[squad])
    if(not os.path.exists(resultFiles[squad])):
        resultFileObj[squad] = open(resultFiles[squad],'a') # still append mode req [THINK!]
        # Create Header Columns
        pd.DataFrame([resultSheetCols[squad]]).to_csv(resultFileObj[squad],header=False,index=False) 
    else:
        print('WARNING : Appending to Previous Result file for: '+squad)
        resultFileObj[squad] = open(resultFiles[squad],'a')

squadlang="XX"
filesCounter=0
mws, mbs = [],[]
# PRELIM_CHECKS for thresholding
if(PRELIM_CHECKS):
    TEMPLATE = TEMPLATES["H"]
    ALL_WHITE = 255 * np.ones((TEMPLATE.dims[1],TEMPLATE.dims[0]), dtype='uint8')
    OMRresponseDict,final_marked,multimarked,multiroll = readResponse("H",ALL_WHITE,name = "ALL_WHITE", save = None, noAlign=True)
    print("ALL_WHITE",OMRresponseDict)
    if(OMRresponseDict!={}):
        print("Preliminary Checks Failed.")
        exit(0)
    ALL_BLACK = np.zeros((TEMPLATE.dims[1],TEMPLATE.dims[0]), dtype='uint8')
    OMRresponseDict,final_marked,multimarked,multiroll = readResponse("H",ALL_BLACK,name = "ALL_BLACK", save = None, noAlign=True)
    print("ALL_BLACK",OMRresponseDict)
    show("Confirm : All are Black",final_marked,1,1)

for filepath in allOMRs:
    filesCounter+=1
    # Including stupid windows convention
    filepath = filepath.replace(os.sep,'/')

    # Prefixing a 'r' to use raw string (escape character '\' is taken literally)
    finder = re.search(r'/.*/(.*)/(.*)',filepath,re.IGNORECASE)

    if(finder):
        squadlang = finder.group(1)
        squad,lang = squadlang[0],squadlang[1]
        #FIXME TEMP JUGAAD
        squadlang = squadlang+'/'
        filename = finder.group(2)
    else:
        filename = 'Nop'+str(filesCounter)
        print("Error: Filepath not matching to Regex: "+filepath)
        continue
    # TODO make it independent of squad rule
    if(squad not in ['H','J']):
        print("Error: Unexpected Squad Folder-",squad, 'Filepath:', filepath)
        exit(0)

    inOMR = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    print('\n\t',(filesCounter ,filepath), inOMR.shape)
    OMRcrop = getROI(inOMR,filename, noCrop=args["noCrop"], noMarkers=args["noMarkers"])
    if(OMRcrop is None):
        err = move(NO_MARKER_ERR, filepath, errorPath+squadlang,filename)
        if(err):
            errorsArray[squad].append(err+respCols[squad])
            # appendArr(err,errorsArray[squad],manualDir+"ErrorFiles_"+squad+".csv")
        continue

    if(args["setLayout"]):
        show("Sample OMR", resize_util(OMRcrop,TEMPLATES[squad].dims[0],TEMPLATES[squad].dims[1]), 0)
        templateLayout = drawTemplateLayout(OMRcrop, TEMPLATES[squad], shifted=False, border=2)
        show("Template Layout", templateLayout)
        print('Setup Layout Note: Press Q to continue, Ctrl+C to exit')
        continue
    #uniquify
    newfilename = filepath.split('/')[-3] + '_' + filename
    OMRresponseDict,final_marked,multimarked,multiroll = readResponse(squad,OMRcrop,name = newfilename, save = saveMarkedDir+squadlang, noAlign=args["noAlign"])

    #convert to ABCD, getRoll,etc
    resp = processOMR(squad,OMRresponseDict)
    print("Read Response: \t", resp)

    #This evaluates and returns the score attribute
    score = evaluate(resp, Answers[squad],Sections[squad],explain=explain)
    respArray=[]
    for k in respCols[squad]:
        respArray.append(resp[k])
    # if((multiroll or not (resp['Roll'] is not None and len(resp['Roll'])==11))):
    if(multimarked == 0):
        # Enter into Results sheet-
        results = [0,0,newfilename,filepath]+respArray+[score] #.append
        filesNotMoved+=1;
        # Write to results file (resultFileObj is opened in append mode)
        pd.DataFrame(results).T.to_csv(resultFileObj[squad],header=False,index=False)
        
        print("Checked:",(filesCounter,newfilename,score))

        # print(filesCounter,newfilename,resp['Roll'],'score : ',score)
    else:
        #multimarked file
        print('multiMarked, moving File: '+newfilename)
        err = move(MULTI_BUBBLE_ERR,filepath,multiMarkedPath+squadlang,newfilename)
        if(err):
            multiMarkedArray[squad].append(err+respArray)
            # appendArr(err+respArray,multiMarkedArray[squad],manualDir+"MultiMarkedFiles_"+squad+".csv")



for squad in templJSON.keys():
    resultFileObj[squad].close()
    pd.DataFrame(errorsArray[squad],columns=sheetCols[squad]).to_csv(manualDir+"ErrorFiles_"+squad+".csv",index=False,header=False)    
    pd.DataFrame(badRollsArray[squad],columns=sheetCols[squad]).to_csv(manualDir+"BadRollNoFiles_"+squad+".csv",index=False,header=False)
    pd.DataFrame(multiMarkedArray[squad],columns=sheetCols[squad]).to_csv(manualDir+"MultiMarkedFiles_"+squad+".csv",index=False,header=False)

timeChecking=int(time()-start_time) if filesCounter else 1
print('')
print('Total files processed : %d ' % (filesCounter))
print('Total files moved : %d ' % (filesMoved))
print('Total files not moved (Sum should tally) : %d ' % (filesNotMoved))

if(showimglvl<=0):
    print('\nFinished Checking %d files in %d seconds(~%.1f minutes)' % 
    (filesCounter, timeChecking, round(timeChecking/float(60*filesCounter+1),2) ))
    print('OMR Processing rate :  ~%.2f sec/OMR' % (round(timeChecking/float(filesCounter+1), 2) ))
    print('OMR Processing speed : ~%.2f OMRs/minute' % (round(float(filesCounter*60+1)/timeChecking, 2) ))
else:
    print("Total time taken :", timeChecking,"seconds")

if(filesCounter==0):
    print("INFO: No Images found. Is your directory structure correct? \n\tPossibly forgot to rename 'OMR_Files_Sample' to 'OMR_Files'?")

# Use this data to train as +ve feedback
if(showimglvl>=0 and filesCounter>10):
    for x in [badThresholds,veryBadPoints,thresholdCircles, mws, mbs]:
        if(x!=[]):
            x=pd.DataFrame(x)
            print( x.describe() )
            plt.plot(range(len(x)),x)
            plt.show()
        else:
            print(x)
