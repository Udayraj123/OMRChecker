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

def processOMR(squad, omrResp):
    # Done : write concatenation etc in generalized way from template here.
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

allOMRs= glob.iglob(OMR_INPUT_DIR+'*/*/*'+ext)

timeNow=strftime("%I%p",localtime())


def appendArr(val,array,filename):
    array.append(val)
    if(not os.path.exists(filename)):
        with open(filename,'a') as f:
            pd.DataFrame([sheetCols],columns=sheetCols).to_csv(f,index=False,header=False)
    pad(val,sheetCols)
    with open(filename,'a') as f:
        pd.DataFrame([val],columns=sheetCols).to_csv(f,index=False,header=False)
        # pd.DataFrame(val).T.to_csv(f,header=False)

# def stitch(img1,img2):
#     return np.hstack((img1,img2));
    # return np.concatenate((img1,img2),axis=1)

start_time = int(time())

# construct the argument parse and parse the arguments
argparser = argparse.ArgumentParser()

argparser.add_argument("-c", "--closeUp", required=False, dest='closeUp', action='store_true', help="Whether or not input images have page contour visible.")
argparser.add_argument("-m", "--nomarkers", required=False, dest='noMarkers', action='store_true', help="Whether or not input images have markers visible.(Page is already cropped)")
args = vars(argparser.parse_args())
print("CloseUp Images(Scanned) : "+str(args["closeUp"]))
print("Page(s) Already Cropped : "+str(args["noMarkers"]))

respCols, sheetCols, resultSheetCols = {},{},{}
resultFiles,resultFileObj = {},{}
errorsArray, badRollsArray, multiMarkedArray = {},{},{}
# Loop over squads
for k in templJSON.keys():
    # Concats + Singles includes : all template keys including RollNo if present
    respCols[k] = list(TEMPLATES[k].concats.keys())+TEMPLATES[k].singles
    sheetCols[k]=['batch','error','filename','path']+respCols[k]
    resultSheetCols[k]=sheetCols[k]+['score'] 
    resultFiles[k] = resultDir+'Results_'+k+'_'+timeNow+'.csv'
    # check if this line required
    errorsArray[k] = badRollsArray[k] = multiMarkedArray[k] = [sheetCols[k]]
    print(resultFiles[k])
    if(not os.path.exists(resultFiles[k])):
        resultFileObj[k] = open(resultFiles[k],'a')
        # Create Header Columns
        pd.DataFrame([resultSheetCols[k]]).to_csv(resultFileObj[k],header=False,index=False) 
    else:
        print('WARNING : Appending to Previous Result file for: '+k)
        resultFileObj[k] = open(resultFiles[k],'a')

squadlang="XX"
filesCounter=0
mws, mbs = [],[]
for filepath in allOMRs:
    filesCounter+=1
    finder = re.search(r'/.*/(.*)/(.*)\.'+ext[1:],filepath,re.IGNORECASE)
    if(finder):
        squadlang = finder.group(1)
        squad,lang = squadlang[0],squadlang[1]
        squadlang = squadlang+'/'
        filename = finder.group(2)
    else:
        filename = 'Nop'+str(filesCounter)
        print("Error: Filepath not matching to Regex: "+filepath)
        continue

    inOMR = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    OMRcrop = getROI(filepath,filename+ext,inOMR, closeUp=args["closeUp"], noMarkers=args["noMarkers"])
    #uniquify
    newfilename = filepath.split('/')[-3] + '_' + filename
    if(OMRcrop is None):
        err = move(NO_MARKER_ERR, filepath, errorPath+squadlang,filename)
        if(err):
            appendErr(err)
        continue
    OMRresponseDict,final_marked,multimarked,multiroll = readResponse(squad,OMRcrop,name = newfilename, save = saveMarkedDir+squadlang)

    # print(OMRresponseDict)
    
    #convert to ABCD, getRoll,etc
    resp = processOMR(squad,OMRresponseDict)
    print("Read Response: \n\t", resp,"\n")
    #This evaluates and returns the score attribute
    score = evaluate(resp, Answers[squad],Sections[squad],explain=explain)
    respArray=[]
    for k in respCols[squad]:
        respArray.append(resp[k])
    # if((multiroll or not (resp['Roll'] is not None and len(resp['Roll'])==11))):
    if(multimarked == 0):
        # Enter into Results sheet-
        results = [0,0,newfilename+'.jpg',filepath]+respArray+[score] #.append
        filesNotMoved+=1;
        # Write to results file (resultFileObj is opened in append mode)
        pd.DataFrame(results).T.to_csv(resultFileObj[squad],header=False,index=False)
        
        print((filesCounter,newfilename+'.jpg',score))

        # print(filesCounter,newfilename+'.jpg',resp['Roll'],'score : ',score)
    else:
        #multimarked file
        print('multiMarked, moving File: '+newfilename)
        err = move(MULTI_BUBBLE_ERR,filepath,multiMarkedPath+squadlang,newfilename+'.jpg')
        if(err):
            appendArr(err+respArray,multiMarkedArray[squad],manualDir+"multiMarkedFiles"+squad+".csv")



for squad in templJSON.keys():
    resultFileObj[squad].close()
    pd.DataFrame(errorsArray[squad],columns=sheetCols[squad]).to_csv(manualDir+"errorFiles_"+squad+".csv",index=False,header=False)    
    pd.DataFrame(badRollsArray[squad],columns=sheetCols[squad]).to_csv(manualDir+"badRollNoFiles_"+squad+".csv",index=False,header=False)
    pd.DataFrame(multiMarkedArray[squad],columns=sheetCols[squad]).to_csv(manualDir+"multiMarkedFiles_"+squad+".csv",index=False,header=False)

timeChecking=(int(time()-start_time))
print('Total files : %d ' % (filesCounter))
print('Total files moved : %d ' % (filesMoved))
print('Total files not moved (Sum should tally) : %d ' % (filesNotMoved))

print('Finished Checking %d files in %d seconds =>  ~%2f sec/OMR:)' % (filesCounter,timeChecking,round(timeChecking/float(filesCounter+1),2)))

# Use this data to train as +ve feedback
if(showimglvl>=0):
    for x in [badThresholds,veryBadPoints,thresholdCircles, mws, mbs]:
        if(x!=[]):
            x=pd.DataFrame(x)
            print( x.describe() )
            plt.plot(range(len(x)),x)
            plt.show()
        else:
            print(x)
