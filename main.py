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
    # TODO : write concatenation etc in generalized way from template here.
    # Note: This is actually a dummy/reference function. It is not part of the OMR checker so its implementation is completely subjective to user's requirements.
    global readFormat
    # for readFormat key
    omrResp['Squad'] = squad 
    resp={}
    # symbol for absent response
    UNMARKED = '' # 'X'
    for col, respKeys in readFormat[squad].items():
        resp[col] = ''.join([omrResp.get(k,UNMARKED) for k in respKeys])
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

resultFile = resultDir+'Results-'+timeNow+'.csv'
if(not os.path.exists(resultFile)):
    with open(resultFile,'a') as f:
        results=[resultSheetCols]
        pd.DataFrame(results).to_csv(f,header=False,index=False) #no .T for single list
else:
    print('WARNING : Appending to Previous Result file!')
counter=1

errorsArray=[sheetCols]
badRollsArray=[sheetCols]
verifyArray=[sheetCols]
multiMarkedArray=[sheetCols]

def appendArr(val,array,filename):
    array.append(val)
    if(not os.path.exists(filename)):
        with open(filename,'a') as f:
            pd.DataFrame([sheetCols],columns=sheetCols).to_csv(f,index=False,header=False)
    pad(val,sheetCols)
    with open(filename,'a') as f:
        pd.DataFrame([val],columns=sheetCols).to_csv(f,index=False,header=False)
        # pd.DataFrame(val).T.to_csv(f,header=False)

def stitch(img1,img2):
    return np.hstack((img1,img2));
    # return np.concatenate((img1,img2),axis=1)

start_time = int(time())

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--closeup", required=False, default=0,
    help="Whether or not input images have page contour visible.")
args = vars(ap.parse_args())

squadlang="XX"
mws, mbs = [],[]
with open(resultFile,'a') as f:
    for filepath in allOMRs:
        counter+=1
        finder = re.search(r'/.*/(.*)/(.*)\.'+ext[1:],filepath,re.IGNORECASE)
        if(finder):
            squadlang = finder.group(1)
            squad,lang = squadlang[0],squadlang[1]
            squadlang = squadlang+'/'
            filename = finder.group(2)
        else:
            filename = 'Nop'+str(counter)

        inOMR = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        OMRcrop = getROI(filepath,filename+ext,inOMR, closeup=int(args["closeup"]))
        #uniquify
        newfilename = filepath.split('/')[-3] + '_' + filename
        if(OMRcrop is None):
            err = move(NO_MARKER_ERR, filepath, errorPath+squadlang,filename)
            if(err):
                appendErr(err)
            continue
        respArray=[]
        OMRresponseDict,final_marked,multimarked,multiroll = readResponse(squad,OMRcrop,name = newfilename, save = saveMarkedDir+squadlang)

        # print(OMRresponseDict)
        
        #convert to ABCD, getRoll,etc
        resp = processOMR(squad,OMRresponseDict)
        print("\n\tRead Response: ", resp)
        #for aligning cols
        for q in readFormat[squad].keys():
            try:
                respArray.append(resp[q])
            except:
                respArray.append('')

        #This evaluates and returns the score attribute
        score = evaluate(resp, Answers[squad],Sections[squad],explain=explain)
        # if((multiroll or not (resp['Roll'] is not None and len(resp['Roll'])==11))):
        if(multimarked == 0):
            # Enter into Results sheet-
            results = [0,0,newfilename+'.jpg',filepath]+respArray+[score] #.append
            filesNotMoved+=1;
            # Write to results file (f is opened in append mode)
            pd.DataFrame(results).T.to_csv(f,header=False,index=False)
            print((counter,resp['Roll'],score))
            # print((counter,newfilename+'.jpg',resp['Roll'],','.join(respArray[1:]),'score : ',score))
        else:
            #multimarked file
            print('multiMarked, moving File: '+newfilename)
            err = move(multiMarkedError,filepath,multiMarkedPath+squadlang,newfilename+'.jpg')
            if(err):
                appendArr(err+respArray,multiMarkedArray,multiMarkedFile)

pd.DataFrame(errorsArray,columns=sheetCols).to_csv(errorFile,index=False,header=False)
pd.DataFrame(verifyArray,columns=sheetCols).to_csv(verifyFile,index=False,header=False)
pd.DataFrame(badRollsArray,columns=sheetCols).to_csv(badRollNosFile,index=False,header=False)
pd.DataFrame(multiMarkedArray,columns=sheetCols).to_csv(multiMarkedFile,index=False,header=False)

counterChecking=counter
timeChecking=(int(time()-start_time))
print('Total files : %d ' % (counterChecking-1))
print('Total files moved : %d ' % (filesMoved))
print('Total files not moved (should tally) : %d ' % (filesNotMoved))

print('Finished Checking %d files in %d seconds =>  ~%2f sec/OMR:)' % (counterChecking-1,timeChecking,round(timeChecking/float(counterChecking),2)))

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
