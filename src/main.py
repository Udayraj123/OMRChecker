
# In[62]:
import re
import os
import sys
import cv2
import glob
from time import localtime,strftime,time
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils #thru the pip package.
# from skimage.filters import threshold_adaptive
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk
from constants import *
from utils import *
from template import *

# Some globals -
window=gtk.Window()
screen = window.get_screen()

# In[14]:

def getRoll(squad,omr):
    roll=squad
    try:
        roll += omr['Medium']
        # length of roll number
        for i in range(9):
            roll+= str(omr['roll'+str(i)])
    except:
        print('WARNING : Incomplete Roll ',roll)
        return 'X'

    return roll

def getInt(Q):
    d1 = str(Q.get('.1','0'))
    d2 = str(Q.get('.2','x'))
    if(d2=='x'):
        return 'X'
    else:
        return d1+d2

def processOMR(squad,omr):
    resp={}
    roll='X'
    roll = getRoll(squad, omr)
    # TODO : write concatenation etc in generalized way from template here.
    # for qNo, qType in TYPEWISE_QS:
    #     resp[qNo]
    if(roll=='X'):
        # print('Warning : Error in Roll number! Moving File')
        return {'roll': None,'resp':resp}
    return {'roll': roll,'resp':resp}

# In[76]:

once = 0
def report(Status,streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks):
    global once
    if(not once):
        once = 1
        print('Question\tStatus \t Streak\tSection \tMarks_Update\tMarked:\tAnswer:')

    print('%s \t %s \t\t %s \t %s \t %s \t %s \t %s ' % (qNo,
          Status,str(streak), '['+scheme+'] ',(str(prevmarks)+' + '+str(currmarks)+' ='+str(marks)),str(marked),str(ans)))

# check sectionwise only.
def evaluate(resp,answers,sections,explain=False):
    marks = 0
    allans = answers.items()
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



# In[ ]:


# In[10]:


# In[78]:

# How about cropping out top 5% and bottom 25% part of the input?
# Also do something to take better nearby point
"""
Now dvlpments-
_// If multi-marked, try lower threshold, if still get multiple dark, move on,
else choose the most dark one.*** (cases of single bubble found marked even on unattempted)
    >> Instead choose the most dark ones always.

_// Make a questions class - It has options & their (4 by default) coordinates
_// pass array of questions to readResp: it just reads the coords and updates whether its marked or not.

>> Maybe get a 'tone' of the image and decide the threshold(clipped) acc to it?
    >> Rather make this 'tone' uniform in all images -> normalizing would give these dots the darkest value

Sort Bad Verifies = Raebareli, Dhirangarh, Ambarnath, Korba
"""

once = 0

allOMRs= glob.iglob(directory+'*/*/*'+ext)
# allOMRs = reversed(list(allOMRs))


timeNow=strftime("%I%p",localtime())

resultFile = 'results/Techno'+('KVJNV' if kv else '')+'Results2018-'+timeNow+'.csv'
if(not os.path.exists(resultFile)):
    with open(resultFile,'a') as f:
        results=[resultSheetCols]
        pd.DataFrame(results).to_csv(f,header=False,index=False) #no .T for single list
else:
    print('WARNING : Appending to Previous Result file!')
counter=1

"""
Done >Detected Images stored for each, also make an excel sheet connecting the filepath with roll
    >>markedOMRs - has ALL OMRS EXCEPT errorFiles(WITHOUT ENOUGH CIRCLE POINTS, debug OMRs)
    >>Should be corrected from Excel sheet.
yup >Manual Need ?
done >Proper Excel of ERRORS,VERIFY,WARNINGS,MULTIMARKED
done >Copy of those stored into verify folder
done >Set template of KV (its same,with change in Q5 only!), answer key & scheme of KV,
Rotate Feat.
KV Template RUN.

"""
#Make it live ?

# add rotate using templ2
# create err entries - fn,fp,batch 1000
p=int(time())
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
    # if(img1.shape[0]!=img2.shape[0]):
    #     print("Can't stitch different sized images")
    #     return None
    # return np.concatenate((img1,img2),axis=1)

squadlang="XX"
mws, mbs = [],[]
# start=35
with open(resultFile,'a') as f:
    # for i,filepath in enumerate(list(allOMRs)[start:start+5]):
    for i,filepath in enumerate(allOMRs):
        finder = re.search(r'/.*/(.*)/(.*)\.'+ext[1:],filepath,re.IGNORECASE)
        if(finder):
            squadlang = finder.group(1)
            squad,lang = squadlang[0],squadlang[1]
            squadlang = squadlang+'/'
            filename = finder.group(2)
        else:
            filename = 'Nop'+str(i)

        origOMR = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

        h,w = origOMR.shape[:2]
        if(w > uniform_width_hd and stitched):
            print("Assuming Stitched input.")
            w=int(w/2)
            OMRs=[origOMR[:,:w],origOMR[:,w:2*w]]
            thresholdReads=[thresholdRead_L,thresholdRead_R]
        else:
            OMRs = [origOMR]
            thresholdReads = [thresholdRead_L]

        local_id=0
        for thresholdRead,inOMR in zip(thresholdReads,OMRs):
            local_id+=1
            # OMRcrop = getROI(filepath,filename+ext,inOMR, closeup=True)
            OMRcrop = getROI(filepath,filename+ext,inOMR, closeup=False)
            #uniquify
            newfilename = filepath.split('/')[-3] + '_' + filename
            if(OMRcrop is None):
                err = move(results_2018error,filepath,errorPath+squadlang,newfilename+'.jpg')
                if(err):
                    appendArr(err,errorsArray,errorFile)
                continue

            respArray=[]
            OMRresponseDict,retimg,multimarked,multiroll = readResponse(squad,OMRcrop,name =newfilename)
            # print(OMRresponseDict)
            #convert to ABCD, getRoll,etc
            resp=processOMR(squad,OMRresponseDict)
            respArray.append(resp['roll']) #May append None
            for q in qNos[squad]:#for align
                try:
                    respArray.append(resp['resp'][q])
                except:
                    respArray.append('')
            # err contains the rest = [results_2018batch,error,filename,filepath2]
            #This evaluates and returns the score attribute
            score = evaluate(resp['resp'],Answers[squad+('K' if kv else '')],Sections[squad+('K' if kv else '')],explain=explain)
            # if((multiroll or not (resp['roll'] is not None and len(resp['roll'])==11))):
            if(multimarked == 0):
                # Enter into Results sheet-
                
                # err contains the rest = [results_2018batch,error,filename,filepath2]
                results = [0,0,newfilename+'.jpg',filepath]+respArray+[score] #.append
                filesNotMoved+=1;
                # Write to results file (f is opened in append mode)
                pd.DataFrame(results).T.to_csv(f,header=False,index=False)
                print((counter,resp['roll'],score))
                # print((counter,newfilename+'.jpg',resp['roll'],','.join(respArray[1:]),'score : ',score))
            else:
                #multimarked file
                print('multiMarked, moving File: '+newfilename)
                err = move(multiMarkedError,filepath,multiMarkedPath+squadlang,newfilename+'.jpg')
                if(err):
                    appendArr(err+respArray,multiMarkedArray,multiMarkedFile)
            counter+=1

            if(respArray==[]):
                respArray.append('') #Roll num
                for q in qNos[squad]:
                    respArray.append('')

            if(errorsArray!=[] and len(errorsArray[-1])!=30 and len(errorsArray[-1])!=25):
                errorsArray[-1]=errorsArray[-1]+respArray

            if(verifyArray!=[] and len(verifyArray[-1])!=30 and len(verifyArray[-1])!=25):
                verifyArray[-1]=verifyArray[-1]+respArray

            if(badRollsArray!=[] and len(badRollsArray[-1])!=30 and len(badRollsArray[-1])!=25):
                badRollsArray[-1]=badRollsArray[-1]+respArray
            if(multiMarkedArray!=[] and len(multiMarkedArray[-1])!=30 and len(multiMarkedArray[-1])!=25):
                multiMarkedArray[-1]=multiMarkedArray[-1]+respArray


pd.DataFrame(errorsArray,columns=sheetCols).to_csv('feedsheets/errorSheet.csv',index=False,header=False)
pd.DataFrame(verifyArray,columns=sheetCols).to_csv('feedsheets/verifySheet.csv',index=False,header=False)
pd.DataFrame(badRollsArray,columns=sheetCols).to_csv('feedsheets/badRollSheet.csv',index=False,header=False)
pd.DataFrame(multiMarkedArray,columns=sheetCols).to_csv('feedsheets/multiMarkedSheet.csv',index=False,header=False)
# print(errorsArray,verifyArray,badRollsArray)


counterChecking=counter
takentimechecking=(int(time()-p)+1)
print('Finished Checking %d files in %d seconds =>  %f sec/OMR:)' % (counterChecking,takentimechecking,float(takentimechecking)/float(counterChecking)))
# print('Total files moved : %d ' % (filesMoved))
# print('Total files not moved (shud match) : %d ' % (filesNotMoved))

# Use this data to train as +ve feedback
# print(thresholdCircles)
if(showimglvl>=0):
    for x in [badThresholds,veryBadPoints,thresholdCircles, mws, mbs]:
        if(x!=[]):
            x=pd.DataFrame(x)
            print( x.describe() )
            plt.plot(range(len(x)),x)
            plt.show()
        else:
            print(x)
