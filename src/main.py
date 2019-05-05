
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



# cv2.createTrackbar('boxDimY', 'imageWindow', 6, 10, someFunctionCallBack)
# waitQ()
"""
Note : You'll get this error if the code is accessing pixels out of bounds of the image
error: (-215) A.size == arrays[i0]->size in function init
"""

# In[14]:

def getRoll(squad,rolldict):
    roll=squad
    try:
        roll += rolldict['Medium']
        for i in range(9):
            roll+= str(rolldict['Roll'+str(i)])
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
    for q in omr.items():
        typeq = q[0]
        Q=q[1]
        if(typeq == 'Roll'):
            roll = getRoll(squad,Q)
        elif('MCQ' in typeq):
            qNo='q'+(typeq.replace('MCQ',''))
            resp[qNo] = Q['val']
        elif('INT' in typeq):
            qNo='q'+(typeq.replace('INT',''))
            resp[qNo] = getInt(Q)
        else:
            pass
            # print("ERROR: Can't process item : ",q)

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


template = cv2.imread('images/FinalCircle_hd.png',cv2.IMREAD_GRAYSCALE) #,cv2.CV_8UC1/IMREAD_COLOR/UNCHANGED 
w,h = template.shape
template_scale=uniform_height_hd/(h*circle_templ_scaledown)
w,h = int(w*template_scale),int(h*template_scale)
template = cv2.resize(template,(w,h))
lontemplateinv = cv2.imread('images/lon-inv-resized.png',cv2.IMREAD_GRAYSCALE)
# lontemplateinv = imutils.rotate_bound(lontemplateinv,angle=180) 
# lontemplateinv = imutils.resize(lontemplateinv,height=int(lontemplateinv.shape[1]*0.75))
# cv2.imwrite('images/lontemplate-inv-resized.jpg',lontemplateinv)
once = 0

allOMRs= glob.iglob(directory+'*/*/*/*'+ext)
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

"""TODO NOW
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

mws, mbs = [],[]
# start=35
with open(resultFile,'a') as f:
    # for i,filepath in enumerate(list(allOMRs)[start:start+5]):
    for i,filepath in enumerate(allOMRs):
        # num = str(i).zfill(4)
    #     filename=folder+prefix+num+ext
        finder = re.search(r'/.*/(.*)/(.*)/(.*)\.'+ext[1:],filepath,re.IGNORECASE)
        if(finder):
            squadlang = finder.group(1)
            squad,lang = squadlang[0],squadlang[1]
            squadlang = squadlang+'/'
            filename = finder.group(3)
            xeno = finder.group(2)
        else:
            filename = 'Nop'+str(i)

        # temp patch
        if("HE_" in filename):squad="H";

        origOMR = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE) 
        h,w = origOMR.shape
        if(w>uniform_width*1.25 and stitched):
            print("Assuming Stitched input.")
            w=int(w/2)
            OMRs=[origOMR[:,:w],origOMR[:,w:2*w]]
            thresholdReads=[thresholdRead_L,thresholdRead_R]

        else:
            OMRs=[origOMR]
            thresholdReads=[thresholdRead_L]

        local_id=0
        for thresholdRead,OMR in zip(thresholdReads,OMRs):
            local_id+=1
            OMR = imutils.resize(OMR,height=uniform_height_hd) 
            print("Template",template.shape,"Image", OMR.shape)
            OMR = cv2.GaussianBlur(OMR,(3,3),0) 
            #>> temp
            if(showimglvl>=3):
                show('OMR',OMR,0)
            # OMR = imutils.rotate_bound(OMR,angle=90) 
            
            OMRcrop,badscan = getROI(errorsArray,squadlang,filepath,filename+ext,filename,OMR,template,pause=0,
                lontemplinv= (lontemplateinv if autorotate else None),showimglvl=showimglvl,verbose=verbose, scaleRange=scaleRange,thresholdCircle=thresholdCircle)

            #uniquify 
            newfilename = filename + '_' + filepath.split('/')[-2] 
            # if(badwidth == 1):
            #     # Shifting the template! Resizing might work!!
            #     # Put detection images into the verify folder

            if(OMRcrop is None):
                # Here badscan == -1 will come

                err = move(results_2018error,filepath,errorpath+squadlang,newfilename+'.jpg')
                if(err):
                    appendArr(err,errorsArray,ErrorFile)
                continue
            else:
                
                OMRcrop = cv2.resize(OMRcrop,(uniform_width_hd,uniform_height_hd))
                # OMRcrop = imutils.resize(OMRcrop,height=uniform_height,width=uniform_width) 

            
            respArray=[]
            try: #TODO - resolve this try catch later 
                counter+=1
                
                OMRresponse,retimg,multimarked,multiroll,mw,mb = readResponse(squad,TEMPLATES[squad],(boxDimX,boxDimY),OMRcrop,
                                                  badscan=badscan,multimarkedTHR= thresholdRead-12.75 ,name =newfilename,save=(saveMarkedDir+squadlang if saveMarked else None),thresholdRead=thresholdRead,explain=explain,bord=-1)
                mws.append(mw)                
                mbs.append(mb)                
                # print("XYZ1")
                resp=processOMR(squad,OMRresponse) #convert to ABCD, getRoll,etc
                # print("XYZ2")
                respArray.append(resp['roll']) #May append None
                for q in qNos['H']:#for align
                    try:
                        respArray.append(resp['resp'][q])
                    except:
                        respArray.append('')
                # err contains the rest = [results_2018batch,error,filename,filepath2]
                #This evaluates and Enters into Results sheet-
                score = evaluate(resp['resp'],Answers[squad+('K' if kv else '')],Sections[squad+('K' if kv else '')],explain=explain)

                if(multiroll or not (resp['roll'] is not None and len(resp['roll'])==11)):
                    #>>temp
                    pass
                    # print('badRollNo, moving File: '+newfilename)
                    # err = move(badRollError,filepath,badRollspath+squadlang,newfilename+'.jpg')
                    # if(err):
                    #     appendArr(err+respArray,badRollsArray,badRollsFile)
                else:
                        
                    if(badscan == 1):
                        # print('File Skipped from verify. Must be in Multimarked or Results')
                        err = move(verifyError,filepath,verifypath+squadlang,newfilename+'.jpg')
                        if(err):
                            appendArr(err+respArray,verifyArray,verifyFile)
                    else:
                        if(multimarked == 0):
                            #TODO check that score stays the last column !
                            # err contains the rest = [results_2018batch,error,filename,filepath2]
                            results = [0,0,newfilename+'.jpg',filepath]+respArray+[score] #.append
                            filesNotMoved+=1;
                            pd.DataFrame(results).T.to_csv(f,header=False,index=False)
                            print((counter,resp['roll'],score))
                            # print((counter,newfilename+'.jpg',resp['roll'],','.join(respArray[1:]),'score : ',score))
                        else:

                            #multimarked file
                            print('multiMarked, moving File: '+newfilename)
                            err = move(multiMarkedError,filepath,multiMarkedpath+squadlang,newfilename+'.jpg')
                            if(err):
                                appendArr(err+respArray,multiMarkedArray,multiMarkedFile)
            

                #>> temp
                if(showimglvl>=1):
                    # >> temp
                    show('processed_'+newfilename+'_'+str(local_id)+'.jpg',imutils.resize(retimg,height=int(display_height)),1, resetpos=resetpos)#0 if i<end else 1)
                    plt.close()
                    
            except Exception as inst:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(">>>OMR",counter,newfilename+'.jpg',"BAD ERROR : Moving file to errorFiles : ",inst)        
                # err = move(results_2018error,filepath,errorpath+squadlang,'debug_'+newfilename+'.jpg')
                # if(err):
                #     appendArr(err+respArray,errorsArray,ErrorFile)
            
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
for x in [badThresholds,veryBadPoints,thresholdCircles, mws, mbs]:
    if(x!=[]):
        x=pd.DataFrame(x)
        print( x.describe() )
        plt.plot(range(len(x)),x)
        plt.show()
    # else:
        # print(x)

