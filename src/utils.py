from constants import *

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

print('Checking Directories...')
for dir in [saveMarkedDir]:
    if(not os.path.exists(dir)):
        print('Created : '+ dir)
        os.mkdir(dir)
        for sl in ['HE','HH','JE','JH']:
            os.mkdir(dir+sl)
            os.mkdir(dir+sl+'/_MULTI_')
            os.mkdir(dir+sl+'/_BadScan_')
    else:
        print('Already present : '+dir)

for dir in ['feedsheets','results']:
    if(not os.path.exists(dir)):
            print('Created : '+ dir)
            os.mkdir(dir)
    else:
        print('Already present : '+dir)
for dir in Directories:
    if(not os.path.exists(dir)):
        print('Created : '+ dir)
        os.mkdir(dir)
        for sl in ['HE','HH','JE','JH']:
            os.mkdir(dir+sl)
    else:
        print('Already present : '+dir)


# In[64]:

def pad(val,array):
    if(len(val) < len(array)):
        for i in range(len(array)-len(val)):
            val.append('V')

def appendArr(val,array,filename):
    array.append(val)
    if(not os.path.exists(filename)):
        with open(filename,'a') as f:
            pd.DataFrame([sheetCols],columns=sheetCols).to_csv(f,index=False,header=False)
    pad(val,sheetCols)
    with open(filename,'a') as f:
        pd.DataFrame([val],columns=sheetCols).to_csv(f,index=False,header=False)
        # pd.DataFrame(val).T.to_csv(f,header=False)


def waitQ():
    while(cv2.waitKey(1)& 0xFF != ord('q')):pass
    cv2.destroyAllWindows()

def show(name,img,pause=1,resetpos=None):
    global windowX
    global windowY
    if(type(img)== type(None)):
        print(name," NoneType image to show!")
        if(pause):
            cv2.destroyAllWindows()
        return
    cv2.imshow(name,img)
    
    w,h = img.shape
    if(resetpos):
        windowX=resetpos[0]
        windowY=resetpos[1]
    cv2.moveWindow(name,windowX,windowY)
    overflowY = windowY+h > windowHeight
    if(windowX+w > windowWidth):
        windowX = 0
        if(not overflowY):windowY+=h
    else:windowX+=w
    if(overflowY):windowY = 0
    if(pause):
        waitQ()
    
def myColor1():
    return (randint(100,250),randint(100,250),randint(100,250))
    
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    minWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [minWidth - 1, 0],
        [minWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (minWidth, maxHeight))

    # return the warped image
    return warped
    


# In[4]:


# In[65]:
# a=[0.5,1.5,0,1,3,5,6]
# map(int,np.multiply(a,10))
# a = [a,a]
# pts = [[0,1],[1,0],[1,5],[0,5]]
# map(lambda x:a[x[0]][x[1]],sorted(pts,key=lambda pt: a[pt[0]][pt[1]],reverse=True))


# In[5]:


def dist(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))
def getslope(pt1,pt2):
    return float(pt2[1]-pt1[1])/float(pt2[0]-pt1[0])
def check_min_dist(pt,pts,min_dist):
    for p in pts:
        if(dist(pt,p) < min_dist):
            return False
    return True
        
filesMoved=0
filesNotMoved=0
def move(error,filepath,filepath2,filename):
    # print("Error-Code: "+str(error))
    # print("Source:  "+filepath)
    # print("Destination: " + filepath2 + filename)
    global filesMoved
    # print(filepath,filepath2,filename,array)
    if(os.path.exists(filepath)):
        if(os.path.exists(filepath2+filename)):
            print('ERROR : Duplicate file at '+filepath2+filename)
        os.rename(filepath,filepath2+filename)
        append = [results_2018batch,error,filename,filepath2]
        filesMoved+=1
        return append
    else:
        print('File already moved')
        return None
        
def get_reflection(pt, pt1,pt2):
    pt, pt1,pt2 = tuple(map(lambda x:np.array(x,dtype=float),[pt, pt1,pt2]))
    return (pt1 + pt2) - pt
def printbuf(x):
    sys.stdout.write(str(x))
    sys.stdout.write('\r')


# In[8]:


# In[ ]:

thresholdCircles=[]
badThresholds=[]
veryBadPoints=[]
def match_template_scaled(errorsArray,squadlang,filepath,filename,img1, template,showimglvl=0,pts=4,
                          scaleRange=(0.5,1.5),fac=50,min_dist=None,thresholdCircle=0.6,append=1,
                          CLR=(255,155,255),iterLim=30,excludepts=None):
    global errorpath
    lon='lon' if append==0 else 'non lon'
    orig=template.copy()
    w1,h1=img1.shape
    w,h=template.shape
    if( not min_dist):
        min_dist = h1/10
    scale=float(h1)/h
    # print('image to template ratio (should be 38)',scale)

    x=list(map(int,np.multiply(scaleRange,fac)))
    if((x[1]-x[0])> iterLim*fac/10):
        print("Too many iterations : %d, reduce scaleRange" % ((x[1]-x[0])*10/fac) )
        return []
        
###### vvv Scope for improvement
    r_max=None
    for r0 in range(x[1],x[0], -1*fac//10): #reverse order
        r=float(r0)/fac
        # printbuf(r)
        if(r==0.0):
            continue
        templ = imutils.resize(orig, height = int(h*r))
        res = cv2.matchTemplate(img1,templ,cv2.TM_CCOEFF_NORMED)
        # res is the black image with white dots
        maxT = res.max()
        if(thresholdCircle < maxT):
            print(r,' %d )Better match %d%%' % (r0,100*maxT))
            r_max=r
            thresholdCircle = maxT
###### ^^^ Scope for improvement

    if(r_max==None):
        if(append):
            print("No matchings for given scaleRange & thresholdCircle",scaleRange,thresholdCircle,lon)
        
        print("No matchingsERRRRR")
        # this goes as resp=None in the main.
        err = move(results_2018error,filepath,errorpath+squadlang,filename)
        if(append and err):
            appendArr(err,errorsArray,ErrorFile)

        return [],0
    print('')
    if(r_max not in (0.85,0.9,0.95)):
        print('WARNING : Changed final scale',r_max,lon)
    
    templ = imutils.resize(orig, height = int(h*r_max))
    
    #make quadrants of image
    quads={}
    origins=[[0,0],[w1//3,0],[0,h1//2],[w1//3,h1//2]]
    quads[1-1]=img1[0:h1//2,0:w1//3];
    quads[2-1]=img1[0:h1//2,w1//3:w1];
    quads[3-1]=img1[h1//2:h1,0:w1//3];
    quads[4-1]=img1[h1//2:h1,w1//3:w1];
    locd = []
    for k in range(0,4):
        img=quads[k]
        res = cv2.matchTemplate(img,templ,cv2.TM_CCOEFF_NORMED)
        maxT = res.max()
        if(maxT > thresholdCircle):
            print("Updated Threshold: ",thresholdCircle)
            thresholdCircle=maxT
        #max(locs,key=lambda pt: res[pt[1]][pt[0]]) 
        pt=np.argwhere(res==maxT)[0];
        pt = [pt[1],pt[0]]
        if(k==0):thresholdCircles.append(maxT)
        pt[0]+=origins[k][0]        
        pt[1]+=origins[k][1]
        # print(">>",pt)
        locd.append(pt)
    
    w,h=templ.shape 
    centres=[]
    for pt in locd:
        img1 = cv2.rectangle(img1,tuple(pt),(pt[0]+w,pt[1]+h),CLR,2)            
        centres.append([pt[0]+w/2,pt[1]+h/2])
        
##############################################################################
    # exclude = type(excludepts)!=type(None) and excludepts!=[]
    badscan = 0
    # for pt in locs: #zipped because it gives Xs and Ys seperated & probab ordered by Xs
    #     if(i==pts):
    #         break
    #     if(check_min_dist(pt,centres,min_dist) and ( not exclude or (exclude and check_min_dist(pt,excludepts,min_dist)))):
    #         # print('adding pt ',pt,check_min_dist(pt,centres,min_dist),exclude,check_min_dist(pt,excludepts,min_dist))
    #         img1 = cv2.rectangle(img1,pt,(pt[0]+w,pt[1]+h),CLR,2)            
    #         centres.append([pt[0]+w/2,pt[1]+h/2])
    #         thresholdCircles.append(res[pt[1]][pt[0]])            
    #         i+=1
    #     else:
    #         badThresholds.append(res[pt[1]][pt[0]])
    #         print(locs,'WARNING Bad Scan : skipped nearby point', pt,res[pt[1]][pt[0]],'min_dist:',min_dist,centres,exclude,excludepts)
    #         badscan=1
    # ### ^^^        
        
    if(showimglvl>=2):
        show('detected',img1,0)    

    # if(len(centres)<3):
    #     for pt in centres:

    #         veryBadPoints.append(res[int(pt[1])][int(pt[0])]);

    return np.array(centres),badscan

def get_fourth_pt(three_pts):
    m=[]
    for i in range(3):
        m.append(dist(three_pts[i],three_pts[(i+1)%3]))
        
    v =max(m)
    for i in range(3):
        if(m[i]!=v and m[(i+1)%3]!=v):
            refl = (i+1) % 3
            break
    fourth_pt = get_reflection( three_pts[refl],three_pts[(refl+1)%3],three_pts[(refl+2)%3])
    return fourth_pt


# In[7]:


# The Wrapper
def getROI(errorsArray,squadlang,filepath,filename,name,orig,templ,pause=0,lontemplinv=None,showimglvl=0,verbose=False,pts=4,scaleRange=(0.25,1.5),fac=100,
           thresholdCircle=0.55,thresholdLon=0.55,iterLim=50):
    template=templ.copy()
    image=orig.copy()
    #Add rotation here
    w,h=image.shape
    image=image-cv2.erode(image,None)
    template=template-cv2.erode(template,None)
    
    excludepts=None
    #Manually adding the lon point
    # excludepts = [[380,60]]
##
    """
    Done> Smoothening/Blurring for better template match!!
    TODO:
    debug this autorotate-
    """

    # if( type(lontemplinv) != type(None)):
    #     lontemplateinv=lontemplinv.copy()
    #     lontemplateinv=lontemplateinv-cv2.erode(lontemplateinv,None)
    #     # show('lontemplateinv',lontemplateinv)
    #     p=float(w)/float(h)
    #     if(p<1):
    #         #90 or 180 works fine
    #         # print('Initiate Rotate by 90 - ')
    #         lontemplateinv = imutils.rotate_bound(lontemplateinv,angle= 270)
            
    #     excludepts,_ = match_template_scaled(errorsArray,squadlang,filepath,filename,image,lontemplateinv,
    #                                     showimglvl=showimglvl,
    #                                     pts=1,
    #                                     append=0,
    #                                     scaleRange=scaleRange,fac=fac,thresholdCircle=thresholdLon,iterLim=iterLim)
        
    #     # print("Top location : ",excludepts)
    #     # show('lontemplateinv',lontemplateinv,0)

    #     if(excludepts!=[]):
    #         angle = 180 if excludepts[0][1] > 500 else ( 270 if excludepts[0][0] > 500 else 90)
    #         origin = h,w if angle == 180 else (h,0 if angle== 270 else (0,w if angle==90 else 0,0))
    #         # print(excludepts[0],origin,image.shape)
    #         #transpose of the point
    #         excludepts[0] = ( excludepts[0][1]-origin[1],origin[0] -excludepts[0][0]    )
    #         # pt = tuple(excludepts[0])
    #         white=(250,250,250)
    #         print('Warning: Rotating File by : '+str(angle))
    #         image = imutils.rotate_bound(image,angle=angle)
    #         # image = cv2.rectangle(image,pt,(pt[0]+10,pt[1]+10),white,-1)
    #         orig = imutils.rotate_bound(orig,angle=angle)
        
    # cv2.createTrackbar('boxDim', 'ImageWindow', 2000, 5000, match_template_scaled_with_time_flag)
    

    four_pts,badscan =match_template_scaled(errorsArray,squadlang,filepath,filename,image,template,
                                showimglvl=showimglvl,
                                pts=pts,
                                scaleRange=scaleRange,fac=fac,thresholdCircle=thresholdCircle,iterLim=iterLim,excludepts=excludepts)
    if(pts == 3 and len(four_pts)==3):
        three_pts=four_pts
        four_pts = np.concatenate([four_pts,[get_fourth_pt(three_pts)]])

    if(verbose):
        print('verbose: ',name,'4 circles : ',list(four_pts))
        
    if(len(four_pts)>=4):
        warped = four_point_transform(orig, four_pts)        
            #########################################################################################################
        # show(name,warped,pause=pause)
        return warped,badscan 
    else:
        print(name,"Unable to find enough points!")
        if(verbose):
            #########################################################################################################
            show('verify_'+name,orig,pause=1) #WARNING - CODE WILL STOP IN THE MIDDLE.
        return None,-1

# Toolbox functions-
# cv2.createTrackbar('boxDim', 'ImageWindow', 2000, 5000, someFunctionCallBack)


# In[8]:


def addInnerKey(dic,key1,key2,val):
#Overwrites
    try:
        #add key
        dic[key1][key2] = val
    except:
        #first key
        dic[key1] = {key2: val}

def checkKey(OMRresponse,key1,key2):
    try:
        temp = OMRresponse[key1][key2]
        return True
    except:
        return False

def readResponse(squad,TEMPLATE,boxDim,image,name,save=None,thresholdRead=127.5,explain=True,bord=-1,
    white=(200,150,150),black=(25,120,20),badscan=0,multimarkedTHR=153,isint=True):
    try: 
        img = image.copy()
        _, t = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
        w,h=boxDim
        mask = 255*np.ones(boxDim, np.uint8)
        lang = ['E','H']
        OMRresponse={}
        grey,skyblue=(0,0,0),(200,150,150)
        clrs=[skyblue,grey]
        clrs=[grey,skyblue]

        multimarked,multiroll=0,0
        alpha=0.65
        output=img.copy()
        retimg=img.copy()
        blackTHRs=[0]
        whiteTHRs=[255]

        if(showimglvl>=1):
            allQboxvals={QTYPE_INT:[],QTYPE_MCQ:[]}#QTYPE_ROLL:[]}#,QTYPE_MED:[]}
            qNums={QTYPE_INT:[],QTYPE_MCQ:[]}#,QTYPE_ROLL:[]}#,QTYPE_MED:[]}
            # f, axes = plt.subplots(len(TEMPLATE)//2,sharey=True, sharex=True)
            # f.canvas.set_window_title(name)
            # f.suptitle("Questionwise Histogram")
        # (1500, 1846)
        print("Cropped dim", img.shape)
        for ctr,Que in enumerate(TEMPLATE):
            Qboxvals=[]
            for pt in Que.pts:
                ptXY=(pt.x,pt.y)
                x,y=ptXY
                #Done - Here add the scan region for (y:y+h,x:x+w)
                xminus,xplus= x-int(w/2),x+w-int(w/2) #supplementary point.
                xminus2,xplus2= x+int(w/2),x+w+int(w/2) #supplementary point.
                yminus,yplus= y-int(h/2),y+h-int(h/2) #supplementary point.
                yminus2,yplus2= y+int(h/3),y+h+int(h/3) #supplementary point.
                # print(img.shape, xminus,xplus,y,y+h)
                mean_color = cv2.mean(img[  y : y+h,x : x+w   ],mask)
                mean_color2 = cv2.mean(img[ y : y+h,xminus : xplus    ],mask)
                mean_color3 = cv2.mean(img[ y : y+h,xminus2 : xplus2  ],mask)
                mean_color4 = cv2.mean(img[ yminus : yplus,x : x+w     ],mask)
                mean_color5 = cv2.mean(img[ yminus2 : yplus2,x : x+w     ],mask)
                boxval = mean_color[0]            
                #for hist
                Qboxvals.append(boxval)

                detected1=(thresholdRead > boxval)
                threshold2 = thresholdRead + 38.25 if detected1 else thresholdRead
                detected2=(threshold2 > mean_color2[0])
                threshold3 = thresholdRead + 38.25 if (detected2 or detected1) else thresholdRead
                detected3=(threshold3 > mean_color3[0])
                threshold4 = thresholdRead + 38.25 if (detected3 or detected2 or detected1) else thresholdRead
                detected4=(threshold4 > mean_color4[0])
                threshold5 = thresholdRead + 38.25 if (detected4 or detected3 or detected2 or detected1) else thresholdRead
                detected5=(threshold5 > mean_color5[0])


                # TODO: fix this- all detecteds should be considered
                if(detected1):
                    blackTHRs.append(boxval)
                else:
                    whiteTHRs.append(boxval)
                    
                clr= black if detected1 else white

                # retimg = cv2.rectangle(retimg,(x,yminus2),(x+w,yplus2),clrs[int(detected5)],bord)#-1 is for fill
                # retimg = cv2.rectangle(retimg,(x,yminus),(x+w,yplus),clrs[int(detected4)],bord)#-1 is for fill
                # retimg = cv2.rectangle(retimg,(xminus2,y),(xplus2,y+h),clrs[int(detected3)],bord)#-1 is for fill
                # retimg = cv2.rectangle(retimg,(xminus,y),(xplus,y+h),clrs[int(detected2)],bord)#-1 is for fill

                retimg = cv2.rectangle(retimg,(x,y),(x+w,y+h),clr,bord)#-1 is for fill

                if (detected1 or detected2 or detected3 or detected4 or detected5):
        #             try:
                    q = Que.qNo
                    val = pt.val
                    if(Que.qType==QTYPE_ROLL):
                        key1,key2 = 'Roll',q[1:] #'r1'
                    elif(Que.qType==QTYPE_MED):
                        key1,key2 = q,q
                    elif(Que.qType==QTYPE_INT):
                        key1,key2= 'INT'+ q[:-2],q[-2:]
                    else:
                        key1,key2= 'MCQ'+str(q),'val'

        #             reject qs with duplicate marking here
                    multiple = checkKey(OMRresponse,key1,key2)
                    if(multiple):
                        if('Roll' in str(q)):
                            multiroll=1
                            multimarked=1 # Only send rolls multi-marked in the directory
                            printbuf("Multimarked In Roll")

                        if(thresholdRead>multimarkedTHR): #observation
                            #This is just for those Dark OMRs
                            multimarked=1 # that its not marked by user, but code is detecting it.
                    
                    addInnerKey(OMRresponse,key1,key2,val)
                    
                    cv2.putText(retimg,str(OMRresponse[key1][key2]),ptXY,cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE,(50,20,10),5)
                    
                    if(np.random.randint(0,10)==0):
                        cv2.putText(retimg,"<"+str(int(boxval))+">",(x-2*w,y+h),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE/2,(10,10,10),3)
                                 
        #             except:
        #                 #No dict key for that point
        #                 print(pt,'This shouldnt print after debugs')
                # endif detected
                elif(np.random.randint(0,20)==0):
                    cv2.putText(retimg,"<"+str(int(boxval))+">",(x-w//2,y+h//2),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE/2,(10,10,10),3)
            
            if(showimglvl>=1):
                # For int types:
                # axes[ctr//2].hist(Qboxvals, bins=range(0,256,16))
                # axes[ctr//2].set_ylabel(Que.qNo[:-2])
                # axes[ctr//2].legend(["D1","D2"],prop={"size":6})
                if(Que.qType == QTYPE_INT or Que.qType == QTYPE_MCQ):
                    qNums[Que.qType].append(Que.qNo)
                    allQboxvals[Que.qType].append(Qboxvals)
        if(showimglvl>=1):
            # plt.draw()
            f, axes = plt.subplots(len(allQboxvals),sharey=True)
            f.canvas.set_window_title(name)
            ctr=0
            for k,boxvals in allQboxvals.items():
                axes[ctr].title.set_text(typeName[k])
                axes[ctr].boxplot(boxvals)
                thrline=axes[ctr].axhline(thresholdRead,color='red',ls='--')
                thrline.set_label("THR")
                axes[ctr].set_ylabel("Intensity")
                axes[ctr].set_xticklabels(qNums[k])
                # axes[ctr].legend()
                ctr+=1
            # imshow will do the waiting
            plt.tight_layout(pad=0.5)
            
            if(showimglvl>=2):
                plt.show() 
            else:
                plt.draw() 
                plt.pause(0.01)

        # Translucent
        retimg = cv2.addWeighted(retimg,alpha,output,1-alpha,0,output)    
        # print('Keep THR between : ',,np.mean(whiteTHRs))
        global maxBlackTHR,minWhiteTHR
        maxBlackTHR = max(maxBlackTHR,np.max(blackTHRs))
        minWhiteTHR = min(minWhiteTHR,np.min(whiteTHRs))
        
        ## Real helping stats:
        cv2.putText(retimg,"avg: "+str(["avgBlack: "+str(round(np.mean(blackTHRs),2)),"avgWhite: "+str(round(np.mean(whiteTHRs),2))]),(20,50),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE,(50,20,10),4)
        cv2.putText(retimg,"ext: "+str(["maxBlack: "+str(round(np.max(blackTHRs),2)),"minW(gray): "+str(round(np.min(whiteTHRs),2))]),(20,90),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE,(50,20,10),4)
        
        if(retimg.shape[1] > 4 + uniform_width_hd): #observation
            cv2.putText(retimg,str(retimg.shape[1]),(50,80),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE,(50,20,10),3)
            # badwidth = 1

        tosave = ( type(save) != type(None))
        if(tosave):
            #ALL SHOULD GET SAVED IN MARKED FOLDER
            if(badscan != 0):
                print('BadScan Saving Image to '+save+'_BadScan_/'+name+'_marked'+'.jpg')
                cv2.imwrite(save+'_BadScan_/'+name+'_marked'+'.jpg',retimg)
            elif(multimarked):
                print('Saving Image to '+save+'_MULTI_/'+name+'_marked'+'.jpg')
                cv2.imwrite(save+'_MULTI_/'+name+'_marked'+'.jpg',retimg)
            else:
                cv2.imwrite(save+name+'_marked'+'.jpg',retimg)

        return OMRresponse,retimg,multimarked,multiroll,minWhiteTHR,maxBlackTHR
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
# In[9]:
