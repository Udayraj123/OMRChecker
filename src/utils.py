from constants import *

# In[62]:
import re
import os
import sys
import cv2
import glob
import imutils 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)

from template import *
from random import randint
from time import localtime,strftime,time
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

def appendErr(val):
    global errorsArray
    appendArr(val,errorsArray,ErrorFile)

def waitQ():
    while(cv2.waitKey(1)& 0xFF != ord('q')):pass
    cv2.destroyAllWindows()

def normalize_util(img, alpha=0, beta=255):
    return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)#, dtype=cv2.CV_32F)

# Make dark pixels darker, light ones lighter >> Square the Image
# Weird behavior with int16 - better use thresholding
# def square_norm(gray):
#     show("gray",gray,0,1)
#     gray = np.uint16(gray)
#     gray = gray*gray / 255
#     show("gray^2",gray,1,1)
#     return gray

def resize_util(img, u_width, u_height=None):
    if u_height == None:
        h,w=img.shape[:2]
        u_height = int(h*u_width/w)        
    return cv2.resize(img,(u_width,u_height))

def show(name,orig,pause=1,resize=False,resetpos=None):
    global windowX, windowY, display_width
    if(type(orig) == type(None)):
        print(name," NoneType image to show!")
        if(pause):
            cv2.destroyAllWindows()
        return
    img = resize_util(orig,display_width) if resize else orig
    h,w = img.shape[:2]
    cv2.imshow(name,img)

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
    
def putLabel(img,label,pos=(100,50),clr=(255,255,255),size=5):
    # TODO extend image using np
    img[:(pos[1]+size*2), :]= 0
    cv2.putText(img,label,pos,cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,clr,size)

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
    # minWidth = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

    # compute the height of the new image, which will be the
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # maxHeight = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

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


def findPage(image_norm):
    # Done: find ORIGIN for the quadrants
    # TODO: Get canny parameters tuned
    edge = cv2.Canny(image_norm, 135, 65)
    """
    A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels
    under the kernel is 1, otherwise it is eroded (made to zero).
    # Closing is reverse of Opening, Dilation followed by Erosion.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Close the small holes, or complete the edges in canny
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel) 

    # findContours returns outer boundaries in CW and inner boundaries in ACW order.
    cnts = imutils.grab_contours(cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
    # hullify to resolve disordered curves due to noise
    cnts = [cv2.convexHull(c) for c in cnts]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    sheet = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_PAGE_AREA: 
            continue
        peri = cv2.arcLength(c,True)
        # ez algo - https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm
        approx = cv2.approxPolyDP(c, epsilon = 0.025 * peri, closed = True)
        print("Area",cv2.contourArea(c), "Peri", peri)

        if(len(approx)==4):
            sheet = np.reshape(approx,(4,-1))
            cv2.drawContours(image_norm, [approx], -1, (0,255, 0), 2)
            break
        # box = perspective.order_points(box)
    print("Found largest quadrilateral: ", sheet)

    if sheet==[]:
        print("Error: Paper boundary not found!")
        show('Morphed Edges',closed,pause=1)  
    return sheet

def getBestMatch(image_eroded_sub, num_steps=10, iterLim=50):
    global template_eroded_sub

    # match_precision is how minutely to scan ?!
    x=[int(scaleRange[0]*match_precision),int(scaleRange[1]*match_precision)]
    if((x[1]-x[0])> iterLim*match_precision/num_steps):
        print("Too many iterations : %d, reduce scaleRange" % ((x[1]-x[0])*num_steps/match_precision) )
        return None

    h, w = template_eroded_sub.shape[:2]
    res, best_scale=None, None
    t_max = 0
    for r0 in range(x[1],x[0], -1*match_precision//num_steps): #reverse order
        s=float(r0)/match_precision
        if(s==0.0):
            continue
        templ_scaled = imutils.resize(template_eroded_sub, height = int(h*s))
        res = cv2.matchTemplate(image_eroded_sub,templ_scaled,cv2.TM_CCOEFF_NORMED)
        
        # res is the black image with white dots
        maxT = res.max()
        if(t_max < maxT):
            print('Scale: '+str(s)+', Circle Match: '+str(round(maxT*100,2))+'%')
            best_scale, t_max = s, maxT
    if(t_max < thresholdCircle):
        print("Warnning: Template matching too low!")
        show("res",res,1,0)
    print('') #close buf
    return best_scale

thresholdCircles=[]
badThresholds=[]
veryBadPoints=[]
def getROI(filepath,filename,image_norm, closeup=False):
    global template_eroded_sub, squadlang

    image_norm = resize_util(image_norm, uniform_width_hd, uniform_height_hd)
    # image_eroded_sub=image_norm-cv2.erode(image_norm,None)
    # Spread the darkness :P - Erode operation takes MIN over kernel

    image_eroded_sub = image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=5)
    """
    TODO:
    Write autorotate-
    """

    # TODO: Automate the case of close up scan(incorrect page)-
    warped_image_eroded_sub = image_eroded_sub
    warped_image_norm = image_norm
    
    if(closeup == False):
        sheet = findPage(image_norm)

        if sheet==[]:
            return None        

        # Warp layer 1
        warped_image_eroded_sub = four_point_transform(image_eroded_sub, sheet)        
        if(showimglvl>=2):
            show("page_check",image_norm, pause=False)
        warped_image_norm = four_point_transform(image_norm, sheet)

        # Resize back to uniform width
        # print(warped_image_eroded_sub.shape)
        # show("1",warped_image_eroded_sub,0)
        warped_image_eroded_sub = resize_util(warped_image_eroded_sub, uniform_width_hd, uniform_height_hd)
        warped_image_norm = resize_util(warped_image_norm, uniform_width_hd, uniform_height_hd)

    # Quads on warped image
    quads={}
    h1, w1=warped_image_eroded_sub.shape[:2]
    # midh,midw=h1//2, w1//3
    midh,midw = h1//3, w1//2
    origins=[[0,0],[midw,0],[0,midh],[midw,midh]]
    quads[0]=warped_image_eroded_sub[0:midh,0:midw];
    quads[1]=warped_image_eroded_sub[0:midh,midw:w1];
    quads[2]=warped_image_eroded_sub[midh:h1,0:midw];
    quads[3]=warped_image_eroded_sub[midh:h1,midw:w1];
        
    # Draw Quadlines
    warped_image_eroded_sub[ : , midw:midw+2] = 255
    warped_image_eroded_sub[ midh:midh+2, : ] = 255
    
    # print(warped_image_eroded_sub.shape)
    # show("2",warped_image_eroded_sub)
    
    best_scale = getBestMatch(warped_image_eroded_sub)    
    if(best_scale == None):
        # TODO: Plot and see performance of scaleRange
        print("No matchings for given scaleRange:",scaleRange)
        show('Quads',warped_image_eroded_sub)  
        err = move(results_2018error, filepath, errorpath+squadlang,filename)
        if(err):
            appendErr(err)

        return None
    
    templ = imutils.resize(template_eroded_sub, height = int(template_eroded_sub.shape[0]*best_scale))
    h,w=templ.shape[:2] 
    centres = []
    sumT, maxT = 0, 0
    for k in range(0,4):
        res = cv2.matchTemplate(quads[k],templ,cv2.TM_CCOEFF_NORMED)
        maxT = res.max()
        if(maxT < thresholdCircle):
            # Warning - code will stop in the middle. Keep Threshold low to avoid.
            print(filename,"\nError: No circle found in Quad",k+1, "maxT", maxT,"best_scale",best_scale)
            if(verbose):
                show('no_pts_'+filename,warped_image_eroded_sub,pause=0) 
                show('res_Q'+str(k),res,pause=1) 

            return None

        pt=np.argwhere(res==maxT)[0];
        pt = [pt[1],pt[0]]
        pt[0]+=origins[k][0]        
        pt[1]+=origins[k][1]
        # print(">>",pt)
        warped_image_norm = cv2.rectangle(warped_image_norm,tuple(pt),(pt[0]+w,pt[1]+h),(150,150,150),2)            
        warped_image_eroded_sub = cv2.rectangle(warped_image_eroded_sub,tuple(pt),(pt[0]+w,pt[1]+h),(150,150,150),2)            
        centres.append([pt[0]+w/2,pt[1]+h/2])
        sumT += maxT

    # analysis data
    thresholdCircles.append(sumT/4)
        
    # show('Detected circles',warped_image_norm,0)    
    warped_image_norm = four_point_transform(warped_image_norm, np.array(centres))
    
    if(showimglvl>=2):
        show('warped_image_eroded_sub',warped_image_eroded_sub,pause=0)    
        show(filename,warped_image_norm,0)    
     
    # images/OMR_Files/4137/HE/Xerox/Durgapur_HE_04_prsp_13.22_18.78_5.jpg
    finder = re.search(r'.*/.*/.*/(.*)/(.*)/(.*)\.'+ext[1:],filepath,re.IGNORECASE)
    squad,lang = 'X','X'
    if(finder):
        squadlang = finder.group(1)
        squad,lang = squadlang[0],squadlang[1]

    newfilename = filename + '_' + filepath.split('/')[-2]
    
    # iterations : Tuned to 2.
    # warped_image_eroded_sub = warped_image_norm - cv2.erode(warped_image_norm, kernel=np.ones((5,5)),iterations=2)

    OMRresponse,retimg,multimarked,multiroll = readResponse(squad,warped_image_norm, name = newfilename)
    
    return warped_image_norm


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

def getCentroid(window):
    h, w = window.shape
    ax = np.array([[x for x in range(w)]] * h)
    ay = np.array([[y]* w for y in range(h)])
    centroid = [np.average(ax,weights = window), np.average(ay,weights = window)]
    return centroid

def getThreshold(QVals):
    gap = (np.max(QVals) - np.min(QVals))
    if((np.std(QVals) < MIN_STD) and gap < MIN_GAP):
        # no marked
        return np.min(QVals)
    if(len(QVals) < 3): # for medium
        return np.min(QVals) if gap < MIN_GAP else np.mean(QVals)
    # Sort the Q vals
    QVals= sorted(QVals)

    # Find the first 'big' jump and set it as threshold:
    l=len(QVals)-1
    max1,thr1=0,255
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        if(jump > max1):
            max1=jump
            thr1=QVals[i-1] + jump/2
    
    # Make use of the fact that the JUMP_DELTA between values at detected jumps would be atleast 20
    max2,thr2=0,255
    # Requires atleast 1 gray box to be present (Roll field will ensure this)
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        d2 = QVals[i-1] + jump/2          
        if(jump > max2 and JUMP_DELTA < abs(thr1-d2)):
            max2=jump
            thr2=d2

    # Updated threshold: The 'first' jump 
    # TODO: Make this more robust
    thresholdRead = min(thr1,thr2)
    if(showimglvl>=3):
        f, ax = plt.subplots() 
        ax.bar(range(len(QVals)),QVals);
        thrline=ax.axhline(thresholdRead,color='red',ls='--')
        thrline.set_label("THR")

        ax.set_title("Intensity distribution")
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Q Boxes sorted by Intensity")
        plt.show()
    return thresholdRead

def saveImg(path, retimg):
    print('Saving Image to '+path)
    cv2.imwrite(path,retimg)
def readResponse(squad,image,name,save=None,explain=True):
    TEMPLATE = TEMPLATES[squad]
    try: 
        img = image.copy()
        print("Cropped dim", img.shape[:2])
        # 1846 x 1500
        img = resize_util(img,TEMPLATE.dims[0],TEMPLATE.dims[1])
        print("Resized dim", img.shape[:2])

        img = normalize_util(img)

        boxW,boxH = TEMPLATE.boxDims
        lang = ['E','H']
        OMRresponse={}
        CLR_GRAY = (200,150,150)

        multimarked,multiroll=0,0
        alpha=0.95
        output=img.copy()

        blackVals=[0]
        whiteVals=[255]

        if( showimglvl>=3):
            allCBoxvals={"Int":[],"Mcq":[]}#"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}
            qNums={"Int":[],"Mcq":[]}#,"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}
        

        ### Find Shifts for the QBlocks --> Before calculating threshold!
        morph = img.copy() # 
        # Open : erode then dilate!
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        morph =  255 - cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=3)
        _, morph = cv2.threshold(morph,10,255,cv2.THRESH_BINARY)
        
        # best tuned to 5x5 now
        # show("open_inv_thr", morph, 0, 1)
        morph = cv2.erode(morph,  np.ones((5,5),np.uint8), iterations = 2)

        alpha1 = 0.55
        overlay = cv2.addWeighted(255 - morph,alpha1,output,1-alpha1,0,output)    
        retimg = img.copy()
        # retimg = overlay.copy()
        # show("Morph Overlay", overlay, 0, 1)
        
        initial=img.copy()
        dims = TEMPLATE.boxDims
        for QBlock in TEMPLATE.QBlocks:
            for col, pts in QBlock.colpts:
                for pt in pts:
                    cv2.rectangle(initial,(pt.x,pt.y),(pt.x+dims[0],pt.y+dims[1]),CLR_GRAY,-1)

        # sq = np.uint16(morph)
        # sq = sq*sq / 255
        # show("sq",sq,0,1)
        # gray = abs(gray) #for Sobel
        



# templ adjust code
        THK = 0 # acc to morph kernel
        ini_templ=overlay.copy()
        for QBlock in TEMPLATE.QBlocks:
            n, s,d = QBlock.key, QBlock.orig, QBlock.dims
            shift = 0
            cv2.rectangle(ini_templ,(s[0]+shift-THK,s[1]-THK),(s[0]+shift+d[0]+THK,s[1]+d[1]+THK),(10,10,10),3)
            # cv2.rectangle(ini_templ,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),(0,0,1),3)

        for QBlock in TEMPLATE.QBlocks:
            n, s,d = QBlock.key, QBlock.orig, QBlock.dims
            ALIGN_STRIDE, MATCH_COL, ALIGN_STEPS = 1, 5, int(boxW * 2 / 3)
            shiftM, shift, steps = 0, 0, 0
            THK = 3
            while steps < ALIGN_STEPS:
                L = np.mean(morph[s[1]:s[1]+d[1],s[0]+shift-THK:-THK+s[0]+shift+MATCH_COL])
                R = np.mean(morph[s[1]:s[1]+d[1],s[0]+shift-MATCH_COL+d[0]+THK:THK+s[0]+shift+d[0]])
                # print(shift, L, R)
                LW,RW= L > 100, R > 100
                if(LW):
                    if(RW):
                        shiftM = shift
                        break
                    else:
                        shift -= ALIGN_STRIDE
                else:
                    if(RW):
                        shift += ALIGN_STRIDE
                    else:
                        shiftM = shift
                        break
                steps += 1

            QBlock.shift = shiftM
            # sums = sorted(sums, reverse=True)
            # print("Aligned QBlock: ",QBlock.key,"Corrected Shift:", QBlock.shift,", Dimensions:", QBlock.dims, "orig:", QBlock.orig,'\n')

        corr_templ=overlay.copy()
        for QBlock in TEMPLATE.QBlocks:
            n, s,d = QBlock.key, QBlock.orig, QBlock.dims
            shift = QBlock.shift
            cv2.rectangle(corr_templ,(s[0]+shift-THK,s[1]-THK),(s[0]+shift+d[0]+THK,s[1]+d[1]+THK),(10,10,10),3)
            # cv2.rectangle(corr_templ,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),(0,0,1),3)
        # show("Initial Template Overlay", 255 -  ini_templ, 0, 1, [0,0])
        # show("Corrected Template Overlay", 255 -  corr_templ, 1, 1)# [corr_templ.shape[1],0])

        thresholdReadAvg, ncols = 0, 0

        for QBlock in TEMPLATE.QBlocks:
            colNo = 0
            key = QBlock.key[:3]
            cv2.putText(retimg,'s%s'% (QBlock.shift), tuple(QBlock.orig - [75,-QBlock.dims[1]//2]),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(50,20,10),4)
            for col, pts in QBlock.colpts:
                colNo += 1
                o,e = col
                CBoxvals=[]
                for pt in pts:
                    x,y =(pt.x + QBlock.shift,pt.y)
                    rect = [y,y+boxH,x,x+boxW]
                    CBoxvals.append(cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0])                
                thresholdRead = getThreshold(CBoxvals)
                thresholdReadAvg += thresholdRead
                CBoxvals=[]
                for pt in pts:
                    # shifted
                    x,y = (pt.x + QBlock.shift,pt.y)
                    check_rects = [[y,y+boxH,x,x+boxW]]
                    detected=False
                    boxval0 = 0
                    for rect in check_rects:
                        # This is NOT the usual thresholding, It is boxed mean-thresholding
                        boxval = cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0]
                        if(boxval0 == 0):
                            boxval0 = boxval
                        if(thresholdRead > boxval):
                            # for critical analysis
                            boxval0 = max(boxval,boxval0)
                            detected=True
                            break;
                    cv2.rectangle(retimg,(x,y),(x+boxW,y+boxH),(150,150,150) if detected else CLR_GRAY,-1)

                    #for hist
                    CBoxvals.append(boxval0)
                    if (detected):
                        q = pt.qNo
                        val = str(pt.val)
                        cv2.putText(retimg,val,(x,y),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(50,20,10),5)
                        # Only send rolls multi-marked in the directory
                        multimarked = q in OMRresponse
                        OMRresponse[q] = (OMRresponse[q] + val) if multimarked else val
                        multiroll = multimarked and 'roll' in str(q)
                        blackVals.append(boxval0)
                    else:
                        whiteVals.append(boxval0)     
                    # /for col
                if( showimglvl>=3):
                    if(key in allCBoxvals):
                        qNums[key].append(key[:2]+'_c'+str(colNo))
                        allCBoxvals[key].append(CBoxvals)
            ncols += colNo
            # /for QBlock        
        thresholdReadAvg /= ncols
        # Translucent
        retimg = cv2.addWeighted(retimg,alpha,output,1-alpha,0,output)    
            
        if( showimglvl>=3):
            # plt.draw()
            f, axes = plt.subplots(len(allCBoxvals),sharey=True)
            f.canvas.set_window_title(name)
            ctr=0
            typeName={"Int":"Integer","Mcq":"MCQ","Med":"MED","Rol":"ROLL"}
            for k,boxvals in allCBoxvals.items():
                axes[ctr].title.set_text(typeName[k]+" Type")
                axes[ctr].boxplot(boxvals)
                # thrline=axes[ctr].axhline(thresholdReadAvg,color='red',ls='--')
                # thrline.set_label("Average THR")
                axes[ctr].set_ylabel("Intensity")
                axes[ctr].set_xticklabels(qNums[k])
                # axes[ctr].legend()
                ctr+=1
            # imshow will do the waiting
            plt.tight_layout(pad=0.5)
            
            if(showimglvl>=2):
                plt.show() 

        if ( type(save) != type(None) ):
            saveImg(save+('_Multi_/' if multiroll else '')+name+'_marked.jpg', retimg)
        
        show("Initial Template", initial,0,1, [0,0])
        show("Corrected",retimg,1,1)

        return OMRresponse,retimg,multimarked,multiroll

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
                

########### Alignment Code dump ########### 

# if(QBlock.shift != 0):
# maxS, shiftM = 0, 0
# sums = []
# for shift in ALIGN_RANGE:
#     # take QVals wise sum of means
#     sm = 0
#     gray2=morph.copy()
#     for col in QBlock.cols:
#         o,e = col
#         # shifted
#         o[0] += shift 
#         e[0] += shift + THK
#         window = morph[ o[1]:e[1] , o[0]:e[0] ]
#         sm += cv2.mean(window)[0]
#         cv2.rectangle(gray2,(o[0],o[1]),(e[0],e[1]),(0,0,1),3)
#     sums.append(sm)
#     # Logs
#     print(shift, sm)                
#     show("Image", gray2, 1, 1, [0,0])

#     if(maxS < sm):
#         maxS = sm
#         shiftM = shift

# centre=(d[0]/2, d[1]/2)
# closestGap, shiftM = d[0], 0
# equals = []
# for shift in ALIGN_RANGE:
#     window = morph[s[1]-THK:s[1]+THK+d[1],s[0]-THK+shift:s[0]+THK+shift+d[0]]
#     show("window", window, 1, 0, [0,0])
#     centroid = getCentroid(window) # quite slow
#     #  Nope, unpredictable on bad data - Decide whether to move towards, and how fast
#     print(QBlock.key, shift, centre[0], '->', round(centroid[0],2))
#     if(closestGap >= abs(centroid[0] - centre[0])):
#         closestGap = abs(centroid[0] - centre[0])
#         if(int(closestGap)==0):
#             equals.append(shift)
#         closest = shift

# shiftM = equals[(len(equals)-1)//2] if (closestGap==0) else closest            


# Supplimentary points
# xminus,xplus= x-int(boxW/2),x+boxW-int(boxW/2) 
# xminus2,xplus2= x+int(boxW/2),x+boxW+int(boxW/2) 
# yminus,yplus= y-int(boxH/2.7),y+boxH-int(boxH/2.7) 
# yminus2,yplus2= y+int(boxH/2.7),y+boxH+int(boxH/2.7) 

#  This Plus method is better than having bigger box as gray would also get marked otherwise. Bigger box is rather an invalid alternative.
# check_rects = [ 
#     [y,y+boxH,x,x+boxW], 
#     # [yminus,yplus,x,x+boxW], 
#     # [yminus2,yplus2,x,x+boxW], 
#     # [y,y+boxH,xminus,xplus], 
#     # [y,y+boxH,xminus2,xplus2], 
# ]


# print('Keep THR between : ',,np.mean(whiteVals))
# global maxBlackTHR,minWhiteTHR
# maxBlackTHR = max(maxBlackTHR,np.max(blackVals))
# minWhiteTHR = min(minWhiteTHR,np.min(whiteVals))

## Real helping stats:
# cv2.putText(retimg,"avg: "+str(["avgBlack: "+str(round(np.mean(blackVals),2)),"avgWhite: "+str(round(np.mean(whiteVals),2))]),(20,50),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(50,20,10),4)
# cv2.putText(retimg,"ext: "+str(["maxBlack: "+str(round(np.max(blackVals),2)),"minW(gray): "+str(round(np.min(whiteVals),2))]),(20,90),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(50,20,10),4)

#     plt.draw() 
#     plt.pause(0.01)

# svals = []
# gaps = []
# gap = round(np.max(CBoxvals) - np.min(CBoxvals))
# # print(QBlock.key, colNo, 'gap', gap, '\tmean', round(np.mean(CBoxvals)), '\tstd', round(np.std(CBoxvals)))
# svals.append(round(np.std(CBoxvals)))                
# gaps.append(gap)
# f, ax = plt.subplots() 
# x = np.array(range(len(svals)))
# ax.bar(x-0.4,svals, width=0.4, color='b');
# ax.bar(x,gaps, width=0.4, color='g');
# ax.set_title("Column-wise stddev/gap distribution")
# ax.set_ylabel("stddev/gap")
# ax.set_xlabel("col")
# thrline=ax.axhline(MIN_STD,color='red',ls='--')
# thrline.set_label("MIN_STD")
# plt.show()