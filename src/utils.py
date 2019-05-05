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

def square_util(gray):
    gray = np.uint16(gray)
    gray **= 2
    gray = gray / 255
    return gray

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
    cv2.putText(img,label,pos,cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE,clr,size)

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

    OMRresponse,retimg,multimarked,multiroll,mw,mb = readResponse(squad,warped_image_norm, name = newfilename)
    show("Corrected Image",retimg,1,1)
    
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

def readResponse(squad,image,name,save=None,thresholdRead=127.5,explain=True,bord=-1,
    white=(200,150,150),black=(25,120,20),badscan=0,multimarkedTHR=153,isint=True):
    TEMPLATE = TEMPLATES[squad]
    try: 
        img = image.copy()
        print("Cropped dim", img.shape[:2])
        # 1846 x 1500
        img = resize_util(img,TEMPLATE.dims[0],TEMPLATE.dims[1])
        print("Resized dim", img.shape[:2])

        img = normalize_util(img)
        # print("m1",img.min())
        # print("m2",img[100:-100,100:-100].min())

        # Our reading is not exactly thresholding
        # _, t = cv2.threshold(img,100,255,cv2.THRESH_BINARY)

        w,h = TEMPLATE.boxDims
        lang = ['E','H']
        OMRresponse={}
        black,grey = (0,0,0),(200,150,150)
        clrs=[grey,black]

        multimarked,multiroll=0,0
        alpha=0.65
        output=img.copy()
        retimg=img.copy()
        blackTHRs=[0]
        whiteTHRs=[255]

        if(showimglvl>=1):
            allQboxvals={"QTYPE_INT":[],"QTYPE_MCQ":[]}#"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}
            qNums={"QTYPE_INT":[],"QTYPE_MCQ":[]}#,"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}
            # f, axes = plt.subplots(len(TEMPLATE.Qs)//2,sharey=True, sharex=True)
            # f.canvas.set_window_title(name)
            # f.suptitle("Questionwise Histogram")
        
        initial=img.copy()
        dims = TEMPLATE.boxDims
        for QBlock in TEMPLATE.QBlocks:
            for Que in QBlock.Qs:
                for pt in Que.pts:
                    cv2.rectangle(initial,(pt.x,pt.y),(pt.x+dims[0],pt.y+dims[1]),grey,-1)
        show("Initial template", initial,0,1)
        
        # For threshold finding
        QVals=[]

        ### Find Shifts for the QBlocks --> Before calculating threshold!
        gray = img.copy() # 
        # Open : erode then dilate!
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
        THK = 2 # acc to kernel
        gray =  255 - cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=3)
        # Make dark pixels darker, light ones lighter >> Square the Image
        gray = square_util(gray)
        show("morph_open_inv_sq", gray, 0, 1)
        # gray = abs(gray) #for Sobel



# templ adjust code
        # gray2=gray.copy()
        # for QBlock in TEMPLATE.QBlocks:
        #     n, s,d = QBlock.key, QBlock.orig, QBlock.dims
        #     shift = 0
        #     # cv2.rectangle(gray2,(s[0]+shift-THK,s[1]-THK),(s[0]+shift+d[0]+THK,s[1]+d[1]+THK),(100,100,100),3 + THK)
        #     cv2.rectangle(gray2,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),(0,0,1),3)
        # show("Template Overlay", gray2, 1, 1, [0,0])

        for QBlock in TEMPLATE.QBlocks:
            s,d = QBlock.orig, QBlock.dims
            print("Aligning QBlock: ",QBlock.key, ", Dimensions:", QBlock.dims, "orig:", QBlock.orig)
            # TODO: save time for sum in original implementation
            maxS, shiftM = 0, 0
            # TODO: Tune factor of scan
            for shift in ALIGN_RANGE:
                # take QVals wise sum of means
                sm = 0
                # gray2=gray.copy()
                for col in QBlock.cols:
                    o,e = col
                    # shifted
                    o[0] += shift 
                    e[0] += shift + THK
                    window = gray[ o[1]:e[1] , o[0]:e[0] ]
                    sm += cv2.mean(window)[0]
                    # cv2.rectangle(gray2,(o[0],o[1]),(e[0],e[1]),(0,0,1),3)

                # Logs
                # print(shift, sm)                
                # show("Image", gray2, 1, 1, [0,0])

                if(maxS < sm):
                    maxS = sm
                    shiftM = shift
            QBlock.shift = shiftM
            # if(QBlock.shift != 0):
            print("Note: QBlock alignment corrected","Shift:", QBlock.shift)

            for Que in QBlock.Qs:
                for pt in Que.pts:
                    # shifted
                    QVals.append( cv2.mean( img[  pt.y:pt.y+h, pt.x+shiftM:pt.x+shiftM+w ] )[0])
        
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
        if(showimglvl>=4):
            f, ax = plt.subplots() 
            ax.bar(range(len(QVals)),QVals);
            thrline=ax.axhline(thresholdRead,color='red',ls='--')
            ax.set_title("Intensity distribution")
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Q Boxes sorted by Intensity")
            plt.show()

        ctr = 0 
        for QBlock in TEMPLATE.QBlocks:
            for Que in QBlock.Qs:
                Qboxvals=[]
                for pt in Que.pts:
                    # shifted
                    ptXY=(pt.x + QBlock.shift,pt.y)
                    x,y=ptXY

                    # Supplimentary points
                    # xminus,xplus= x-int(w/2),x+w-int(w/2) 
                    # xminus2,xplus2= x+int(w/2),x+w+int(w/2) 
                    # yminus,yplus= y-int(h/2.7),y+h-int(h/2.7) 
                    # yminus2,yplus2= y+int(h/2.7),y+h+int(h/2.7) 

                    #  This Plus method is better than having bigger box as gray would also get marked otherwise. Bigger box is rather an invalid alternative.
                    check_rects = [ 
                        [y,y+h,x,x+w], 
                        # [yminus,yplus,x,x+w], 
                        # [yminus2,yplus2,x,x+w], 
                        # [y,y+h,xminus,xplus], 
                        # [y,y+h,xminus2,xplus2], 
                    ]
                    
                    # This is NOT usual thresholding, rather call it boxed mean-thresholding
                    detected=False
                    for rect in check_rects:
                        boxval = cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0]
                        if(thresholdRead > boxval):
                            detected=True
                        if(detected):break;

                    if(not detected): 
                        #reset boxval to first rect 
                        rect = check_rects[0]
                        boxval = cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0]
                    
                    #for hist
                    Qboxvals.append(boxval)

                    # cv2.rectangle(retimg,(x,y),(x+w,y+h),clrs[int(detected)],bord)
                    cv2.rectangle(retimg,(x,y),(x+w,y+h),grey,bord)

                    if (detected):
                        blackTHRs.append(boxval)
            #             try:
                        q = Que.qNo
                        val = pt.val
                        if(Que.qType=="QTYPE_ROLL"):
                            key1,key2 = 'Roll',q[1:] #'r1'
                        elif(Que.qType=="QTYPE_MED"):
                            key1,key2 = q,q
                        elif(Que.qType=="QTYPE_INT"):
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
                        
                        # if(np.random.randint(0,10)==0):
                        #     cv2.putText(retimg,"["+str(int(boxval))+"]",(x-2*w,y+h),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE/2,(10,10,10),3)
                                     
            #             except:
            #                 #No dict key for that point
            #                 print(pt,'This shouldnt print after debugs')
                    
                    # // if(detected)
                    else:
                        whiteTHRs.append(boxval)                    
                        # if(np.random.randint(0,20)==0):
                        #     cv2.putText(retimg,"["+str(int(boxval))+"]",(x-w//2,y+h//2),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE/2,(10,10,10),3)
                
                if(showimglvl>=1):
                    # For int types:
                    # axes[ctr//2].hist(Qboxvals, bins=range(0,256,16))
                    # axes[ctr//2].set_ylabel(Que.qNo[:-2])
                    # axes[ctr//2].legend(["D1","D2"],prop={"size":6})
                    if(Que.qType == "QTYPE_INT" or Que.qType == "QTYPE_MCQ"):
                        qNums[Que.qType].append(Que.qNo)
                        allQboxvals[Que.qType].append(Qboxvals)
                ctr += 1
            
        if(showimglvl>=2):
            # plt.draw()
            f, axes = plt.subplots(len(allQboxvals),sharey=True)
            f.canvas.set_window_title(name)
            ctr=0
            for k,boxvals in allQboxvals.items():
                axes[ctr].title.set_text(typeName[k]+" Type")
                axes[ctr].boxplot(boxvals)
                thrline=axes[ctr].axhline(thresholdRead,color='red',ls='--')
                thrline.set_label("THR")
                axes[ctr].set_ylabel("Intensity")
                axes[ctr].set_xticklabels(qNums[k])
                # axes[ctr].legend()
                ctr+=1
            # imshow will do the waiting
            plt.tight_layout(pad=0.5)
            
            if(showimglvl>=4):
                plt.show() 
            # if(showimglvl>=2):
            #     plt.show() 
            # else:
            #     plt.draw() 
            #     plt.pause(0.01)

        # Translucent
        retimg = cv2.addWeighted(retimg,alpha,output,1-alpha,0,output)    
        # print('Keep THR between : ',,np.mean(whiteTHRs))
        global maxBlackTHR,minWhiteTHR
        maxBlackTHR = max(maxBlackTHR,np.max(blackTHRs))
        minWhiteTHR = min(minWhiteTHR,np.min(whiteTHRs))
        
        ## Real helping stats:

        # cv2.putText(retimg,"avg: "+str(["avgBlack: "+str(round(np.mean(blackTHRs),2)),"avgWhite: "+str(round(np.mean(whiteTHRs),2))]),(20,50),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE,(50,20,10),4)
        # cv2.putText(retimg,"ext: "+str(["maxBlack: "+str(round(np.max(blackTHRs),2)),"minW(gray): "+str(round(np.min(whiteTHRs),2))]),(20,90),cv2.FONT_HERSHEY_SIMPLEX, CV2_FONTSIZE,(50,20,10),4)
        
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
