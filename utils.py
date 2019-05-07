"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""
from globals import *

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
for _dir in [saveMarkedDir]:
    if(not os.path.exists(_dir)):
        print('Created : '+ _dir)
        os.mkdir(_dir)
        for sl in ['HE','JE']:#,'HH','JH']:
            os.mkdir(_dir+sl)
            os.mkdir(_dir+sl+'/stack')
            os.mkdir(_dir+sl+'/_MULTI_')
            os.mkdir(_dir+sl+'/_MULTI_'+'/stack')
            # os.mkdir(_dir+sl+'/_BADSCAN_')
            # os.mkdir(_dir+sl+'/_BADSCAN_'+'/stack')
    else:
        print('Already present : '+_dir)

for _dir in [manualDir,resultDir]:
    if(not os.path.exists(_dir)):
            print('Created : '+ _dir)
            os.mkdir(_dir)
    else:
        print('Already present : '+_dir)

for _dir in [multiMarkedPath,errorPath,verifyPath,badRollsPath]:
    if(not os.path.exists(_dir)):
        print('Created : '+ _dir)
        os.mkdir(_dir)
        for sl in ['HE','JE']:#,'HH','JH']:
            os.mkdir(_dir+sl)
    else:
        print('Already present : '+_dir)


# In[64]:

def pad(val,array):
    if(len(val) < len(array)):
        for i in range(len(array)-len(val)):
            val.append('V')


def waitQ():
    while(cv2.waitKey(1)& 0xFF != ord('q')):pass
    cv2.destroyAllWindows()

def normalize_util(img, alpha=0, beta=255):
    return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)#, dtype=cv2.CV_32F)

def normalize_hist(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]

def resize_util(img, u_width, u_height=None):
    if u_height == None:
        h,w=img.shape[:2]
        u_height = int(h*u_width/w)
    return cv2.resize(img,(u_width,u_height))

def resize_util_h(img, u_height, u_width=None):
    if u_width == None:
        h,w=img.shape[:2]
        u_width = int(w*u_height/h)
    return cv2.resize(img,(u_width,u_height))

### Image Template Part ###
# TODO : Create class to put these into 
marker = cv2.imread('inputs/omr_marker.jpg',cv2.IMREAD_GRAYSCALE) #,cv2.CV_8UC1/IMREAD_COLOR/UNCHANGED
marker = resize_util(marker, int(uniform_width/templ_scale_fac))
marker = cv2.GaussianBlur(marker, (5, 5), 0)
marker = cv2.normalize(marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# marker_eroded_sub = marker-cv2.erode(marker,None)
marker_eroded_sub = marker - cv2.erode(marker, kernel=np.ones((5,5)),iterations=5)
# lonmarkerinv = cv2.imread('inputs/omr_autorotate.jpg',cv2.IMREAD_GRAYSCALE)
# lonmarkerinv = imutils.rotate_bound(lonmarkerinv,angle=180)
# lonmarkerinv = imutils.resize(lonmarkerinv,height=int(lonmarkerinv.shape[1]*0.75))
# cv2.imwrite('inputs/lonmarker-inv-resized.jpg',lonmarkerinv)
### //Image Template Part ###

def show(name,orig,pause=1,resize=False,resetpos=None):
    global windowX, windowY, display_width
    if(type(orig) == type(None)):
        print(name," NoneType image to show!")
        if(pause):
            cv2.destroyAllWindows()
        return
    origDim = orig.shape[:2]
    img = resize_util(orig,display_width,display_height) if resize else orig
    cv2.imshow(name,img)
    if(resetpos):
        windowX=resetpos[0]
        windowY=resetpos[1]
    cv2.moveWindow(name,windowX,windowY)
    
    h,w = img.shape[:2]
    
    # Set next window position
    if(windowX+w > windowWidth):
        windowX = 0
        if(windowY+h > windowHeight):
            windowY = 0
        else:
            windowY+=h
    else:
        windowX+=w

    if(pause):
        waitQ()
        

def putLabel(img,label, size):
    scale = img.shape[1]/display_width
    bgVal = int(np.mean(img))
    pos = (int(scale*80), int(scale*30))
    clr = (255 - bgVal,)*3
    img[(pos[1]-size*30):(pos[1]+size*2), : ] = bgVal
    cv2.putText(img,label,pos,cv2.FONT_HERSHEY_SIMPLEX, size, clr, 3)


def getPlotImg():
    plt.savefig('tmp.png')
    # img = cv2.imread('tmp.png',cv2.IMREAD_COLOR)
    img = cv2.imread('tmp.png',cv2.IMREAD_GRAYSCALE)
    os.remove("tmp.png")
    # plt.cla() 
    # plt.clf()
    plt.close()
    return img

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


    maxWidth = max(int(widthA), int(widthB))
    # maxWidth = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

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
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def dist(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))
def getslope(pt1,pt2):
    return float(pt2[1]-pt1[1])/float(pt2[0]-pt1[0])
def check_min_dist(pt,pts,min_dist):
    for p in pts:
        if(dist(pt,p) < min_dist):
            return False
    return True

def move(error,filepath,filepath2,filename):
    print("Error-Code: "+str(error))
    print("Source:  "+filepath)
    print("Destination: " + filepath2 + filename)
    return None
    global filesMoved
    # print(filepath,filepath2,filename,array)
    if(os.path.exists(filepath)):
        if(os.path.exists(filepath2+filename)):
            print('ERROR : Duplicate file at '+filepath2+filename)
        os.rename(filepath,filepath2+filename)
        append = [BATCH_NO,error,filename,filepath2]
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

def angle(p1, p2, p0):
    dx1 = float(p1[0] - p0[0])
    dy1 = float(p1[1] - p0[1])
    dx2 = float(p2[0] - p0[0])
    dy2 = float(p2[1] - p0[1])
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);


def checkMaxCosine(approx):
    # assumes 4 pts present
    maxCosine = 0
    minCosine = 1.5
    for i in range(2, 5):
        cosine  = abs(angle(approx[i % 4], approx[i - 2], approx[i - 1]));
        maxCosine = max(cosine, maxCosine);
        minCosine = min(cosine, minCosine);
    # TODO add to plot dict
    print(maxCosine)
    if(maxCosine >= 0.35):
        print('Quadrilateral is not a rectangle.')
        return False
    return True;

def validateRect(approx):
    # TODO: add logic from app?!
    return len(approx)==4 and checkMaxCosine(approx.reshape(4,2))
     
def auto_canny(image, sigma=0.93):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged


def resetSaveImg(key):
    global saveImgList
    saveImgList[key] = []

def appendSaveImg(key,img):
    global saveImgList
    if(saveimglvl >= int(key)):
        if(key not in saveImgList):
            saveImgList[key] = []
        saveImgList[key].append(img.copy())

def findPage(image_norm):
    # Done: find ORIGIN for the quadrants
    # Done, Auto tune! : Get canny parameters tuned (https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)    
    

    image_norm = normalize_util(image_norm)
    ret, image_norm = cv2.threshold(image_norm,200,255,cv2.THRESH_TRUNC)
    image_norm = normalize_util(image_norm)
    
    appendSaveImg(0,image_norm)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    """
    # Closing is reverse of Opening, Dilation followed by Erosion.
    A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels
    under the kernel is 1, otherwise it is eroded (made to zero).
    """
    # Close the small holes, i.e. Complete the edges on canny image
    closed = cv2.morphologyEx(image_norm, cv2.MORPH_CLOSE, kernel)
    
    appendSaveImg(0,closed)

    edge = cv2.Canny(closed, 185, 55)

    # findContours returns outer boundaries in CW and inner boundaries in ACW order.
    cnts = imutils.grab_contours(cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
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
        # print("Area",cv2.contourArea(c), "Peri", peri)

        # check its rectangle-ness:
        if(validateRect(approx)):
            sheet = np.reshape(approx,(4,-1))
            cv2.drawContours(image_norm, [approx], -1, (0,255, 0), 2)
            cv2.drawContours(edge, [approx], -1, (255,255,255), 10)
            break
        # box = perspective.order_points(box)
    print("Found largest quadrilateral: ", sheet)
    # sobel = cv2.addWeighted(cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=3),0.5,cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=3),0.5,0,edge)
    if sheet==[]:
        print("Error: Paper boundary not found! Should closeUp be = True?")

    # ExcessDo : make it work on killer images
    # edge2 = auto_canny(image_norm)
    # show('Morphed Edges',np.hstack((closed,edge)),1,1)
        
    appendSaveImg(0,edge)
    return sheet

# Nope, sometimes exact page contour is missed. - perhaps deprecated now - we have cropped page now.
def getBestMatch(image_eroded_sub, num_steps=10, iterLim=50):
    global marker_eroded_sub

    # match_precision is how minutely to scan ?!
    x=[int(scaleRange[0]*match_precision),int(scaleRange[1]*match_precision)]
    if((x[1]-x[0])> iterLim*match_precision/num_steps):
        print("Too many iterations : %d, reduce scaleRange" % ((x[1]-x[0])*num_steps/match_precision) )
        return None

    h, w = marker_eroded_sub.shape[:2]
    res, best_scale=None, None
    allMaxT = 0
    for r0 in range(x[1],x[0], -1*match_precision//num_steps): #reverse order
        s=float(r0)/match_precision
        if(s==0.0):
            continue
        templ_scaled = imutils.resize(marker if ERODE_SUB_OFF else marker_eroded_sub, height = int(h*s))
        res = cv2.matchTemplate(image_eroded_sub,templ_scaled,cv2.TM_CCOEFF_NORMED)

        # res is the black image with white dots
        maxT = res.max()
        if(allMaxT < maxT):
            # print('Scale: '+str(s)+', Circle Match: '+str(round(maxT*100,2))+'%')
            best_scale, allMaxT = s, maxT
    if(allMaxT < thresholdCircle):
        print("Warnning: Template matching too low! Should pass closeUp = True?")
        if(showimglvl>-1):
            show("res",res,1,0)
    print('') #close buf
    return best_scale, allMaxT

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
thresholdCircles=[]
badThresholds=[]
veryBadPoints=[]
def getROI(filepath,filename,image, closeup=False):
    global clahe, marker_eroded_sub
    resetSaveImg(0)
    resetSaveImg(1)
    appendSaveImg(0,image)


    """
    TODO later Autorotate:
        - Rotate 90 : check page width:height, CW/ACW? - do CW, then pass to 180 check.
        - Rotate 180 : 
            Nope, OMR specific, paper warping may be imperfect. - check markers centroid
            Nope - OCR check
            Match logo - can work, but 'lon' too big and may unnecessarily rotate? - but you know the scale
            Check roll field morphed 
    """

    # TODO: (remove closeup bool) Automate the case of close up scan(incorrect page)-
    # ^Note: App rejects closeups along with others

    # image = resize_util(image, uniform_width, uniform_height)

    # Preprocessing the image
    img = image.copy()
    # TODO: need to detect if image is too blurry already! (M1: check crop dimensions b4 resizing; coz it won't be blurry otherwise _/)
    img = cv2.GaussianBlur(img,(3,3),0)
    image_norm = normalize_util(img);

    if(closeup == False):
        #Need this resize for arbitrary high res images: before passing to findPage
        if(image_norm.shape[1] > uniform_width*2):
            image_norm = resize_util(image_norm, uniform_width*2)
        sheet = findPage(image_norm)
        if sheet==[]:
            return None
        # Warp layer 1
        image_norm = four_point_transform(image_norm, sheet)
    
    # Resize only after cropping the page for clarity
    image_norm = resize_util(image_norm, uniform_width, uniform_height)
    appendSaveImg(0,image_norm)

    image = resize_util(image, uniform_width, uniform_height)
    if(showimglvl>=3):
        show('Before Template Matching',np.hstack((image,image_norm)),0,1)

    image_eroded_sub = normalize_util(image_norm) if ERODE_SUB_OFF else normalize_util(image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=5))
    # Quads on warped image
    quads={}
    h1, w1 = image_eroded_sub.shape[:2]
    midh,midw = h1//3, w1//2
    origins=[[0,0],[midw,0],[0,midh],[midw,midh]]
    quads[0]=image_eroded_sub[0:midh,0:midw];
    quads[1]=image_eroded_sub[0:midh,midw:w1];
    quads[2]=image_eroded_sub[midh:h1,0:midw];
    quads[3]=image_eroded_sub[midh:h1,midw:w1];

    # Draw Quadlines
    image_eroded_sub[ : , midw:midw+2] = 255
    image_eroded_sub[ midh:midh+2, : ] = 255

    # print(image_eroded_sub.shape)
    # show("2",image_eroded_sub)

    best_scale, allMaxT = getBestMatch(image_eroded_sub)
    if(best_scale == None):
        # TODO: Plot and see performance of scaleRange
        print("No matchings for given scaleRange:",scaleRange)
        show('Quads',image_eroded_sub)
        return None

    templ = imutils.resize(marker if ERODE_SUB_OFF else marker_eroded_sub, height = int(marker_eroded_sub.shape[0]*best_scale))
    h,w=templ.shape[:2]
    centres = []
    sumT, maxT = 0, 0
    print("best_scale",best_scale)
    for k in range(0,4):
        res = cv2.matchTemplate(quads[k],templ,cv2.TM_CCOEFF_NORMED)
        maxT = res.max()
        print("Q"+str(k)+": maxT", round(maxT,3))
        if(maxT < thresholdCircle or abs(allMaxT-maxT) >= thresholdVar):
            # Warning - code will stop in the middle. Keep Threshold low to avoid.
            print(filename,"\nError: No circle found in Quad",k+1, "\n\tthresholdVar", thresholdVar, "maxT", maxT,"allMaxT",allMaxT, "Should closeUp be = False?")
            if(showimglvl>-1):
                show('no_pts_'+filename,image_eroded_sub,0,1)
                show('res_Q'+str(k),res,1,1)
            return None

        pt=np.argwhere(res==maxT)[0];
        pt = [pt[1],pt[0]]
        pt[0]+=origins[k][0]
        pt[1]+=origins[k][1]
        # print(">>",pt)
        image_norm = cv2.rectangle(image_norm,tuple(pt),(pt[0]+w,pt[1]+h),(150,150,150),2)
        # display: 
        image_eroded_sub = cv2.rectangle(image_eroded_sub,tuple(pt),(pt[0]+w,pt[1]+h),(50,50,50) if ERODE_SUB_OFF else (155,155,155), 4)
        centres.append([pt[0]+w/2,pt[1]+h/2])
        sumT += maxT

    # analysis data
    thresholdCircles.append(sumT/4)

    image_norm = four_point_transform(image_norm, np.array(centres))
    # appendSaveImg(0,image_eroded_sub)
    # appendSaveImg(0,image_norm)

    appendSaveImg(1,image_eroded_sub)
    res = cv2.matchTemplate(image_eroded_sub,templ,cv2.TM_CCOEFF_NORMED)
    res[ : , midw:midw+2] = 255
    res[ midh:midh+2, : ] = 255
    if(showimglvl>=2):# and showimglvl < 4):
        image_eroded_sub = resize_util_h(image_eroded_sub, image_norm.shape[0])
        image_eroded_sub[:,-5:] = 0
        show('Warped',np.hstack((image_eroded_sub, image_norm)),0)

    # iterations : Tuned to 2.
    # image_eroded_sub = image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=2)
    return image_norm


def getGlobalThreshold(QVals):
    """
        Note: Cannot assume col has only-gray or only-white bg (in which case there is only one jump). 
              So there will be either 1 or 2 jumps.
        1 Jump : 
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||
        
        2 Jumps : 
                  ......
                  |||||| <-- wrong THR 
              ....||||||
              |||||||||| <-- safe THR
            ..||||||||||
            ||||||||||||

        The abstract "First LARGE GAP" is perfect for this.
        Current code is considering ONLY TOP 2 jumps(>= MIN_GAP) to be big, gives the smaller one
    
    """    
    # Sort the Q vals
    QVals= sorted(QVals)
    # Find the FIRST LARGE GAP and set it as threshold:
    l=len(QVals)-1
    max1,thr1=MIN_JUMP,255
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        if(jump > max1):
            max1 = jump
            thr1 = QVals[i-1] + jump/2

    # Make use of the fact that the JUMP_DELTA(Vertical gap ofc) between values at detected jumps would be atleast 20
    max2,thr2=MIN_JUMP,255
    # Requires atleast 1 gray box to be present (Roll field will ensure this)
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        newThr = QVals[i-1] + jump/2
        if(jump > max2 and abs(thr1-newThr) > JUMP_DELTA):
            max2=jump
            thr2=newThr
    # thresholdRead = min(thr1,thr2) 
    thresholdRead, j_low, j_high = thr1, thr1 - max1//2, thr1 + max1//2
    if(thr1 > thr2):
        thresholdRead, j_low, j_high = thr2, thr2 - max2//2, thr2 + max2//2

    return thresholdRead, j_low, j_high


def getLocalThreshold(QVals, globalTHR):
    """
    Assumption : Colwise background color is uniformly gray or white, but not alternating 
    In this case there is atmost one jump.

    0 Jump :
                    <-- safe THR?
           .......
        ...|||||||
        ||||||||||  <-- safe THR?
    How to decide given range is above or below gray?
    -> global QVals shall absolutely help here. Just run same function on total QVals instead of colwise _//

    1 Jump : 
            ......
            ||||||
            ||||||  <-- risky THR
            ||||||  <-- safe THR
        ....||||||
        ||||||||||
    
    """
    # Sort the Q vals
    QVals= sorted(QVals)

    # Small no of pts cases:
    qmin, qmax, qmean, qstd = np.min(QVals), np.max(QVals), round(np.mean(QVals),2), round(np.std(QVals),2)
    gap = (qmax - qmin)
    # base case: 1 or 2 pts
    if(len(QVals) < 3): 
        return globalTHR if gap < MIN_GAP else qmean

    # All Black or All White case
    noOutliers = qstd < MIN_STD
    GVals = [round(abs(q-qmean),2) for q in QVals]
    gmean, gstd = round(np.mean(GVals),2), round(np.std(GVals),2)
    # print("qstd",qstd, "gstd", gstd,"Gaps in gvals",sorted([round(abs(g-gmean),2) for g in GVals],reverse=True))
    
    # TODO: Make this dependent on max jump!
    # DISCRETION: Pretty critical factor in reading response
    DISCRETION = 2.7 # 2.59 was closest hit, 3.0 is too far
    for g in GVals:
        if(abs(g-gmean) > DISCRETION*gstd):
            noOutliers = False
            break

    if(noOutliers):
        # All Black or All White case
        return globalTHR

    # Find the FIRST LARGE GAP i.e. LARGEST GAP and set it as threshold:
    l=len(QVals)-1
    max1,thr1=MIN_JUMP,255
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        if(jump > max1):
            max1 = jump
            thr1 = QVals[i-1] + jump/2


    # if(thr1 != 255 and qmax < globalTHR and max1 < MIN_JUMP*2):
    #     f, ax = plt.subplots()
    #     ax.bar(range(len(QVals)),QVals);
    #     thrline=ax.axhline(globalTHR,color='red',ls='--', linewidth=4)
    #     thrline.set_label("globalTHR")
    #     thrline=ax.axhline(thr1,color='blue',ls='--', linewidth=4)
    #     thrline.set_label("THR")
    #     ax.set_title("Intensity distribution")
    #     ax.set_ylabel("Intensity")
    #     ax.set_xlabel("Q Boxes sorted by Intensity")
    #     plt.show()

    return thr1

def saveImg(path, final_marked):
    print('Saving Image to '+path)
    cv2.imwrite(path,final_marked)

def readResponse(squad,image,name,save=None,explain=True):
    global clahe
    TEMPLATE = TEMPLATES[squad]
    try:
        img = image.copy()
        origDim = img.shape[:2]
        # print("Cropped dim", origDim)
        # 1846 x 1500
        img = resize_util(img,TEMPLATE.dims[0],TEMPLATE.dims[1])
        print("Resized dim", img.shape[:2])
        img = normalize_util(img)
        # Processing copies
        transp_layer = img.copy()
        final_marked = img.copy()
        # putLabel(final_marked,"Crop Size: " + str(origDim[0])+"x"+str(origDim[1]) + " "+name, size=1)
        
        
        morph = img.copy() #
        appendSaveImg(2,morph)
        # Note: clahe is good for morphology, bad for thresholding
        morph = clahe.apply(morph) 
        appendSaveImg(2,morph)
        # Remove shadows further, make columns/boxes darker (less gamma)
        morph = adjust_gamma(morph,GAMMA_LOW)
        ret, morph = cv2.threshold(morph,220,220,cv2.THRESH_TRUNC)
        morph = normalize_util(morph)
        appendSaveImg(2,morph)
        if(showimglvl>=3):
            show("morph1",morph,0,1)

        alpha = 0.65
        alpha1 = 0.55

        boxW,boxH = TEMPLATE.boxDims
        lang = ['E','H']
        OMRresponse={}
        CLR_BLACK = (50,150,150)
        CLR_WHITE = (250,250,250)
        CLR_GRAY = (220,150,150)
        CLR_DARK_GRAY = (150,150,150)

        multimarked,multiroll=0,0

        blackVals=[0]
        whiteVals=[255]

        if(showimglvl>=5):
            allCBoxvals={"Int":[],"Mcq":[]}#"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}
            qNums={"Int":[],"Mcq":[]}#,"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}


        ### Find Shifts for the QBlocks --> Before calculating threshold!
        # Open : erode then dilate
        # Vertical kernel 
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        morph_v = cv2.morphologyEx(morph, cv2.MORPH_OPEN, v_kernel, iterations=3)
        ret, morph_v = cv2.threshold(morph_v,200,200,cv2.THRESH_TRUNC)
        morph_v = 255 - normalize_util(morph_v)
        
        if(showimglvl>=3):
            show("morph_v",morph_v,0,1)
        appendSaveImg(2,morph_v)

        morphTHR = 60 # for Mobile images
        # morphTHR = 40 # for scan Images
        # best tuned to 5x5 now
        _, morph_v = cv2.threshold(morph_v,morphTHR,255,cv2.THRESH_BINARY)
        morph_v = cv2.erode(morph_v,  np.ones((5,5),np.uint8), iterations = 2)
        
        appendSaveImg(2,morph_v)
        # h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        # morph_h = cv2.morphologyEx(morph, cv2.MORPH_OPEN, h_kernel, iterations=3)
        # ret, morph_h = cv2.threshold(morph_h,200,200,cv2.THRESH_TRUNC)
        # morph_h = 255 - normalize_util(morph_h)
        # show("morph_h",morph_h,0,1)
        # _, morph_h = cv2.threshold(morph_h,morphTHR,255,cv2.THRESH_BINARY)
        # morph_h = cv2.erode(morph_h,  np.ones((5,5),np.uint8), iterations = 2)
        if(showimglvl>=3):
            show("morph_thr_eroded", morph_v, 0, 1)
        
        
        appendSaveImg(6,morph_v)

        # templ alignment code
        for QBlock in TEMPLATE.QBlocks:
            s,d = QBlock.orig, QBlock.dims
            # internal constants - wont need change much
            ALIGN_STRIDE, MATCH_COL, ALIGN_STEPS = 1, 5, int(boxW * 2 / 3)
            shift, steps = 0, 0
            THK = 3
            while steps < ALIGN_STEPS:
                L = np.mean(morph_v[s[1]:s[1]+d[1],s[0]+shift-THK:-THK+s[0]+shift+MATCH_COL])
                R = np.mean(morph_v[s[1]:s[1]+d[1],s[0]+shift-MATCH_COL+d[0]+THK:THK+s[0]+shift+d[0]])
                if(QBlock.key=="Int1"):
                    ret = morph_v.copy()
                    cv2.rectangle(ret,(s[0]+shift-THK,s[1]),(s[0]+shift+THK+d[0],s[1]+d[1]),CLR_WHITE,3)
                    appendSaveImg(6,ret)
                # print(shift, L, R)
                LW,RW= L > 100, R > 100
                if(LW):
                    if(RW):
                        break
                    else:
                        shift -= ALIGN_STRIDE
                else:
                    if(RW):
                        shift += ALIGN_STRIDE
                    else:
                        break
                steps += 1

            QBlock.shift = shift
            # sums = sorted(sums, reverse=True)
            # print("Aligned QBlock: ",QBlock.key,"Corrected Shift:", QBlock.shift,", Dimensions:", QBlock.dims, "orig:", QBlock.orig,'\n')

    # if(showimglvl>=3):
        initial_align=img.copy()
        final_align=img.copy()
        mean_vals =img.copy()
        for QBlock in TEMPLATE.QBlocks:
            s,d = QBlock.orig, QBlock.dims
            cv2.rectangle(initial_align,(s[0],s[1]),(s[0]+d[0],s[1]+d[1]),CLR_BLACK,3)
            shift = QBlock.shift
            cv2.rectangle(final_align,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),CLR_BLACK,3)
            for col, pts in QBlock.colpts:
                for pt in pts:
                    x,y = pt.x,pt.y
                    cv2.rectangle(initial_align,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_GRAY,-1)
                    x,y = (pt.x + QBlock.shift,pt.y)
                    cv2.rectangle(final_align,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_GRAY,-1)
                    rect = [y,y+boxH,x,x+boxW]
                    cv2.rectangle(mean_vals,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_GRAY,-1)
                    cv2.putText(mean_vals,'%d'% (cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0]), (rect[2]+2, rect[0] + (boxH*2)//3),cv2.FONT_HERSHEY_SIMPLEX, 0.6,CLR_BLACK,2)
            cv2.putText(final_align,'s%s'% (shift), tuple(s - [TEMPLATE.dims[0]//20,-d[1]//2]),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,CLR_BLACK,4)

        appendSaveImg(1,initial_align)
        # appendSaveImg(1,morph_v)
        appendSaveImg(1,final_align)
        # show("Initial Template Overlay", initial_align, 0, 1, [0,0])
        # show("Corrected Template Overlay", final_align, 1, 1)# [final_align.shape[1],0])
        
        
        # show("Template Overlay", mean_vals, 0, 1)
        appendSaveImg(3,mean_vals)
        appendSaveImg(4,img)
        
        # for All Black or All White case -
        AllQBlockvals=[]
        for QBlock in TEMPLATE.QBlocks:
            for col, pts in QBlock.colpts:
                for pt in pts:
                    # shifted
                    x,y = (pt.x + QBlock.shift,pt.y)
                    rect = [y,y+boxH,x,x+boxW]
                    AllQBlockvals.append(cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0])
        AllQBlockvals = sorted(AllQBlockvals)
        globalTHR, j_low, j_high = getGlobalThreshold(AllQBlockvals)
        f, ax = plt.subplots()
        ax.bar(range(len(AllQBlockvals)),AllQBlockvals);
        ax.set_title("Mean Intensity Histogram")
        thrline=ax.axhline(globalTHR,color='green',ls='-', linewidth=4)
        thrline.set_label("Global Threshold")
        # thrline=ax.axhline(j_low,color='red',ls='--', linewidth=4)
        # thrline=ax.axhline(j_high,color='red',ls='--', linewidth=4)
        # thrline.set_label("Boundary Line")
        ax.set_ylabel("Bubble Mean Intensity")
        ax.set_xlabel("Bubble Number(sorted)")
        # plt.show()
        # ax.legend()
        hist = getPlotImg()
        # show("Hist", hist, 0, 1)
        appendSaveImg(3,hist)
        appendSaveImg(4,hist)
        appendSaveImg(1,hist)

        

        print(name,"globalTHR: ",round(globalTHR,2))
        thresholdReadAvg, colNos = 0, 0
        for QBlock in TEMPLATE.QBlocks:
            blockColNo = 0
            shift=QBlock.shift
            s,d = QBlock.orig, QBlock.dims
            key = QBlock.key[:3]
            # cv2.rectangle(final_marked,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),CLR_BLACK,3)
            for col, pts in QBlock.colpts:
                colNos += 1
                blockColNo += 1
                QBlockvals=[]
                for pt in pts:
                    # shifted
                    x,y = (pt.x + QBlock.shift,pt.y)
                    rect = [y,y+boxH,x,x+boxW]
                    QBlockvals.append(cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0])
                
                QBlockvals= sorted(QBlockvals)
                thresholdRead = getLocalThreshold(QBlockvals,globalTHR)
                # thresholdRead = globalTHR
                # print(pts[0].qNo,key,blockColNo, "THR: ",thresholdRead)
                thresholdReadAvg += thresholdRead
                if(
                    # (pts[0].qNo)=="q10" or 
                    # (pts[0].qNo+str(blockColNo))=="q15" or 
                    showimglvl>=6
                 ):
                    show("QBlock: "+key, img[s[1] : s[1] + d[1], s[0]+shift : s[0]+shift+ d[0]],0,1)
                    f, ax = plt.subplots()
                    ax.bar(range(len(QBlockvals)),QBlockvals);
                    thrline=ax.axhline(thresholdRead,color='green',ls='-', linewidth=4)
                    thrline.set_label("Local Threshold")
                    thrline=ax.axhline(globalTHR,color='red',ls='--', linewidth=4)
                    # thrline.set_label("Safe Threshold")
                    thrline.set_label("Global Threshold")
                    ax.set_title("Mean Intensity Histogram for "+ key +"."+ pts[0].qNo+"."+str(blockColNo))
                    ax.set_ylabel("Bubble Mean Intensity")
                    ax.set_xlabel("Bubble Number(sorted)")
                    ax.legend()
                    # plt.show()
                    # appendSaveImg(5,getPlotImg())
                
                QBlockvals=[]
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
                    
                    if (detected):
                        cv2.rectangle(final_marked,(int(x+boxW/12),int(y+boxH/12)),(int(x+boxW-boxW/12),int(y+boxH-boxH/12)), CLR_DARK_GRAY,-1)
                    else:
                        cv2.rectangle(final_marked,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_GRAY,-1)

                    #for hist
                    QBlockvals.append(boxval0)
                    if (detected):
                        q = pt.qNo
                        val = str(pt.val)
                        cv2.putText(final_marked,val,(x,y),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(20,20,10),5)
                        # Only send rolls multi-marked in the directory
                        multimarkedL = q in OMRresponse
                        multimarked = multimarkedL or multimarked
                        OMRresponse[q] = (OMRresponse[q] + val) if multimarkedL else val
                        multiroll = multimarkedL and 'roll' in str(q)
                        blackVals.append(boxval0)
                    else:
                        whiteVals.append(boxval0)
                    # /for col
                if( showimglvl>=5):
                    if(key in allCBoxvals):
                        qNums[key].append(key[:2]+'_c'+str(blockColNo))
                        allCBoxvals[key].append(QBlockvals)
            # /for QBlock
        thresholdReadAvg /= colNos
        # Translucent
        cv2.addWeighted(final_marked,alpha,transp_layer,1-alpha,0,final_marked)

        if( showimglvl>=5):
            # plt.draw()
            f, axes = plt.subplots(len(allCBoxvals),sharey=True)
            f.canvas.set_window_title(name)
            ctr=0
            typeName={"Int":"Integer","Mcq":"MCQ","Med":"MED","Rol":"Roll"}
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
            plt.show()

        if ( type(save) != type(None) ):
            save = save+('_MULTI_/' if multiroll else '')
            saveImg(save+name+'_marked.jpg', final_marked)

        if(showimglvl>=1):
            show("Final Template: "+name,final_marked,1,1)

        appendSaveImg(1,final_marked)

        saveImgList[3] = [hist, final_marked]
        # to show img
        # save = None 

        saveOrShowStacks(0, name, save,0)
        saveOrShowStacks(1, name, save)

        return OMRresponse,final_marked,multimarked,multiroll

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error from readResponse: ",e)
        print(exc_type, fname, exc_tb.tb_lineno)

def saveOrShowStacks(key, name, save=None,pause=1):
    global saveImgList
    if(saveimglvl >= int(key)):
        result = np.hstack(tuple([resize_util_h(img,uniform_height) for img in saveImgList[key]]))
        result = resize_util(result,min(len(saveImgList[key])*uniform_width//3,int(uniform_width*2.5)))
        if (saveimglvl>=1 or type(save) != type(None) ):
            saveImg(save+'stack/'+name+'_'+str(key)+'_stack.jpg', result)
        else:
            show(name+'_'+str(key),result,pause,0)
            