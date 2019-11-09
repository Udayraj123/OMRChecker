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

def check_dirs(paths):
    print('Checking Directories...')
    for _dir in [paths.saveMarkedDir]:
        if(not os.path.exists(_dir)):
            print('Created : '+ _dir)
            os.makedirs(_dir)
            os.mkdir(_dir+'/stack')
            os.mkdir(_dir+'/_MULTI_')
            os.mkdir(_dir+'/_MULTI_'+'/stack')
            # os.mkdir(_dir+sl+'/_BADSCAN_')
            # os.mkdir(_dir+sl+'/_BADSCAN_'+'/stack')
        else:
            print('Present : '+_dir)

    for _dir in [paths.manualDir,paths.resultDir]:
        if(not os.path.exists(_dir)):
                print('Created : '+ _dir)
                os.makedirs(_dir)
        else:
            print('Present : '+_dir)

    for _dir in [paths.multiMarkedDir,paths.errorsDir,paths.badRollsDir]:
        if(not os.path.exists(_dir)):
            print('Created : '+ _dir)
            os.makedirs(_dir)
        else:
            print('Present : '+_dir)


# In[64]:
def waitQ():
    ESC_KEY = 27
    while(cv2.waitKey(1) & 0xFF not in [ord('q'), ESC_KEY]):pass
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
    margin = 25
    w += margin
    h += margin
    if(windowX+w > windowWidth):
        windowX = 0
        if(windowY+h > windowHeight):
            windowY = 0
        else:
            windowY += h 
    else:
        windowX += w

    if(pause):
        print("Showing '"+name+"'\n\tPress Q on image to continue; Press Ctrl + C in terminal to exit")
        waitQ()
        

def putLabel(img,label, size):
    scale = img.shape[1]/display_width
    bgVal = int(np.mean(img))
    pos = (int(scale*80), int(scale*30))
    clr = (255 - bgVal,)*3
    img[(pos[1]-size*30):(pos[1]+size*2), : ] = bgVal
    cv2.putText(img,label,pos,cv2.FONT_HERSHEY_SIMPLEX, size, clr, 3)

def drawTemplateLayout(img, template, shifted=True, draw_qvals=False, border=-1):
    img = resize_util(img,template.dims[0],template.dims[1])
    final_align = img.copy()
    boxW,boxH = template.bubbleDims
    for QBlock in template.QBlocks:
        s,d = QBlock.orig, QBlock.dims
        shift = QBlock.shift
        if(shifted):
            cv2.rectangle(final_align,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),CLR_BLACK,3)
        else:
            cv2.rectangle(final_align,(s[0],s[1]),(s[0]+d[0],s[1]+d[1]),CLR_BLACK,3)
        for qStrip, qBoxPts in QBlock.traverse_pts:
            for pt in qBoxPts:
                x,y = (pt.x + QBlock.shift,pt.y) if shifted else (pt.x,pt.y)
                cv2.rectangle(final_align,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_DARK_GRAY,border)
                if(draw_qvals):
                    rect = [y,y+boxH,x,x+boxW]
                    cv2.putText(final_align,'%d'% (cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0]), (rect[2]+2, rect[0] + (boxH*2)//3),cv2.FONT_HERSHEY_SIMPLEX, 0.6,CLR_BLACK,2)
        if(shifted):
            cv2.putText(final_align,'s%s'% (shift), tuple(s - [template.dims[0]//20,-d[1]//2]),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,CLR_BLACK,4)
    return final_align

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
    # print(maxCosine)
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
    if(saveimglvl >= int(key)):
        global saveImgList
        if(key not in saveImgList):
            saveImgList[key] = []
        saveImgList[key].append(img.copy())

def findPage(image_norm):
    # Done: find ORIGIN for the quadrants
    # Done, Auto tune! : Get canny parameters tuned (https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)    
    

    image_norm = normalize_util(image_norm)
    ret, image_norm = cv2.threshold(image_norm,200,255,cv2.THRESH_TRUNC)
    image_norm = normalize_util(image_norm)
    
    appendSaveImg(1,image_norm)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    """
    # Closing is reverse of Opening, Dilation followed by Erosion.
    A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels
    under the kernel is 1, otherwise it is eroded (made to zero).
    """
    # Close the small holes, i.e. Complete the edges on canny image
    closed = cv2.morphologyEx(image_norm, cv2.MORPH_CLOSE, kernel)
    
    appendSaveImg(1,closed)

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
    # sobel = cv2.addWeighted(cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=3),0.5,cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=3),0.5,0,edge)
    # ExcessDo : make it work on killer images
    # edge2 = auto_canny(image_norm)
    # show('Morphed Edges',np.hstack((closed,edge)),1,1)
        
    appendSaveImg(1,edge)
    return sheet




# Resizing the marker within scaleRange at rate of descent_per_step to find the best match.
def getBestMatch(image_eroded_sub, marker):

    descent_per_step = (markerScaleRange[1]-markerScaleRange[0])//markerScaleSteps
    h, w = marker.shape[:2]
    res, best_scale=None, None
    allMaxT = 0

    for r0 in np.arange(markerScaleRange[1],markerScaleRange[0],-1*descent_per_step): #reverse order
        s=float(r0*1/100)
        if(s == 0.0):
            continue
        templ_scaled = imutils.resize(marker if ERODE_SUB_OFF else marker, height = int(h*s))
        # res is the black image with white dots
        res = cv2.matchTemplate(image_eroded_sub,templ_scaled,cv2.TM_CCOEFF_NORMED)

        maxT = res.max()
        if(allMaxT < maxT):
            # print('Scale: '+str(s)+', Circle Match: '+str(round(maxT*100,2))+'%')
            best_scale, allMaxT = s, maxT

    if(allMaxT < thresholdCircle):
        print("\tWarning: Template matching too low! Should you pass --noCropping flag?")
        if(showimglvl>=1):
            show("res",res,1,0)

    if(best_scale == None):
            print("No matchings for given scaleRange:",markerScaleRange)
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
#TODO Fill these for stats
thresholdCircles=[]
badThresholds=[]
veryBadPoints=[]

def getROI(image, filename, noCropping=False):
    global clahe
    for i in range(saveimglvl):
        resetSaveImg(i+1)
    
    appendSaveImg(1,image)
    """
    TODO later Autorotate:
        - Rotate 90 : check page width:height, CW/ACW? - do CW, then pass to 180 check.
        - Rotate 180 : 
            Nope, OMR specific, paper warping may be imperfect. - check markers centroid
            Nope - OCR check
            Match logo - can work, but 'lon' too big and may unnecessarily rotate? - but you know the scale
            Check roll field morphed 
    """

    # TODO: (remove noCropping bool) Automate the case of close up scan(incorrect page)-
    # ^Note: App rejects croppeds along with others

    # image = resize_util(image, uniform_width, uniform_height)

    # Preprocessing the image
    img = image.copy()
    # TODO: need to detect if image is too blurry already! (M1: check noCropping dimensions b4 resizing; coz it won't be blurry otherwise _/)
    img = cv2.GaussianBlur(img,(3,3),0)
    image_norm = normalize_util(img);

    if(noCropping == False):
        #Need this resize for arbitrary high res images: before passing to findPage
        if(image_norm.shape[1] > uniform_width*2):
            image_norm = resize_util(image_norm, uniform_width*2)
        sheet = findPage(image_norm)
        if sheet==[]:
            print("\tError: Paper boundary not found! Should you pass --noCropping flag?")
            return None
        else:
            print("Found page corners: ", sheet.tolist())

        # Warp layer 1
        image_norm = four_point_transform(image_norm, sheet)
    
    # Resize only after cropping the page for clarity as well as uniformity for non noCropping images
    image_norm = resize_util(image_norm, uniform_width, uniform_height)
    image = resize_util(image, uniform_width, uniform_height)
    appendSaveImg(1,image_norm)

    # Return preprocessed image
    return image_norm
    
    
def handle_markers(image_norm, marker):
    global curr_filename

    if ERODE_SUB_OFF:
        image_eroded_sub = normalize_util(image_norm) 
    else:
        image_eroded_sub = normalize_util(image_norm 
                                            - cv2.erode(image_norm, 
                                                        kernel=np.ones((5,5)),
                                                        iterations=5))
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

    best_scale, allMaxT = getBestMatch(image_eroded_sub)
    if(best_scale == None):
        # TODO: Plot and see performance of markerscaleRange
        if(showimglvl>=1):
            show('Quads',image_eroded_sub)
        return None

    templ = imutils.resize(marker if ERODE_SUB_OFF else marker, height = int(marker.shape[0]*best_scale))
    h,w=templ.shape[:2]
    centres = []
    sumT, maxT = 0, 0
    print("Matching Marker:\t", end=" ")
    for k in range(0,4):
        res = cv2.matchTemplate(quads[k],templ,cv2.TM_CCOEFF_NORMED)
        maxT = res.max()
        print("Q"+str(k+1)+": maxT", round(maxT,3), end="\t")
        if(maxT < thresholdCircle or abs(allMaxT-maxT) >= thresholdVar):
            # Warning - code will stop in the middle. Keep Threshold low to avoid.
            print(curr_filename, "\nError: No circle found in Quad",k+1, "\n\tthresholdVar", thresholdVar, "maxT", maxT,"allMaxT",allMaxT, "Should you pass --noCropping flag?")
            if(showimglvl>=1):
                show("no_pts_"+curr_filename, image_eroded_sub,0)
                show("res_Q"+str(k+1),res,1)
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
    print("Scale",best_scale)
    # analysis data
    thresholdCircles.append(sumT/4)

    image_norm = four_point_transform(image_norm, np.array(centres))
    # appendSaveImg(1,image_eroded_sub)
    # appendSaveImg(1,image_norm)

    appendSaveImg(2,image_eroded_sub)
    # res = cv2.matchTemplate(image_eroded_sub,templ,cv2.TM_CCOEFF_NORMED)
    # res[ : , midw:midw+2] = 255
    # res[ midh:midh+2, : ] = 255
    # show("Markers Matching",res)
    if(showimglvl>=2 and showimglvl < 4):
        image_eroded_sub = resize_util_h(image_eroded_sub, image_norm.shape[0])
        image = resize_util_h(image, image_norm.shape[0])
        image_eroded_sub[:,-5:] = 0
        h_stack = np.hstack((image,image_eroded_sub, image_norm))
        show("Warped: "+curr_filename, resize_util(h_stack,int(display_width*1.6)),0,0,[0,0])
    # iterations : Tuned to 2.
    # image_eroded_sub = image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=2)
    return image_norm


def getGlobalThreshold(QVals_orig, plotTitle=None, plotShow=True, sortInPlot=True):
    """
        Note: Cannot assume qStrip has only-gray or only-white bg (in which case there is only one jump). 
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
    QVals = sorted(QVals_orig)
    # Find the FIRST LARGE GAP and set it as threshold:
    l=len(QVals)-1
    max1,thr1=MIN_JUMP,255
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        if(jump > max1):
            max1 = jump
            thr1 = QVals[i-1] + jump/2

# NOTE: thr2 is deprecated, thus is JUMP_DELTA
    # Make use of the fact that the JUMP_DELTA(Vertical gap ofc) between values at detected jumps would be atleast 20
    max2,thr2=MIN_JUMP,255
    # Requires atleast 1 gray box to be present (Roll field will ensure this)
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        newThr = QVals[i-1] + jump/2
        if(jump > max2 and abs(thr1-newThr) > JUMP_DELTA):
            max2=jump
            thr2=newThr
    # globalTHR = min(thr1,thr2) 
    globalTHR, j_low, j_high = thr1, thr1 - max1//2, thr1 + max1//2

    # # For normal images
    # thresholdRead =  116
    # if(thr1 > thr2 and thr2 > thresholdRead):
    #     print("Note: taking safer thr line.")
    #     globalTHR, j_low, j_high = thr2, thr2 - max2//2, thr2 + max2//2

    if(plotTitle is not None):    
        f, ax = plt.subplots()
        ax.bar(range(len(QVals_orig)),QVals if sortInPlot else QVals_orig);
        ax.set_title(plotTitle)
        thrline=ax.axhline(globalTHR,color='green',ls='--', linewidth=5)
        thrline.set_label("Global Threshold")
        thrline=ax.axhline(thr2,color='red',ls=':', linewidth=3)
        thrline.set_label("THR2 Line")
        # thrline=ax.axhline(j_low,color='red',ls='-.', linewidth=3)
        # thrline=ax.axhline(j_high,color='red',ls='-.', linewidth=3)
        # thrline.set_label("Boundary Line")
        # ax.set_ylabel("Mean Intensity")
        ax.set_ylabel("Values")
        ax.set_xlabel("Position")
        ax.legend()
        if(plotShow):
            plt.title(plotTitle)
            plt.show()
    
    return globalTHR, j_low, j_high


def getLocalThreshold(qNo, QVals, globalTHR, noOutliers, plotTitle=None, plotShow=True):
    """
    TODO: Update this documentation too-
    //No more - Assumption : Colwise background color is uniformly gray or white, but not alternating. In this case there is atmost one jump.

    0 Jump :
                    <-- safe THR?
           .......
        ...|||||||
        ||||||||||  <-- safe THR?
    // How to decide given range is above or below gray?
        -> global QVals shall absolutely help here. Just run same function on total QVals instead of colwise _//
    How to decide it is this case of 0 jumps

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
    # base case: 1 or 2 pts
    if(len(QVals) < 3): 
        thr1 = globalTHR if np.max(QVals)-np.min(QVals) < MIN_GAP else np.mean(QVals)
    else:
        # qmin, qmax, qmean, qstd = round(np.min(QVals),2), round(np.max(QVals),2), round(np.mean(QVals),2), round(np.std(QVals),2)
        # GVals = [round(abs(q-qmean),2) for q in QVals]
        # gmean, gstd = round(np.mean(GVals),2), round(np.std(GVals),2)
        # # DISCRETION: Pretty critical factor in reading response
        # # Doesn't work well for small number of values.
        # DISCRETION = 2.7 # 2.59 was closest hit, 3.0 is too far
        # L2MaxGap = round(max([abs(g-gmean) for g in GVals]),2)
        # if(L2MaxGap > DISCRETION*gstd):
        #     noOutliers = False
        
        # # ^Stackoverflow method
        # print(qNo, noOutliers,"qstd",round(np.std(QVals),2), "gstd", gstd,"Gaps in gvals",sorted([round(abs(g-gmean),2) for g in GVals],reverse=True), '\t',round(DISCRETION*gstd,2), L2MaxGap)

        # else:
        # Find the LARGEST GAP and set it as threshold: //(FIRST LARGE GAP)
        l=len(QVals)-1
        max1,thr1=MIN_JUMP,255
        for i in range(1,l):
            jump = QVals[i+1] - QVals[i-1]
            if(jump > max1):
                max1 = jump
                thr1 = QVals[i-1] + jump/2
        # print(qNo,QVals,max1)

        # If not confident, then only take help of globalTHR
        if(max1 < CONFIDENT_JUMP):
            if(noOutliers):
                # All Black or All White case
                thr1 = globalTHR   
            else:
                # TODO: Low confidence parameters here
                pass  

        # if(thr1 == 255):
        #     print("Warning: threshold is unexpectedly 255! (Outlier Delta issue?)",plotTitle)

    if(plotShow and plotTitle is not None):    
        f, ax = plt.subplots()
        ax.bar(range(len(QVals)),QVals);
        thrline=ax.axhline(thr1,color='green',ls=('-.'), linewidth=3)
        thrline.set_label("Local Threshold")
        thrline=ax.axhline(globalTHR,color='red',ls=':', linewidth=5)
        thrline.set_label("Global Threshold")
        ax.set_title(plotTitle)
        ax.set_ylabel("Bubble Mean Intensity")
        ax.set_xlabel("Bubble Number(sorted)")
        ax.legend()
        #TODO append QStrip to this plot-
        # appendSaveImg(6,getPlotImg())
        if(plotShow):
            plt.show()
    return thr1

# from matplotlib.ticker import MaxNLocator
# def plotArray(QVals, plotTitle, sort = False, plot=True ):
#     f, ax = plt.subplots()
#     if(sort):
#         QVals = sorted(QVals)
#     ax.bar(range(len(QVals)),QVals);
#     ax.set_title(plotTitle)
#     ax.set_ylabel("Values")
#     ax.set_xlabel("Position")
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     if(plot):
#         plt.show()
#     # else: they will call this
#     #     appendSaveImg(appendImgLvl,getPlotImg())

def saveImg(path, final_marked):
    print('Saving Image to '+path)
    cv2.imwrite(path,final_marked)


def readResponse(template, image, name, savedir=None, autoAlign=False):
    global clahe

    try:
        img = image.copy()
        origDim = img.shape[:2]
        # print("noCropping dim", origDim)
        img = resize_util(img,template.dims[0],template.dims[1])
        # print("Resized dim", img.shape[:2])

        if(img.max()>img.min()):
            img = normalize_util(img)
        # Processing copies
        transp_layer = img.copy()
        final_marked = img.copy()
        # putLabel(final_marked,"Crop Size: " + str(origDim[0])+"x"+str(origDim[1]) + " "+name, size=1)
        
        morph = img.copy() #
        appendSaveImg(3,morph)

        # TODO: evaluate if CLAHE is really req
        if(autoAlign==True):
            # Note: clahe is good for morphology, bad for thresholding
            morph = clahe.apply(morph) 
            appendSaveImg(3,morph)
            # Remove shadows further, make columns/boxes darker (less gamma)
            morph = adjust_gamma(morph,GAMMA_LOW)
            ret, morph = cv2.threshold(morph,220,220,cv2.THRESH_TRUNC)
            morph = normalize_util(morph)
            appendSaveImg(3,morph)
            if(showimglvl>=4):
                show("morph1",morph,0,1)

        # Overlay Transparencies
        alpha = 0.65
        alpha1 = 0.55

        boxW,boxH = template.bubbleDims
        lang = ['E','H']
        OMRresponse={}

        multimarked,multiroll=0,0

        blackVals=[0]
        whiteVals=[255]

        if(showimglvl>=5):
            allCBoxvals={"Int":[],"Mcq":[]}#"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}
            qNums={"Int":[],"Mcq":[]}#,"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}


        ### Find Shifts for the QBlocks --> Before calculating threshold!
        if(autoAlign == True):
            # print("Begin Alignment")
            # Open : erode then dilate
            # Vertical kernel 
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
            morph_v = cv2.morphologyEx(morph, cv2.MORPH_OPEN, v_kernel, iterations=3)
            ret, morph_v = cv2.threshold(morph_v,200,200,cv2.THRESH_TRUNC)
            morph_v = 255 - normalize_util(morph_v)
            
            if(showimglvl>=3):
                show("morphed_vertical",morph_v,0,1)
            
            # show("morph1",morph,0,1)
            # show("morphed_vertical",morph_v,0,1)
            
            appendSaveImg(3,morph_v)

            morphTHR = 60 # for Mobile images
            # morphTHR = 40 # for scan Images
            # best tuned to 5x5 now
            _, morph_v = cv2.threshold(morph_v,morphTHR,255,cv2.THRESH_BINARY)
            morph_v = cv2.erode(morph_v,  np.ones((5,5),np.uint8), iterations = 2)
            
            appendSaveImg(3,morph_v)
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
            # OUTPUT : each QBlock.shift is updated
            for QBlock in template.QBlocks:
                s,d = QBlock.orig, QBlock.dims
                # internal constants - wont need change much
                #TODO - ALIGN_STRIDE would depend on template's Dimensions
                ALIGN_STRIDE, MATCH_COL, ALIGN_STEPS = 1, 5, int(boxW * 2 / 3)
                shift, steps = 0, 0
                THK = 3
                while steps < ALIGN_STEPS:
                    L = np.mean(morph_v[s[1]:s[1]+d[1],s[0]+shift-THK:-THK+s[0]+shift+MATCH_COL])
                    R = np.mean(morph_v[s[1]:s[1]+d[1],s[0]+shift-MATCH_COL+d[0]+THK:THK+s[0]+shift+d[0]])
                    
                    # For demonstration purposes-
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
                # print("Aligned QBlock: ",QBlock.key,"Corrected Shift:", QBlock.shift,", Dimensions:", QBlock.dims, "orig:", QBlock.orig,'\n')
            # print("End Alignment")
        
        final_align = None
        if(showimglvl>=2):
            initial_align = drawTemplateLayout(img, template, shifted=False)
            final_align = drawTemplateLayout(img, template, shifted=True, draw_qvals=True)
            # appendSaveImg(4,mean_vals)
            appendSaveImg(2,initial_align)
            appendSaveImg(2,final_align)
            appendSaveImg(5,img)
            if(autoAlign == True):
                final_align = np.hstack((initial_align, final_align))

        # Get mean vals n other stats
        allQVals, allQStripArrs, allQStdVals =[], [], []
        totalQStripNo = 0
        for QBlock in template.QBlocks:
            QStdVals=[]
            for qStrip, qBoxPts in QBlock.traverse_pts:
                QStripvals = []
                for pt in qBoxPts:
                    # shifted
                    x,y = (pt.x + QBlock.shift,pt.y)
                    rect = [y,y+boxH,x,x+boxW]
                    QStripvals.append(cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0])
                QStdVals.append(round(np.std(QStripvals),2))
                allQStripArrs.append(QStripvals)
                # _, _, _ = getGlobalThreshold(QStripvals, "QStrip Plot", plotShow=False, sortInPlot=True)
                # hist = getPlotImg()
                # show("QStrip "+qBoxPts[0].qNo, hist, 0, 1)
                allQVals.extend(QStripvals)
                # print(totalQStripNo, qBoxPts[0].qNo, QStdVals[len(QStdVals)-1])
                totalQStripNo+=1
            allQStdVals.extend(QStdVals)
        # print("Begin getGlobalThresholdStd")
        globalStdTHR, jstd_low, jstd_high = getGlobalThreshold(allQStdVals)#, "Q-wise Std-dev Plot", plotShow=True, sortInPlot=True)
        # print("End getGlobalThresholdStd")
        # print("Begin getGlobalThreshold")
        # plt.show()
        # hist = getPlotImg()
        # show("StdHist", hist, 0, 1)
        
        #Note: Plotting takes Significant times here --> Change Plotting args to support showimglvl
        globalTHR, j_low, j_high = getGlobalThreshold(allQVals)#, "Mean Intensity Histogram", plotShow=True, sortInPlot=True)
        
        # TODO colorama
        print("Thresholding:\t globalTHR: ",round(globalTHR,2),"\tglobalStdTHR: ",round(globalStdTHR,2),"\t(Looks like a Xeroxed OMR)" if(globalTHR == 255) else "")
        # plt.show()
        # hist = getPlotImg()
        # show("StdHist", hist, 0, 1)
        
        # print("End getGlobalThreshold")

        # if(showimglvl>=1):
        #     hist = getPlotImg()
        #     show("Hist", hist, 0, 1)
        #     appendSaveImg(4,hist)
        #     appendSaveImg(5,hist)
        #     appendSaveImg(2,hist)
        # name,


        perOMRThresholdAvg, totalQStripNo, totalQBoxNo = 0, 0, 0
        for QBlock in template.QBlocks:
            blockQStripNo = 1 # start from 1 is fine here
            shift=QBlock.shift
            s,d = QBlock.orig, QBlock.dims
            key = QBlock.key[:3]
            # cv2.rectangle(final_marked,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),CLR_BLACK,3)
            for qStrip, qBoxPts in QBlock.traverse_pts:
                # All Black or All White case        
                noOutliers = allQStdVals[totalQStripNo] < globalStdTHR
                # print(totalQStripNo, qBoxPts[0].qNo, allQStdVals[totalQStripNo], "noOutliers:", noOutliers)
                perQStripThreshold = getLocalThreshold(qBoxPts[0].qNo, allQStripArrs[totalQStripNo], 
                    globalTHR, noOutliers, 
                    "Mean Intensity Histogram for "+ key +"."+ qBoxPts[0].qNo+'.'+str(blockQStripNo), 
                    # None,
                    # "q15.1" in (qBoxPts[0].qNo+'.'+str(blockQStripNo)) or 
                    showimglvl>=6)
                # print(qBoxPts[0].qNo,key,blockQStripNo, "THR: ",round(perQStripThreshold,2))
                perOMRThresholdAvg += perQStripThreshold
                

                # if(
                #     0  
                #     # or "q17" in (qBoxPts[0].qNo) 
                #     # or (qBoxPts[0].qNo+str(blockQStripNo))=="q15" 
                #  ):
                #     st, end = qStrip
                #     show("QStrip: "+key+"-"+str(blockQStripNo), img[st[1] : end[1], st[0]+shift : end[0]+shift],0)

                for pt in qBoxPts:
                    # shifted
                    x,y = (pt.x + QBlock.shift,pt.y)
                    boxval0 = allQVals[totalQBoxNo]
                    detected = perQStripThreshold > boxval0
                    
                    #TODO: add an option to select PLUS SIGN 
                    # extra_check_rects = []
                    # # [y,y+boxH,x,x+boxW]
                    # for rect in extra_check_rects:
                    #     # Note: This is NOT pixel-based thresholding, It is boxed mean-thresholding
                    #     boxval = cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0]
                    #     if(perQStripThreshold > boxval):
                    #         # for critical analysis
                    #         boxval0 = max(boxval,boxval0)
                    #         detected=True
                    #         break;

                    if (detected):
                        cv2.rectangle(final_marked,(int(x+boxW/12),int(y+boxH/12)),(int(x+boxW-boxW/12),int(y+boxH-boxH/12)), CLR_DARK_GRAY, 3)
                    else:
                        cv2.rectangle(final_marked,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_GRAY,-1)

                    # TODO Make this part useful! (Abstract visualizer to check status)
                    if (detected):
                        q, val = pt.qNo, str(pt.val)
                        cv2.putText(final_marked,val,(x,y),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(20,20,10),int(1+3.5*TEXT_SIZE))
                        # Only send rolls multi-marked in the directory
                        multimarkedL = q in OMRresponse
                        multimarked = multimarkedL or multimarked
                        OMRresponse[q] = (OMRresponse[q] + val) if multimarkedL else val
                        multiroll = multimarkedL and 'roll' in str(q)
                        blackVals.append(boxval0)
                    else:
                        whiteVals.append(boxval0)

                    totalQBoxNo+=1
                    # /for qBoxPts
                # /for qStrip

                if( showimglvl>=5):
                    if(key in allCBoxvals):
                        qNums[key].append(key[:2]+'_c'+str(blockQStripNo))
                        allCBoxvals[key].append(allQStripArrs[totalQStripNo])
                
                blockQStripNo += 1
                totalQStripNo += 1
            # /for QBlock
        if(totalQStripNo==0):
            print("\n\t UNEXPECTED Template Incorrect Error: totalQStripNo is zero! QBlocks: ",TEMPLATE.QBlocks)
            exit(7)
        perOMRThresholdAvg /= totalQStripNo
        perOMRThresholdAvg = round(perOMRThresholdAvg,2)
        # Translucent
        cv2.addWeighted(final_marked,alpha,transp_layer,1-alpha,0,final_marked)
        # Box types
        if( showimglvl>=5):
            # plt.draw()
            f, axes = plt.subplots(len(allCBoxvals),sharey=True)
            f.canvas.set_window_title(name)
            ctr=0
            typeName={"Int":"Integer","Mcq":"MCQ","Med":"MED","Rol":"Roll"}
            for k,boxvals in allCBoxvals.items():
                axes[ctr].title.set_text(typeName[k]+" Type")
                axes[ctr].boxplot(boxvals)
                # thrline=axes[ctr].axhline(perOMRThresholdAvg,color='red',ls='--')
                # thrline.set_label("Average THR")
                axes[ctr].set_ylabel("Intensity")
                axes[ctr].set_xticklabels(qNums[k])
                # axes[ctr].legend()
                ctr+=1
            # imshow will do the waiting
            plt.tight_layout(pad=0.5)
            plt.show()

        if(showimglvl>=3 and final_align is not None):
            final_align = resize_util_h(final_align,int(display_height))
            show("Template Alignment Adjustment", final_align, 0, 0)# [final_align.shape[1],0])
        
        # TODO: refactor "type(savedir) != type(None) "
        if (saveMarked and type(savedir) != type(None) ):
            if(multiroll):
                savedir = savedir+'_MULTI_/'
            saveImg(savedir+name, final_marked)

        if(showimglvl>=1):
            # final_align = resize_util_h(final_align,int(display_height))
            # show("Final Alignment : "+name,final_align,0,0)
            show("Final Marked Bubbles : "+name,resize_util_h(final_marked,int(display_height*1.3)),1,1)

        appendSaveImg(2,final_marked)

        # saveImgList[3] = [hist, final_marked]

        for i in range(saveimglvl):
            saveOrShowStacks(i+1, name, savedir)

        return OMRresponse,final_marked,multimarked,multiroll

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error from readResponse: ",e)
        print(exc_type, fname, exc_tb.tb_lineno)

def saveOrShowStacks(key, name, savedir=None,pause=1):
    global saveImgList
    if(saveimglvl >= int(key) and saveImgList[key]!=[]):
        result = np.hstack(tuple([resize_util_h(img,uniform_height) for img in saveImgList[key]]))
        result = resize_util(result,min(len(saveImgList[key])*uniform_width//3,int(uniform_width*2.5)))
        if (type(savedir) != type(None)):
            saveImg(savedir+'stack/'+name+'_'+str(key)+'_stack.jpg', result)
        else:
            show(name+'_'+str(key),result,pause,0)
            
"""
# qmin, qmax, qmean, qstd = round(np.min(QVals),2), round(np.max(QVals),2), round(np.mean(QVals),2), round(np.std(QVals),2)
# gap = (qmax - qmin)
# print("qmean",qmean, "qstd", qstd)
# gstd = 0
# GVals = [round(abs(q-qmean),2) for q in QVals]
# gmean, gstd = round(np.mean(GVals),2), round(np.std(GVals),2)
# if(plotTHR):
#     print("qstd",qstd, "gstd", gstd,"Gaps in gvals",sorted([round(abs(g-gmean),2) for g in GVals],reverse=True))                

"""
