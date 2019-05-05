
import re
import cv2
import os
import glob
from time import localtime,strftime,time

def waitQ():
    while(cv2.waitKey(1)& 0xFF != ord('q')):pass
    cv2.destroyAllWindows()

windowX,windowY = 0,0
windowWidth,windowHeight = 1366,768 
def show(name,img,pause=1,resetpos=None):
    global windowX
    global windowY
    if(type(img)== type(None)):
        print(name," NoneType image to show!")
        if(pause):
            cv2.destroyAllWindows()
        return
    cv2.imshow(name,img)
    print(img.shape)
    w,h = img.shape
    if(resetpos):
        windowX=resetpos[0]
        windowY=resetpos[1]
    cv2.moveWindow(name,windowX,windowY)
    overflowX = windowX+w > windowWidth
    overflowY = windowY+h > windowHeight
    if(overflowX):
        windowX = 0
        if(not overflowY):windowY+=h
    else:windowX+=w
    if(overflowY):windowY = 0
    if(pause):
        waitQ()
kv=0
directory ='OMRs/KV_OMRs_2017/' if kv else 'OMRs/OMR_Files_2017/'

p=int(time())
# TIF
ext = '.tif'
allOMRs= glob.iglob(directory+'*/*/*/*'+ext)
# allOMRs= glob.iglob(directory+'ErrorFiles/*/*.jpg')
counter=1
for filepath in allOMRs:
    counter+=1
    # finder = re.search(directory+r'ErrorFiles/(.*)/(.*)\.jpg',filepath,re.IGNORECASE)
    finder = re.search(r'/.*/(.*)/(.*)/(.*)\.'+ext[1:],filepath,re.IGNORECASE)
    squadlang = finder.group(1)
    filename = finder.group(3)
    xeno = finder.group(2)
    jpg=filepath[:-4]+'.jpg'
    if(not os.path.exists(jpg)):
        filepath_full=os.path.join(os.getcwd(),filepath)
        print(filepath_full)
        img = cv2.imread(filepath_full,cv2.IMREAD_GRAYSCALE)
        # show("image",img)
        cv2.imwrite(jpg,img)
        print(jpg)
    os.remove(filepath)

takentimeconverting=int(time()-p)+1
print('Finished Converting %d files in %d seconds =>  %f sec/File:)' % (counter,takentimeconverting,float(takentimeconverting)/float(counter)))       
