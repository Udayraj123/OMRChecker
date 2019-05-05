from imutils import perspective
from imutils import contours
from matplotlib import pyplot as plt
from random import random
import numpy as np
import imutils
import cv2

def resize_util(img, u_width, u_height=None):
    if u_height == None:
        h,w=img.shape[:2]
        u_height = int(h*u_width/w)        
    return cv2.resize(img,(u_width,u_height))
def waitQ():
 	while(cv2.waitKey(1)& 0xFF != ord('q')):pass
 	cv2.destroyAllWindows()

u_width = 900

orig = cv2.cvtColor(resize_util(cv2.imread('test.jpg'), u_width), cv2.COLOR_BGR2GRAY)
# gray = orig.copy()
# Make dark pixels darker, light ones lighter
# > Square the Image
# gray = np.uint16(gray)
# gray **= 2
# gray = gray / 255
# erode operation takes MIN over kernel

# gray = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# cv2.imshow("Input", orig)
# closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
# waitQ()
# 7x2 with 3 iterations seems best.
# for j in range(5,8):
# 	for i in range(5,8):
# 		for it in range(1,5):
# 			# blur = cv2.GaussianBlur(orig, (5,5), 0)
# 			erode = cv2.erode(orig, kernel=np.ones((i,j)),iterations=it)
# 			cv2.imshow("erode %sx%s (%s it)" % (i,j,it), erode)
# 			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (j, i))
# 			# erode then dilate!
# 			morph =  cv2.morphologyEx(orig, cv2.MORPH_OPEN, kernel, iterations=it)
# 			cv2.imshow("morph %sx%s" % (i,j), morph)
# 			cv2.moveWindow("morph %sx%s" % (i,j),u_width,u_width)
# 			waitQ()
# # (_, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
morph =  255 - cv2.morphologyEx(orig, cv2.MORPH_OPEN, kernel, iterations=3)
_, morph = cv2.threshold(morph,10,255,cv2.THRESH_BINARY)
morph = cv2.erode(morph,  np.ones((3,3),np.uint8), iterations = 2)
# exit(0)


# box = np.array([[250,156], [20, 100]] )
box = np.array([[-7+450,280], [36, 180]] )
s,d = box
morph2=morph.copy()
cv2.rectangle(morph2,(s[0],s[1]),(s[0]+d[0],s[1]+d[1]),(100,101,10),2)
cv2.imshow("morph", morph2)
waitQ()

def getCentroid(window):
    centroid = np.array([0,0], np.uint32)
    ctr = 1
    for y in range(len(window)):
        for x in range(len(window[0])):
            centroid[0] += x * window[y,x]
            centroid[1] += y * window[y,x]
            ctr += window[y,x]
    centroid[0] //= ctr
    centroid[1] //= ctr
    h, w = window.shape
    ax = np.array([[x for x in range(w)]] * h)
    ay = np.array([[y]* w for y in range(h)])
    centroid2 = [np.average(ax,weights = window), np.average(ay,weights = window)]
    print(centroid, "==", centroid2)
    return centroid

closestGap, shiftM = 100, 0
equals = []
MATCH_COL=5
THK = 0
shift = 0
while abs(shift)<12:
	window = morph[s[1]-THK:s[1]+THK+d[1],s[0]-THK+shift:s[0]+THK+shift+d[0]]
	# window = morph[s[1]:s[1]+d[1],s[0]+shift:s[0]+shift+d[0]]
	# h, w = window.shape
	# s is width then height here.
	lw = morph[s[1]-THK:s[1]+THK+d[1],s[0]-THK+shift:s[0]+THK+shift+MATCH_COL]
	rw = morph[s[1]-THK:s[1]+THK+d[1],s[0]-THK+shift-MATCH_COL+d[0]:s[0]+THK+shift+d[0]]
	cv2.imshow("window", window)
	# cv2.imshow("lw", lw)
	# cv2.imshow("rw", rw)
	L = np.mean(lw)
	R = np.mean(rw)
	print(shift, L, R)
	morph2=morph.copy()
	cv2.rectangle(morph2,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),(100,101,10),2)
	cv2.imshow("Image", morph2)
	waitQ()	
	LW,RW= L > 100, R > 100
	if(LW):
		if(RW):
			shiftM = shift
			break
		else:
			shift -= 1
	else:
		if(RW):
			shift += 1
		else:
			shiftM = shift
			break
	
	# centroid = getCentroid(window)
	# centre=(w/2, h/2)
	# # Decide whether to move towards, and how fast
	# print(centre, '->', centroid)
	# if(closestGap >= abs(centroid[0] - centre[0])):
	# 	closestGap = abs(centroid[0] - centre[0])
	# 	if(closestGap==0):
	# 		equals.append(shift)
		# closest = shift

	# plotIntensity(window)
	# m,M = window.min(),window.max()
	# window = ((window-m)*255)/(M-m)
	# sm = np.sum(abs(window))
	# print(shift, sm)
	# if(maxS < sm):
	# 	maxS = sm
	# 	shiftM = shift

# shiftM = equals[(len(equals)-1)//2] if (closestGap==0) else closest
print("shiftM", shiftM)
morph = orig.copy()
cv2.rectangle(morph,(s[0]+shiftM,s[1]),(s[0]+shiftM+d[0],s[1]+d[1]),(0,1,0),2)
print(morph.shape)
cv2.imshow("Image", morph)
waitQ()