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

u_width = 500

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


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
morph =  255 - cv2.morphologyEx(orig, cv2.MORPH_OPEN, kernel, iterations=3)
_, morph = cv2.threshold(morph,10,255,cv2.THRESH_BINARY)
morph = cv2.erode(morph,  np.ones((3,3),np.uint8), iterations = 1)
cv2.imshow("morph", morph)
waitQ()
# exit(0)


box = [[250,156], [20, 100]]
s,d = box

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
	return centroid

# TODO: save time for sum in original implementation
maxS, shiftM = 0, 0
for shift in range(-d[0]//3,d[0]//3,2):
	window = morph[s[1]:s[1]+d[1],s[0]+shift:s[0]+shift+d[0]]
	h, w = window.shape
	centroid = getCentroid(window)
	# Decide whether to move towards, and how fast
	print((w//2, h//2), '->', centroid)
	
	# if(closestGap > abs(centroid[0] - w//2)):
	# 	closestGap = abs(centroid[0] - w//2)
	# 	closest = shift

	# plotIntensity(window)
	# m,M = window.min(),window.max()
	# window = ((window-m)*255)/(M-m)
	sm = np.sum(abs(window))
	print(shift, sm)
	morph2=morph.copy()
	cv2.rectangle(morph2,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),(0,1,0),2)
	cv2.imshow("Image", morph2)
	waitQ()	
	if(maxS < sm):
		maxS = sm
		shiftM = shift
print("shiftM", shiftM)
morph = orig.copy()
cv2.rectangle(morph,(s[0]+shiftM,s[1]),(s[0]+shiftM+d[0],s[1]+d[1]),(0,1,0),2)
print(morph.shape)
cv2.imshow("Image", morph)
waitQ()