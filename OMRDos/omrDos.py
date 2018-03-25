
# coding: utf-8

# In[1]:

import cv2
from random import randint
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

"""
Pseudo code- Sound and Simple
Image processing to :
    Scale and rotate image correctly
    Detect responses correctly
    Return the response to each question in a list.
Pure python to :
    Check the answers
    Provide a GUI ? (tkinker)
    
But first, get going with a static template and read the color
"""


# In[3]:

def readResponse(d1,d2,img):
#     box = img[d1[0]:d2[0],d1[1]:d2[1]]
#     meanOfRows = np.mean(img,axis=0)
#     mean = np.mean(meanOfRows,axis=0)
    # Get the average color around the coordinates. If
    l = abs(d1[0]-d2[0])
    h = abs(d1[1]-d2[1])
    mask = np.zeros((h, l), np.uint8)
    cv2.circle(mask,d1, 10, (255, 255, 555), -1)
    # cv2.rectangle(mask,d1,d2,(0,255,0),2)
    mean_color = cv2.mean(img, mask)
  
    return mean_color

d1= (424,562)
d2= (432,567)
# # In[5]:

img = cv2.imread('OMRSample.jpg',cv2.CV_8UC1) #IMREAD_COLOR/UNCHANGED
mask = np.zeros(img.shape, np.uint8)
cv2.rectangle(img,d1,d2,(100,255,100),2)
# print(readResponse(d1,d2,img))
cv2.imshow('imageWindow',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # In[ ]:

# #underscore consumes the variable - retval here
# img = cv2.imread('OMRSample.jpg',cv2.CV_8UC1) #IMREAD_COLOR/UNCHANGED
# #retval, threshold = cv2.threshold(img,signumThrVal,maxVal,cv2.THRESH_BINARY_INV)
# threshold=[]
# _, t = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
# threshold.append(t)
# threshold.append(cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1))
# ret, t = cv2.threshold(img,126,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# threshold.append(t)
# cv2.imshow('imageWindow',img)

# for i in range(0,3):
#     cv2.imshow('th'+str(i),threshold[i]);
#     cv2.moveWindow('th'+str(i),1280,0)


# # In[3]:

# # ret=0
# # cap = cv2.VideoCapture(0);
# # ret,_ = cap.read()
# # while(ret==True):
# #     ret,frame = cap.read();
# #     cv2.imshow('feed',frame)
    
# #     laplacian = cv2.Laplacian(frame,cv2.CV_64F) #CV_64F is just a datatype
# #     #emboss x & y
# #     sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
# #     sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
# #     cv2.imshow('lapla',laplacian);    cv2.moveWindow('lapla',680,0)
# #     cv2.imshow('sobelx',sobelx);    cv2.moveWindow('sobelx',0,500)
# #     cv2.imshow('sobely',sobely);    cv2.moveWindow('sobely',680,500)
# #     if (cv2.waitKey(1) & 0xFF == ord('q')):#on pressing q
# #         break;

# #     #These laplacians can be used for building the edge detectors,
# #     #But they are all built in ! E.g.  Canny Edges -
    
# #     edges = cv2.Canny(frame,50,150)
# #     cv2.imshow('edges',edges);    cv2.moveWindow('edges',1200,500)
        
# #         #cv2.waitKey() returns a 32 Bit integer value (dependent on the platform).
# # print('yup')
# # cv2.destroyAllWindows()


# # In[5]:

# # img1=cv2.imread('b.jpg')
# # img1 = cv2.resize(img1,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
# # img2=img1.copy()

# # cv2.imshow('1',img1)
# # img2 = cv2.pyrUp(img1)
# # img2 = cv2.pyrUp(img2)
# # img2 = cv2.pyrDown(img2)
# # img2 = cv2.pyrDown(img2)
# # cv2.imshow('2',img2)


# # In[ ]:

# img1=cv2.imread('OMRSample.jpg')
# x1,y1=10,44
# x2,y2=18,566
# img2=img1.copy()
# # pts = np.array([(y1,x1),(y2,x2)])
# # r = cv2.boundingRect(pts)
# # cv2.imwrite('template.jpg', img2[r[0]:r[0]+r[2], r[1]:r[1]+r[3]])
# template=cv2.imread('template.jpg')

# h,w,_=template.shape

# res = cv2.matchTemplate(img1,template,cv2.TM_CCOEFF_NORMED)
# _,_,minLoc,maxLoc = cv2.minMaxLoc(res)
# threshold = 0.95
# loc = np.where(res>=threshold) # locate values greater than 0.8. Just like array filter
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img1,pt,(pt[0]+w,pt[0]+h),(0,255,255),2)

# templ=cv2.imread('roi.png')

# cv2.imshow('template',template)
# cv2.imshow('detected',img1)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # In[15]:

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('OMRSample.jpg',0)
# img2 = img.copy()
# template = cv2.imread('template.jpg',0)
# w, h = template.shape[::-1]

# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv2.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(img,top_left, bottom_right, 0, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
    
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()


# # In[ ]:




# # In[ ]:



