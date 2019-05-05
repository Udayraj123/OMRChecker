import cv2
import glob
import imutils
import unittest
import numpy as np
from utils import *
from template import *
from constants import *
from random import random
from math import sin,cos,pi as PI
from matplotlib import pyplot as plt
# 1: View without saving, else: Save files without showing
review=1;

dir_glob = directory+'*/*/*'+ext
bg_glob = 'images/omrbgs/omrbg*'+ext
# Original scan dimensions: 3543 x 2478
display_width=1000
IMCOUNT=10

def waitQ():
	while(0xFF & cv2.waitKey(1) != ord('q')):pass
	cv2.destroyAllWindows()

def showOrSave(filepath,orig,title="",resize=True,pause=True,forced=False):
	global review,windowX,windowY
	img = resize_util(orig,display_width) if resize else orig
	filename=filepath[filepath.rindex("/")+1:]
	if(review):
		# TODO: TEMPLINE
		windowX,windowY=(0,0)
		show(filename,img,pause)
	elif(forced or pause):
		cv2.imwrite("outputs/"+filename,img)

def testImg(filepath,orig,title="",resize=True):
	filename=filepath[filepath.rindex("/")+1:]
	# images/OMR_Files/4137/HE/Durgapur_HE_04_prsp_13.22_18.78_5.jpg
	finder = re.search(r'.*/.*/.*/(.*)/(.*)\.'+ext[1:],filepath,re.IGNORECASE)
	squad,lang = 'X','X'
	if(finder):
		squadlang = finder.group(1)
		squad,lang = squadlang[0],squadlang[1]
	print("File: ",filename)
	OMRcrop = getROI(filepath,filename,orig)
	newfilename = filename + '_' + filepath.split('/')[-2]
	OMRresponse,retimg,multimarked,multiroll = readResponse(squad,OMRcrop, name = newfilename)
	print("\n\tRead response: ", OMRresponse,'\n')


def warpPts(pts,M_2d):
	if(M_2d.shape[0]!=2):
		print("Warning: warpPts input matrix has invalid shape: ", M_2d.shape)
	# convert to 3d coordinates
	# np.insert(arr, indicesAt, value(s), axis)
	pts2 = np.insert(pts, 2, values=1, axis=1)
	M2 = np.append(M_2d,[[0,0,1]],axis=0)
	pts2 = np.matmul(pts2,M2.T).astype(int)
	pts2 = np.delete(pts2, 2, axis=1)
	return pts2

def warpPts3D(pts,M_3d):
	if(M_3d.shape[0]!=3):
		print("Warning: warpPts3D input matrix has invalid shape: ", M_3d.shape)
	# convert to 4d coordinates
	# print(pts)
	pts2 = np.insert(pts, 2, values=1, axis=1)
	# print(M_3d.shape)
	M_3d = np.insert(M_3d, 3, values=0, axis=1)
	M_3d = np.insert(M_3d, 3, values=[0,0,0,1], axis=0)
	# print(M_3d.shape)
	pts2 = np.matmul(M_3d.T,pts2).astype(int)
	pts2 = np.delete(pts2, 2, axis=1)
	# print(pts2)
	return pts2

def rotateImg(img, pts, i):
	# scale=0.5 can do the zero padding eqv
	h,w = img.shape[:2]
	# Careful about w and h! <-- Sill
	M = cv2.getRotationMatrix2D((w//2,h//2),i,scale=1)
	# third arg is the output image size
	img2 = cv2.warpAffine(img,M,(w,h))
	wp = warpPts(pts,M)
	return img2,wp

def getBoundPts(pts):
	w,h = tuple(np.max(pts,axis=0).astype(int)[:2])
	x,y = tuple(np.min(pts,axis=0).astype(int)[:2])
	return getPts(x,y,w-x,h-y)

def four_point_transform_bound(img, pts):
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

	# img bound pts may lie anywhere(outside or inside) of the warped image bounds
	h, w = img.shape[:2]
	pts_img = getPts(0,0,w,h)
	bound_pts = warpPts3D(pts_img, M)

	warped = cv2.warpPerspective(img, M, (minWidth, maxHeight))

	# return the warped image
	return warped, bound_pts


def rotateLine(p,q,a):
	a *= PI/180
	x0,y0 = q
	x1,y1 = p
	# M = cv2.getRotationMatrix2D((h//2,w//2),i,scale=1)
	p[0] = int(((x1 - x0) * cos(a)) - ((y1 - y0) * sin(a)) + x0);
	p[1] = int(((x1 - x0) * sin(a)) + ((y1 - y0) * cos(a)) + y0);
	return p

def drawPoly(img, pts,color=(255,255,255), thickness=5):
	l = len(pts)
	for i in range(0,l+1):
		cv2.line(img,tuple(pts[(i-1)%l]),tuple(pts[i%l]),color=color, thickness=thickness)

def getPts(x,y,w,h):
	return np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])

def zeroPad(img, padFrac = 0.1):
	h, w = img.shape[:2]
	bg = np.zeros((int((1+2*padFrac)*h),int((1+2*padFrac)*w)), np.uint8)
	# M2: img = cv2.copyMakeBorder(img,u_height//2,u_height//2,display_width//2,display_width//2,cv2.BORDER_CONSTANT, value=(0,0,0))
	x,y,wi,hi = int(w*padFrac),int(h*padFrac), w//2, h//2
	bg[y:(y+h) , x:(x+w)] = img;
	c = [x+wi,y+hi]
	# use list than tuple to support list assignment
	# pts=[[-wi,-hi],[-wi,hi],[wi,hi],[wi,-hi]]
	pts=getPts(x,y,w,h)
	# ^ pts relative to centre
	return bg,c,pts

def contrast(img, gamma=0.5):
	lookUpTable = np.empty((1,256), np.uint8)
	for i in range(256):
		lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
	return cv2.LUT(img, lookUpTable)

def applyBG(img_orig, bg_orig, pts):
	# Done : Make it work for general poly
	# img will have black region outside points
	h,w = img_orig.shape[:2]
	bg = cv2.resize(bg_orig,(w,h))
	mask = np.ones_like(bg)
	mask *= 255;
	cv2.fillConvexPoly(mask,pts,0)
	bg = cv2.bitwise_and(bg,bg,mask = mask)
	mask = np.zeros_like(img_orig)
	cv2.fillConvexPoly(mask,pts,255)
	img = cv2.bitwise_and(img_orig,img_orig,mask = mask)
	# showOrSave("",img)
	# showOrSave("",bg)
	return cv2.addWeighted(bg,1,img,1,0)

	#  Will Work only for rectangles!
	# # print(bg.shape, img.shape)
	# x,y = diag[0]
	# w,h = np.subtract(diag[1],diag[0])
	# # print(x,y,w,h)
	# bg[y:(y+h) , x:(x+w)] = img[y:(y+h) , x:(x+w)];
	# return bg

def warpImg(img,M,outdim=None):
	# warpAffine is basically transformation using matrix!!
	h,w = img.shape[:2]
	pts = getPts(0,0,w,h) #np.float32([[0,0],[h,0],[h,w],[w,0]])
	# Drop translation column
	M2 = np.delete(M, 2, axis=1)
	# Multiply the pts
	pts = np.matmul(pts,M2.T)
	if(not outdim):
		outdim = tuple(np.max(pts,axis=0).astype(int)[:2])
		# h,w = tuple(np.max(pts,axis=0).astype(int)[:2])
		# outdim= (w,h)
	# print(img.shape,outdim)
	img = cv2.warpAffine(img,M,outdim)
	return img, pts

class testImageWarps(unittest.TestCase):

	def setUp(self):
		self.allIMGs=[]
		allOMRs= glob.iglob(dir_glob)
		allBGs= glob.iglob(bg_glob)
		bgs=[]
		for bgpath in allBGs:
			bg  = cv2.imread(bgpath, cv2.IMREAD_GRAYSCALE)
			# bg = cv2.equalizeHist(bg)
			# ^ Dont work well
			# cv2.normalize(bg , bg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			# ^ Dont work
			bg = contrast(bg,gamma=0.3)
			bgs.append(bg)
		self.bgs=bgs
		bglen = len(bgs)
		for i,filepath in enumerate(allOMRs):
			if(i==IMCOUNT):
				break
			img=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
			#img = contrast(img)
			cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			img = resize_util(img,uniform_width_hd)
			self.allIMGs.append((img, i%bglen, filepath))
		print("\nTotal OMRs: ",len(self.allIMGs))

	def testRotation(self):
		for baseimg, bgindex, filepath in self.allIMGs:
			# self.assertEqual(w,display_width, 'width not resized properly!')
			img=baseimg.copy()
			img, c, pts = zeroPad(img,padFrac=0.3)
			# img = applyBG(img,self.bgs[bgindex],pts)
			h,w = img.shape[:2]

			for i in range(-20,20,1):
				img2, wp = rotateImg(img,pts,i)
				# showOrSave(filepath[:-4]+"_rot"+str(i)+ext,imutils.rotate_bound(img,i))
				img2 = applyBG(img2,self.bgs[bgindex],wp)
				# Apply label patch
				putLabel(img2, "Rotation Angle: "+str(i))

				# showOrSave(filepath[:-4]+"_rot"+str(i)+ext,img2)
				testImg(filepath[:-4]+"_rot"+str(i)+ext,img2)

	def testTranslate(self):
		for baseimg, bgindex, filepath in self.allIMGs:
			h,w=baseimg.shape[:2]
			img=baseimg.copy()
			# padding be sufficient for warped points to be within limits
			img, c, pts = zeroPad(img,padFrac=0.3)
			# showOrSave("",applyBG(img,self.bgs[bgindex],pts))

			for i in range(-w//5,w//5+1,w//25): # 100 variations
				for j in range(-h//5,h//5+1,h//25):
			# for i in range(-w//5,w//5+1,w//5):
			# 	for j in range(h//5,h//5+1,h//5):
					if(i and j):
						M=np.float32([[1,0,i],[0,1,j]])
						img2,_ = warpImg(img,M)
						wp = warpPts(pts,M)
						# print (i,j)
						# print(pts, wp)
						# if(wp.min()<0):
						# 	print("Warning: warp out of image!")
						# img2 = applyBG(img,self.bgs[bgindex],pts)
						img2 = applyBG(img2,self.bgs[bgindex],wp)
						# Apply label patch
						img2[:60, :] = 0
						h2,w2=img2.shape[:2]
						# (unequal) quadrants
						# img2[ : , w2//3:w2//3+2] = 0
						# img2[ h2//2:h2//2+2, : ] = 0

						cv2.putText(img2,"Translate Pos: ("+str(i)+","+str(j)+")",(100,50),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(255,255,255),5)
						# showOrSave(filepath[:-4]+"_mov"+str(i)+str(j)+ext,img2)
						#
						testImg(filepath[:-4]+"_mov("+str(i)+","+str(j)+")"+ext,img2)


	def testPerspective(self):
		#  perspective on whole image (including bg)
		for baseimg, bgindex, filepath in self.allIMGs:
			h,w = baseimg.shape[:2]
			img_orig = baseimg.copy()
			img_orig, c, pts = zeroPad(img_orig,padFrac=0.2+random()/4)
			img_orig = applyBG(img_orig,self.bgs[bgindex],pts)
			for thetaBase,thetaDist,thetaWide in [
				(20*random(),5,random()*20),
				# (-20*random(),10,random()*20),
				# (20*random(),15,random()*20),
				# (-20*random(),25,random()*20),
				]:

				# rotate image about midpoint
				# img = imutils.rotate_bound(img_orig, thetaBase)
				img, pts_img = rotateImg(img_orig,pts,thetaBase)
				# img, pts_img = rotateImg(img_orig,pts,0)
				pts1 = getBoundPts(pts_img)
				# drawPoly(img,pts1,color=(100,0,0))

				# Create inverse warp rectangle
				pts2 = pts1.copy()
				pts2[3] = rotateLine(pts2[3],pts2[0],-thetaDist)
				pts2[0] = rotateLine(pts2[0],pts2[3],thetaDist)
				pts2[0] = rotateLine(pts2[0],pts2[1],-thetaWide)
				pts2[3] = rotateLine(pts2[3],pts2[2],thetaWide)

				# Done - draw above as lines on image.
				# cv2.polylines(img,[np.array(pts2, np.int32)],isClosed=True,color=(255,255,255), thickness=10)
				# % is positive in python
				# drawPoly(img,pts_img,color=(100,0,0))
				# drawPoly(img,pts2,color=(100,0,0))
				# showOrSave(filepath,img,pause=False)

				# Apply perspective
				# img = cv2.warpPerspective(img,M,(w,h))
				img, bound_pts = four_point_transform_bound(img,np.array(pts2))
				# drawPoly(img,bound_pts,color=(50,0,0))

				# showOrSave(filepath[:-4]+"_prsp"+ext,img,pause=0)
				testImg(filepath[:-4]+"_prsp"+"_"+str(int(thetaBase))+"_"+str(int(thetaWide))+"_"+str(int(thetaDist))+ext,img)


# Run individual -
# suite = unittest.TestLoader().loadTestsFromName('unit_testing.testImageWarps.testTranslate')
# suite = unittest.TestLoader().loadTestsFromName('unit_testing.testImageWarps.testRotation')
suite = unittest.TestLoader().loadTestsFromName('unit_testing.testImageWarps.testPerspective')

# Run all -
# suite = unittest.TestLoader().loadTestsFromTestCase(testImageWarps)

# 2 is max verbosity offered
unittest.TextTestRunner(verbosity=2).run(suite)


#	#	#	#	#	#	#	#	#	 ROUGH CODEWORKS:	#	#	#	#	#	#	#	#	#

# if run as script and not imported as pkg
# if __name__ == '__main__' :
# 	unittest.main()

# for pts- M1: Can also warp the mask, but its better to keep bg aligned with edges of sheets
# M2: imutils! But will loose track of pts (or will we),https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
# https://stackoverflow.com/questions/32266605/warp-perspective-and-stitch-overlap-images-c#_=__name__
