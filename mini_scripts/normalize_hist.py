"""
Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

April 2019
"""
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 1: View without saving, 2: Save files without showing
review=1;

# dir_glob ='../src/images/OMR_Files'+'/*/*/*.jpg'
dir_glob ='inputs/hist'+'/*.jpg'
hist_width, u_width = 400, 900

def waitQ():
	while(0xFF & cv2.waitKey(1) != ord('q')):pass
	cv2.destroyAllWindows()

show_count=0
def show(img,title="",pause=True):	
	global show_count
	if(title==""):
		show_count+=1
		title="Image "+str(show_count)
		
	cv2.imshow(title,img)
	if(pause):
		waitQ()

def plot_hist(title, img, plotCode=111,show=False,resize=True):

## cv2 method
	# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
	# BINS is represented by the term histSize
	# For particular region, use the mask argument 
# eg
	# hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
	# cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	img = resize_util(img,hist_width)
## np method
	hist,bins=np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_norm = cdf * hist.max()/cdf.max()
	ax = plt.subplot(plotCode)
	ax.set_title(title)
	ax.plot(cdf_norm,color='g')
	ax.hist(img.flatten(),256,[0,256],color='b')
	ax.set_xlim([0,256])
	ax.legend(('cdf','histogram'),loc='upper left')
	if(show):
		plt.show()

def normalize2(img):
	return cv2.equalizeHist(img)
	# hist,bins = np.histogram(img.flatten(),256,[0,256])
	# cdf = hist.cumsum()
	# cdf_m = np.ma.masked_equal(cdf,0)
	# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	# cdf = np.ma.filled(cdf_m,0).astype('uint8')
	# return cdf[img]

def normalize(img):
	return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)#, dtype=cv2.CV_32F)
	
	# Somehow producing different result, has more noise 
	# from sklearn.preprocessing import minmax_scale
	# img = minmax_scale(img,feature_range=(0,255),copy=False)

	"""
	Time to write from scratch!
	"""
	# MAX=np.max(img);
	# MIN=np.min(img);
	# for i in range(len(img)):
	# 	for j in range(len(img[0])):
	# 		img[i][j] = ((img[i][j]-MIN)* 255)/(MAX-MIN) 
def resize_util(img, u_width, u_height=None):
	if u_height == None:
		h,w=img.shape[:2]
		u_height = int(h*u_width/w)
	return cv2.resize(img,(u_width,u_height))

def stitch(img1,img2):
	if(img1.shape[0]!=img2.shape[0]):
		print("Can't stitch different sized images")
		return None
	# np.hstack((img1,img2)) does this!
	return np.concatenate((img1,img2),axis=1)

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

allOMRs= glob.iglob(dir_glob)
""" 
In CLAHE, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.

use of CLAHE is justified as there is too many white points in whole image, making AHE ineffective
The real balancing of these white points is done using truncated theshold followed by normalization

Now difference in normalization methods :
cv2.normalize makes the cdf more like gaussian rising,
cv2.equalizeHist makes the cdf more like line ==> dark points become too dark
""" 
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
for filepath in allOMRs:
	print (filepath)
	img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE= 0 reads in grayscale
	img = resize_util(img,u_width)
	# img = adjust_gamma(img,2.5)
	img = cv2.GaussianBlur(img,(5,5),0)
	orig = img.copy()
	if(review):
		img1 = normalize(img);
		img1 = adjust_gamma(img1,0.35)
		ret, img1 = cv2.threshold(img1,220,255,cv2.THRESH_TRUNC)
		img1 = normalize(img1);

		img2 = clahe.apply(img)	
		img2 = adjust_gamma(img2,1.5)
		ret, img2 = cv2.threshold(img2,220,255,cv2.THRESH_TRUNC)
		img2 = normalize(img2);

		img = stitch(orig,stitch(img1,img2))
		img = resize_util(img,1700)
		show(img)
		plot_hist("Orignal",orig,221)
		plot_hist("Normalize",img1,222)
		plot_hist("Clahe+Normalize",img2,223)
		plt.show()
	else:
		img = normalize(img);
		filename=filepath[filepath.rindex("/")+1:]
		img = stitch(orig,img)
		cv2.imwrite("outputs/hist/"+filename,img)
