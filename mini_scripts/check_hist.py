import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 1: View without saving, 2: Save files without showing
review=0;

# dir_glob ='../src/images/OMR_Files'+'/*/*/*/*.jpg'
dir_glob ='hist_inputs'+'/*.jpg'
u_width=1000

def waitQ():
    while(0xFF & cv2.waitKey(1) != ord('q')):pass
    cv2.destroyAllWindows()

show_count=0
def show(img,title="",wait=True):     
    global show_count
    if(title==""):
        show_count+=1
        title="Image "+str(show_count)
        
    cv2.imshow(title,img)
    if(wait):
    	waitQ()

def plot_hist(img):

## cv2 method
	# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
	# BINS is represented by the term histSize
	# For particular region, use the mask argument 
# eg
	# hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    # cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


## np method
	hist,bins=np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_norm = cdf * hist.max()/cdf.max()
	plt.plot(cdf_norm,color='g')
	plt.hist(img.flatten(),256,[0,256],color='b')
	plt.xlim([0,256])
	plt.legend(('cdf','histogram'),loc='upper left')
	plt.show()

from sklearn.preprocessing import minmax_scale
def normalize(img):
	img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	# img = minmax_scale(img,feature_range=(0,255),copy=False)

	"""
	Time to write from scratch!
	"""
	# MAX=np.max(img);
	# MIN=np.min(img);
	# for i in range(len(img)):
	# 	for j in range(len(img[0])):
	# 		img[i][j] = ((img[i][j]-MIN)* 255)/(MAX-MIN) 

def stitch(img1,img2):
	if(img1.shape!=img2.shape):
		print("Can't stitch different sized images")
		return None
	return np.concatenate((img1,img2),axis=1)

allOMRs= glob.iglob(dir_glob)
for filepath in allOMRs:
	print (filepath)
	img=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE= 0 reads in grayscale
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	h,w=img.shape
	img = cv2.resize(img,(u_width,int(h*u_width/w)))
	orig=img.copy()
	if(review):
		show(img,wait=False)
		plot_hist(img);
		normalize(img);
		show(img)
		plot_hist(img);
	else:
		normalize(img);
		filename=filepath[filepath.rindex("/")+1:]
		img=stitch(orig,img)
		cv2.imwrite("hist_outputs/"+filename,img)
