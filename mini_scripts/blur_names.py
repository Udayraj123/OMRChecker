import glob
import cv2

# 1: View without saving, 2: Save files without showing
review=0;

dir_glob ='images/OMR_Files(copy)'+'/*/*/*/*.jpg'
u_width=500
fac1=0.65
fac2=0.95

def waitQ():
    while(0xFF & cv2.waitKey(1) != ord('q')):pass
    cv2.destroyAllWindows()

show_count=0
def show(img,title=""):     
    global show_count
    if(title==""):
        show_count+=1
        title="Image "+str(show_count)
        
    cv2.imshow(title,img)
    waitQ()

def coverNames(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	h,w = img.shape
	# dbl blur
	img[int(h*fac1):int(h*fac2),:] = cv2.blur(img[int(h*fac1):int(h*fac2),:],(35,35))
	return img;

allOMRs= glob.iglob(dir_glob)
for filepath in allOMRs:
	print (filepath)
	img = coverNames(cv2.imread(filepath));
	if(review):
		h,w=img.shape
		show(cv2.resize(img,(u_width,int(h*u_width/w))))
	else:
		cv2.imwrite(filepath,img)