import glob
import cv2

# 1: View without saving, 2: Save files without showing
review=1;

dir_glob ='*/*/*.jpg'
u_width=800

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

allOMRs= glob.iglob(dir_glob)
for filepath in allOMRs:
	print (filepath)
	img = cv2.imread(filepath);
	h,w,_= img.shape
	img=cv2.resize(img,(u_width,int(h*u_width/w)))
	if(review):
		show(img)
	else:
		cv2.imwrite(filepath,img)