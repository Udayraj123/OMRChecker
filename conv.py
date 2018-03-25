import re
import glob
import cv2
fs = glob.iglob('*.tif')
for f in fs:
	fn = re.search('(.*).tif',f).group(1)
	cv2.imwrite(fn+'.jpg',cv2.imread(f))
