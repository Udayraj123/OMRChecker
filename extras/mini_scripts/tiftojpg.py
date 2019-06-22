"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""
import os
import glob
import cv2
# review = 1
review = 0
# backupOMRs\Agra_1057\HE\Normal>Agra_HE_0054.tif
dir_glob ='backupOMRs\\*\\*\\*'
counter = 0
for x in glob.iglob(dir_glob):
	try:
		for filename in os.listdir(x):
			if filename.endswith(".tif"):
				filepath = x+"\\"+filename
				counter += 1
				if(review):
					print(counter, filepath)
				else:
					newx = "convertedOMRs"+x[len("backupOMRs"):]
					if(not os.path.exists(newx)):
						print("Making dirs:",newx)
						os.makedirs(newx)
					else:
						print("Already exists:",newx)

					newfilepath = newx +filename[:-4]+".jpg"
					cv2.imwrite(newfilepath,cv2.imread(filepath))
					# print("System cmd:",os.system(R'"C:\Program Files\ImageMagick-7.0.8-Q16\convert.exe" '+filepath+' '+newfilepath))
					print(counter, newfilepath)
				exit(0)
	except NotADirectoryError:
		pass

print("Total Tiff files: " + str(counter))