"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""
import os
import glob
# 1: View without saving, 2: Save files without showing
review = 0
dir_glob ='backupOMRs/*/*/*'
if(os.sep=="\\"):
	# eg backupOMRs\Agra_1057\HE\Normal>Agra_HE_0054.tif
	dir_glob ='backupOMRs\\*\\*\\*'

counter = 0
for folder in glob.iglob(dir_glob):
	try:
		for filename in os.listdir(folder):
			if filename.endswith(".tif"):
				filepath = folder+os.sep+filename
				counter += 1
				if(review):
					print(counter, filepath)
				else:
					newx = "convertedOMRs"+folder[len("backupOMRs"):]
					if(not os.path.exists(newx)):
						print("Making dirs:",newx)
						os.makedirs(newx)
					else:
						print("Dir already exists:",newx)

					newfilepath = newx +os.sep+filename[:-4]+".jpg"
					#  -resize 50%
					if(os.system("convert "+filepath+' '+newfilepath)==0):
						print(counter, newfilepath)
					else:
						print(counter,"Failed to convert: ", filename)
	except NotADirectoryError:
		pass

print("Total Tiff files: " + str(counter))