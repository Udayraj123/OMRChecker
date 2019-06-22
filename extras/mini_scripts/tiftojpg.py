"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""
import os
import glob
# review = 1
review = 0
# backupOMRs\Agra_1057\HE\Normal>Agra_HE_0054.tif
dir_glob ='backupOMRs/*/*/*'
counter = 0
for x in glob.iglob(dir_glob):
	try:
		for filename in os.listdir(x):
			if filename.endswith(".tif"):
				filepath = x+"/"+filename
				counter += 1
				if(review):
					print(counter, filepath)
				else:
					newx = "convertedOMRs"+x[len("backupOMRs"):]
					if(not os.path.exists(newx)):
						print("Making dirs:",newx)
						os.makedirs(newx)
					else:
						print("Dir already exists:",newx)

					newfilepath = newx +"/"+filename[:-4]+".jpg"
					#  -resize 50%
					if(os.system("convert "+filepath+' '+newfilepath)==0):
						print(counter, newfilepath)
					else:
						print(counter,"Failed to convert: ", filename)
	except NotADirectoryError:
		pass

print("Total Tiff files: " + str(counter))