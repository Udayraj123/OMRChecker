import glob
import cv2
import re
kv=1
directory ='KV_OMRs_2017/' if kv else 'OMR_Files_2017/'

allOMRs= glob.iglob(directory+'/*/*/*.jpg')
for filepath in allOMRs:
	finder = re.search(r''+directory+r'/(.*)/(.*)/(.*)/(.*)\.jpg',filepath,re.IGNORECASE)
	citycode = finder.group(1)
	cityname = citycode.split('_')[0]

	squadlang = finder.group(2)
	prerror = finder.group(3)
	filename = finder.group(4)
	if (('_JE_' in filename and squadlang!='JE') or ('_HE_' in filename and squadlang!='HE') or ('_JH_' in filename and squadlang!='JH') or ('_HH_' in filename and squadlang!='HH') ):
		print(filepath)
	# if((cityname.lower() not in filename.lower())):
	# 	print(cityname,'<<<<<<<',filepath)
	# if(('_JH_' in filename) or ('_HH_' in filename)):
		# print('Hindi Files here : '+filepath)
	# cv2.imwrite(filename+'.jpg',cv2.imread(filepath))


	
"""
Hindi Files	- Korba, Gwalior, Gonda, Rajouri

('OMR_Files_2017/Secunderabad_2060/JE/{}{}{}{}Normal/Mainpuri_JE_0045.tif', 'Secunderabad', '<<<<<<<')
('Bhilwara', '<<<<<<<', 'OMR_Files_2017/Bhilwara_5007/JE/Normal/Bhilawara_JE_0030.tif')
('najibabad', '<<<<<<<', 'OMR_Files_2017/najibabad_1147/JH/Normal/KV new tehri town_5135_JE_0003.tif')
('Bhiwadi', '<<<<<<<', 'OMR_Files_2017/Bhiwadi_5039/JH/Normal/amalapuram_JE_0079.tif')
('rewari', '<<<<<<<', 'OMR_Files_2017/rewari_1012/HH/Normal/Thrissur_HE_0001.tif')

"""