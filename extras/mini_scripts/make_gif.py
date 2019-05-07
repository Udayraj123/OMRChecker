"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""
import glob
from PIL import Image
in_dir = '../../outputs/checkedOMRs/'
im1 = Image.open("inputs/gif/gif_start.jpg")
GAP = 200 #ms
for suffix1 in ["JE/","HE/","JH/","HH/"]:
	for suffix2 in ["","_MULTI_","_BADSCAN_"]:
		dir_glob =in_dir+suffix1 + suffix2+'/*.jpg'
		allOMRs= list(glob.iglob(dir_glob))
		if(len(allOMRs)):
			filename = "outputs/gif/checking_"+suffix1[:-1] + suffix2+".gif"
			im1.save(filename, save_all=True, append_images=[Image.open(filepath) for filepath in allOMRs], duration=GAP*(2 if len(suffix2) else 1), loop=0)
			print("Saved : "+filename)
		# else:
		# 	print("Empty glob: "+dir_glob)
