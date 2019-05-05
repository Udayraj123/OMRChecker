import os
for x in os.listdir():
	if not x.endswith(".txt") and not x.endswith(".py") and  not x.endswith(".tif"): 
		for y in os.listdir(x):
			for z in os.listdir(x+"/"+y):
				for w in os.listdir(x+"/"+y+"/"+z):
					if w.endswith(".tif"):
						if w[:-4]+".jpg" in os.listdir(x+"/"+y+"/"+z):
							print(">>>>>>>>>>"+x+">"+y+">"+z+">"+w)
							continue
						os.system("convert "+x+"/"+y+"/"+z+"/"+w+" "+x+"/"+y+"/"+z+"/"+w[:-4]+".jpg")
						os.system("rm "+x+"/"+y+"/"+z+"/"+w)