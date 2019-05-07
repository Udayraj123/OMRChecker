from imutils import perspective
from imutils import contours
from matplotlib import pyplot as plt
from random import random
import numpy as np
import imutils
import cv2

def resize_util(img, u_width, u_height=None):
    if u_height == None:
        h,w=img.shape[:2]
        u_height = int(h*u_width/w)        
    return cv2.resize(img,(u_width,u_height))
def waitQ():
 	while(cv2.waitKey(1)& 0xFF != ord('q')):pass
 	cv2.destroyAllWindows()

u_width = 800

# orig = cv2.cvtColor(resize_util(cv2.imread('Screenshot from 2019-03-21 04-14-00.png'), u_width), cv2.COLOR_BGR2GRAY)
orig = cv2.cvtColor(resize_util(cv2.imread('test.jpg'), u_width), cv2.COLOR_BGR2GRAY)
# orig = cv2.cvtColor(resize_util(cv2.imread('cv2_dft_xfiles.png'), u_width), cv2.COLOR_BGR2GRAY)

img = orig.copy()

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = rows//2 , cols//2     # center
STRENGTH = 40
RSTRENGTH = 20

for shift in range(-4*STRENGTH, 4*STRENGTH, (STRENGTH*2)//3):
	# create a mask first, center square is 1, remaining all zeros
	# mask = np.zeros((rows, cols, 2), np.uint8)
	# mask[crow-STRENGTH:crow+STRENGTH, ccol-STRENGTH:ccol+STRENGTH] = 1
	mask = np.ones((rows, cols, 2), np.uint8)
	# mask[crow-STRENGTH:crow+STRENGTH, ccol-STRENGTH:ccol+STRENGTH] = 0
	# mask[crow+shift:shift+crow+3*STRENGTH, ccol-STRENGTH:ccol+STRENGTH] = 0
	mask[crow-STRENGTH:crow+STRENGTH, ccol+shift:shift+ccol+3*STRENGTH] = 0

	# apply mask and inverse DFT
	dft_eff = dft*mask
	# dft_eff = dft_shift*mask # <- Rotated

	f_ishift = np.fft.ifftshift(dft_eff)
	img_back = cv2.idft(f_ishift)
	img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

	plt.figure(figsize=(20,10))

	plt.subplot(221),plt.imshow(img, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(222),plt.imshow(img_back, cmap = 'gray')
	# plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
	plt.title('Recovered'), plt.xticks([]), plt.yticks([])
	magnitude_spectrum = RSTRENGTH*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
	plt.subplot(223),plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Spectrum'), plt.xticks([]), plt.yticks([])
	magnitude_spectrum = RSTRENGTH*np.log(cv2.magnitude(dft_eff[:,:,0],dft_eff[:,:,1]))
	plt.subplot(224),plt.imshow(magnitude_spectrum, cmap = 'gray')
	# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Effect'), plt.xticks([]), plt.yticks([])

	plt.show()                