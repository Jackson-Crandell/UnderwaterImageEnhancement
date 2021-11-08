import dippykit as dip
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

img_good = cv2.imread('/home/jcrandell/Desktop/ECE4803/Project/image_dehaze/image/fish_good.jpg')
img_bad = cv2.imread('/home/jcrandell/Desktop/ECE4803/Project/image_dehaze/image/fish_bad.jpg')
img_recovered = cv2.imread('/home/jcrandell/Desktop/ECE4803/Project/image_dehaze/image/fish_sceneRadiance_Enhance.png')

print("Good vs. Recovered\n")
mse = dip.metrics.MSE(img_good,img_recovered)
psnr = dip.metrics.PSNR(img_good,img_recovered)
ssim, im_ssim = dip.metrics.SSIM(img_good,img_recovered,win_size = 3)
psnr = dip.metrics.PSNR(img_good,img_recovered,np.max(img_good))
print("MSE", mse, "\nPSNR: ", psnr, "\nSSIM", ssim)

print("Good vs. Bad\n")
mse = dip.metrics.MSE(img_good,img_bad)
psnr = dip.metrics.PSNR(img_good,img_bad)
ssim, im_ssim = dip.metrics.SSIM(img_good,img_bad,win_size = 3)
psnr = dip.metrics.PSNR(img_good,img_bad,np.max(img_good))
print("MSE", mse, "\nPSNR: ", psnr, "\nSSIM", ssim)

# Plot images
dip.figure()
dip.subplot(2,2,1)
dip.imshow(cv2.cvtColor(img_good,cv2.COLOR_BGR2RGB))
dip.title('Original Good Image')
dip.subplot(2,2,2)
dip.imshow(cv2.cvtColor(img_bad,cv2.COLOR_BGR2RGB))
dip.title('Original Bad Image')
dip.subplot(2,2,3)
dip.imshow(cv2.cvtColor(img_recovered,cv2.COLOR_BGR2RGB))
dip.title('UDCP Recovered Image')


""" dip.subplot(2,2,4)
dip.imshow(img_diff,cmap='gray')
dip.title('Difference Image') """

dip.show()