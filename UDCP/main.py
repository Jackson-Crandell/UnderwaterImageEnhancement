import os
import numpy as np
import cv2
import natsort

from RefinedTramsmission import Refinedtransmission
from getAtomsphericLight import getAtomsphericLight
from getGbDarkChannel import getDarkChannel
from getTM import getTransmission
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/UDCP"
#folder = "C:/Users/Administrator/Desktop/Databases/Dataset"
""" path = folder + "/InputImages"
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        img = cv2.imread(folder +'/InputImages/' + file) """

img = cv2.imread('/home/jcrandell/Desktop/ECE4803/Project/image_dehaze/image/fish_bad.jpg')

#print('img',img)

blockSize = 9
GB_Darkchannel = getDarkChannel(img, blockSize)
AtomsphericLight = getAtomsphericLight(GB_Darkchannel, img)

print('AtomsphericLight', AtomsphericLight)
# print('img/AtomsphericLight', img/AtomsphericLight)

# AtomsphericLight = [231, 171, 60]

transmission = getTransmission(img, AtomsphericLight, blockSize)

# cv2.imwrite('OutputImages/' + prefix + '_UDCP_Map.jpg', np.uint8(transmission))

transmission = Refinedtransmission(transmission, img)
sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

Ir = img[:,:,2] # read red channel
Ig = img[:,:,1] # read green channel 
Ib = img[:,:,0] # read blue channel

R_total= 0
G_total= 0
B_total= 0

R_total = np.sum(Ir)
G_total = np.sum(Ig)
B_total = np.sum(Ib)
print(R_total)
print(G_total)
print(B_total)
#Calculate the image total intensity
I_total = R_total + G_total + B_total

intensity_average = I_total / 3
print(intensity_average)
# print('AtomsphericLight',AtomsphericLight)


cv2.imwrite("/home/jcrandell/Desktop/ECE4803/Project/image_dehaze/image/fish_transmission.png", np.uint8(transmission * 255))
cv2.imwrite("/home/jcrandell/Desktop/ECE4803/Project/image_dehaze/image/fish_sceneRadiance.png", sceneRadiance)


