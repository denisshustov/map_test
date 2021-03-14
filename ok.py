#!/usr/bin/env python3.7
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import skimage
#from skimage.data import imread
from skimage.morphology import watershed
from skimage.color import rgb2gray
#!!!!!!!!!!!!!!!!!
pct = cv2.imread('/content/drive/My Drive/mymap_2.jpg',-1)#map
img = rgb2gray(pct)
ret1,threshed1 = cv2.threshold(pct, 205, 255,0)

seeds, n_seeds = nd.label( nd.binary_erosion( img > 0.4, iterations=11) )

labels = watershed(-img, seeds)
print(labels.shape)
fig, axes = plt.subplots(ncols=3, figsize=(30, 10))
ax0, ax1, ax2 = axes

ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title('Map')
ax1.imshow(seeds, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title('Init')
ax2.imshow(labels, cmap=plt.cm.jet, interpolation='nearest')
ax2.set_title('Separated objects')

for ax in axes:
    ax.axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)


# plt.figure(figsize = (30,10))
# plt.imshow(labels, interpolation='nearest', aspect='auto')
# plt.show()
# plt.savefig('good.png')

# from skimage.morphology import watershed
# from skimage.feature import peak_local_max
# from skimage import morphology
# from scipy import ndimage

# img = cv2.imread('/content/drive/My Drive/mymap_2.jpg',-1)#mymap_2

# # Prepocess
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# flag, image = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
# #image = cv2.GaussianBlur(image, (3,3), 1)

# distance = ndimage.distance_transform_edt(image)
# local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((30, 30)), labels=image)
# markers = morphology.label(local_maxi)
# labels_ws = watershed(-distance, markers, mask=image)

# print(labels_ws.shape)
# print(labels_ws)
# plt.figure(figsize = (30,10))
# plt.imshow(distance, cmap=plt.cm.nipy_spectral)


# plt.figure(figsize = (30,10))
# plt.imshow(image, cmap=plt.cm.nipy_spectral)


# plt.figure(figsize = (30,10))
# plt.imshow(markers, cmap=plt.cm.nipy_spectral)

# plt.figure(figsize = (30,10))
# plt.imshow(labels_ws, cmap=plt.cm.nipy_spectral)
