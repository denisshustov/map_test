from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops
import cv2

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_lables(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  flag, image = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
  #image = cv2.GaussianBlur(image, (3,3), 1)

  #distance = ndimage.distance_transform_edt(image)
  distance = cv2.distanceTransform(image, cv2.DIST_C ,5)
  local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((60, 60)), labels=image)
  markers = morphology.label(local_maxi)
  labels_ws = watershed(-distance, markers, mask=image)

  return cv2.normalize(labels_ws, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


#-----------CONVERT TO CONTURS-----------

def get_counturs_from_label(label_prop_coords, scr_img_shape):
  c = np.flip(label_prop_coords, axis=None)
  tmp_image = np.zeros(shape=(scr_img_shape))
  tmp_image[:,:]=255
  for z in c:
    tmp_image[z[1],z[0]]=0

  edges = cv2.Canny( np.uint8(tmp_image),0, 255, 1)
  contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
  return contours
  
img = cv2.imread('/content/drive/My Drive/mymap_2.jpg',-1)#mymap_2
labels_ws = get_lables(img)
props = regionprops(labels_ws)
print(props[1].coords)
z = get_counturs_from_label(props[1].coords,image.shape)

for c in z:
  canvas = cv2.polylines(image, [c], True, (255, 0, 0) , 1)
#-----------CONVERT TO CONTURS-----------

# print(labels_ws)
plt.figure(figsize = (30,10))
plt.imshow(image) #, cmap=plt.cm.nipy_spectral)

# plt.figure(figsize = (30,10))
# plt.imshow(image, cmap=plt.cm.nipy_spectral)


# plt.figure(figsize = (30,10))
# plt.imshow(markers, cmap=plt.cm.nipy_spectral)

# plt.figure(figsize = (30,10))
# plt.imshow(labels_ws, cmap=plt.cm.nipy_spectral)


