from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
import cv2
import numpy as np
from skimage.measure import regionprops

img = cv2.imread('/home/pi/git/map_test/img/mymap_22.jpg',-1)#mymap_22

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
flag, image = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)

#distance = ndimage.distance_transform_edt(image)
distance = cv2.distanceTransform(image, cv2.cv.CV_DIST_C ,3)
# distance = cv2.normalize(distance, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((180, 180)), labels=image)
markers = morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=image)
labels_ws = cv2.normalize(labels_ws, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

props = regionprops(labels_ws)

c = np.flip(props[0].coords, axis=None)
c1 = np.flip(props[1].coords, axis=None)

cv2.polylines(img,[c],True,(255,255,0),1)
cv2.polylines(img,[c1],True,(0,255,0),1)

cv2.imshow("drawCntsImg2.jpg",img)
cv2.imshow("drawCntsImg23.jpg",labels_ws)
cv2.waitKey(0)


