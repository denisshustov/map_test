from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
from skimage.measure import regionprops
import numpy as np
import cv2

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_lables(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  flag, image = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
  #image = cv2.GaussianBlur(image, (3,3), 1)

  #distance = ndimage.distance_transform_edt(image)
  distance = cv2.distanceTransform(image, cv2.DIST_C, 5)
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

  gray = cv2.cvtColor(np.uint8(tmp_image), cv2.COLOR_RGB2GRAY)
  contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
  return contours
  
img = cv2.imread('/home/pi/git/map_test/img/mymap_22.jpg',-1)#mymap_2
labels_ws = get_lables(img)
props = regionprops(labels_ws)
# props[1].filled_image.astype(np.uint8)
# ret, m2 = cv2.threshold(props[10].filled_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
# for c in contours:
#     cv2.drawContours(img, c, -1, (0, 255, 0), 1)

i=0
for p in props:
  counturs = get_counturs_from_label(p.coords,img.shape)

  for c in counturs:
    if cv2.contourArea(c)<cv2.arcLength(c,True):
      qqqz=123
      continue
    canvas = cv2.polylines(img, [c], True, (255, 0, 0) , 1)

    # cv2.putText(img,str(i), (c[0][0][0],c[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
  i+=1
#-----------CONVERT TO CONTURS-----------

cv2.imshow("drawCntsImg23.jpg",img)
cv2.waitKey(0)


# plt.figure(figsize = (30,10))
# plt.imshow(labels_ws, cmap=plt.cm.nipy_spectral)


