from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage

img = cv2.imread('/content/drive/My Drive/mymap_2.jpg',-1)#mymap_2

# Prepocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
flag, image = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
#image = cv2.GaussianBlur(image, (3,3), 1)

distance = ndimage.distance_transform_edt(image)
# distance = cv2.distanceTransform(image, cv2.DIST_L1 ,5)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((60, 60)), labels=image)
markers = morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=image)

print(labels_ws.shape)
print(labels_ws)
plt.figure(figsize = (30,10))
plt.imshow(distance, cmap=plt.cm.nipy_spectral)


plt.figure(figsize = (30,10))
plt.imshow(image, cmap=plt.cm.nipy_spectral)


plt.figure(figsize = (30,10))
plt.imshow(markers, cmap=plt.cm.nipy_spectral)

plt.figure(figsize = (30,10))
plt.imshow(labels_ws, cmap=plt.cm.nipy_spectral)
