import cv2
import numpy as np
import os
import random
from numpy.lib.stride_tricks import as_strided

def sliding_window(arr, window_size_x,window_size_y):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size_x = int(window_size_x)
    window_size_y = int(window_size_y)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size_x > 0):
        raise ValueError("need a positive x window size")
    if not (window_size_y > 0):
        raise ValueError("need a positive y window size")
    shape = (arr.shape[0] - window_size_x + 1,
             arr.shape[1] - window_size_y + 1,
             window_size_x, window_size_y)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

def cell_neighbors(arr, i, j, d, d2):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d, 2*d2)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d2, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d2 - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d2 - j + jx)

    return w[ix, jx][i0:i1,j0:j1]#.ravel()

#img = cv2.imread('f:\Project\map\mymap_2.jpg')#
img = cv2.imread('f:\Project\map.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,threshed = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)


kernel = np.ones((2,2),np.uint8)

opening = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
sure_bg = cv2.dilate(opening,kernel,iterations=2)           #Расширение и эрозия
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_C ,3)
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
#np.uint8(dist_transform)

#http://www.bim-times.com/opencv/3.3.0/d3/db4/tutorial_py_watershed.html


# sure_fg = np.uint8(sure_fg)

used = {}

threshed_2 = threshed.copy()

i,j = np.unravel_index(dist_transform.argmax(), dist_transform.shape)

k=0
s1=1
s2=1
while True:
    if np.amax(dist_transform)< 0.05:
        break
    if threshed_2.shape[0]<=i+s1 or threshed_2.shape[1]<=j+s2:
        cv2.rectangle(threshed_2,(j,i),(j,i),(0),-1)
        cv2.rectangle(dist_transform,(j,i),(j,i),(0),-1)
        i,j = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
        continue
    
    i0=i-int(s1/2) if i-int(s1/2)>0 else 0
    i1=i+int(s1/2) if threshed_2.shape[1]>i+int(s1/2) else threshed_2.shape[1]
    j0=i-int(s2/2) if i-int(s2/2)>0 else 0
    j1=i+int(s2/2) if threshed_2.shape[0]>i+int(s2/2) else threshed_2.shape[0]

    w_out=threshed_2[i0:i1+1,j0:j1+1]
    # w_out = threshed_2[i:i+s1+1,j:j+s2+1]#cell_neighbors(threshed_2,i,j,s1+1,s2+1)
    
    if 0 in w_out.flatten():
        w_current = threshed_2[i0:i1,j0:j1] #cell_neighbors(threshed_2,i,j,s1,s2)

        if not 0 in w_current.flatten():
            # s1=s
            # s2=s
            cond = True
            cond_done = False
            while True:
                i0=i-int(s1/2) if i-int(s1/2)>0 else 0
                i1=i+int(s1/2) if threshed_2.shape[1]>i+int(s1/2) else threshed_2.shape[1]
                j0=i-int(s2/2) if i-int(s2/2)>0 else 0
                j1=i+int(s2/2) if threshed_2.shape[0]>i+int(s2/2) else threshed_2.shape[0]

                w_current = threshed_2[i0:i1,j0:j1] #cell_neighbors(threshed_2,i,j,s1,s2)
                cond = not 0 in w_current.flatten()
                
                if not cond_done:
                    if cond or threshed_2.shape[1]>i+int(s1/2):
                        s1+=1
                    else:
                        s1-=1
                        cond_done = True
                else:
                    if cond or threshed_2.shape[0]>i+int(s2/2):
                        s2+=1
                    else:
                        s2-=1
                        w_current =threshed_2[i0:i1,j0:j1] # cell_neighbors(threshed_2,i,j,s1,s2)
                        break

            if s1>1 and s2>2:
                used[(i,j,s1,s2)]=True
            
        w_current[:,:]=0
        zzz=dist_transform[i0:i1,j0:j1]#cell_neighbors(dist_transform,i,j,s1,s2)
        zzz[:,:]=0
        
        # xxx=cell_neighbors(threshed,i,j,s)
        # xxx[:,:]=0

        s1=1
        s2=1
        i,j = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
    else:
        s1+=1
        s2+=1
    
    if k>10000:
        break
    k+=1

# for i,j,s in used:
#     cv2.rectangle(threshed,(j-s/2,i-s/2),(j+s/2,i+s/2),(0),1)

cv2.imshow("drawCntsImg.jpg", dist_transform)
cv2.imshow("drawCntsImg1.jpg", threshed_2)
cv2.imshow("drawCntsImg2.jpg", threshed)
cv2.waitKey(0)