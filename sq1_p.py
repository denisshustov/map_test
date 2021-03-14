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

def cell_neighbors(arr, i, j, d,d2):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1, 2*d2+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1]#.ravel()

def get_nearest_free_area(current_window, a, default_sq, free_cell, obsticle_cell, markered_cell):
    (s,x,y) = current_window
    s += 2
    current_sq = cell_neighbors(a,x,y,s,s)
    max = current_sq.shape[0]-1
    # is_free = lambda area: free_cell in area and markered_cell in area

    if not free_cell in current_sq: #np.any(current_sq == free_cell):
        return []

    xy_combination = np.linspace((1,max),(1),max)
    xy_combination = np.concatenate([xy_combination, np.linspace((1),(max,1),max) ])
    xy_combination = np.concatenate([xy_combination, np.linspace((max),(max,1),max) ])
    xy_combination = np.concatenate([xy_combination, np.linspace((1,max),(max,max),max) ])

    positions = []
    for comb in xy_combination.astype(int):
        tmp_sq = cell_neighbors(current_sq, comb[0], comb[1],default_sq,default_sq)
        if not obsticle_cell in tmp_sq and not markered_cell in tmp_sq:
            positions.append([comb[0], comb[1]])
    return positions

img = cv2.imread('f:\Project\map.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,threshed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#blurred = cv2.GaussianBlur(gray, (5,5), 51)
edges = cv2.Canny(gray,120, 255, 1)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

canvas = img.copy()

cv2.drawContours(canvas, contours, -1, (128,255,0), 3, cv2.LINE_AA, hierarchy, 1 )

free_cell = 255
obsticle_cell = 0
markered_cell = 55

default_sq = 2
x=1
y=1
s=default_sq
while True:
    z = cell_neighbors(threshed,x,y,s,s)
    if obsticle_cell in z: #np.any(z == obsticle_cell):
        #found obsticle -> find neibor free x/y
        #x=s+x
        #y=s+y
        # s=3
        positions = get_nearest_free_area((s,x,y),threshed,default_sq,free_cell,obsticle_cell,markered_cell)
        # while True:
        #     next_pos = cell_neighbors(threshed,x,y,s,s)
        #     if np.any(next_pos!=0) and np.any(next_pos!=55):
        #         pass
        #     else:
                
        break
    else:
        s+=2
        cv2.rectangle(z,(x,y),(x+s,y+s),(markered_cell),10)        
cv2.imshow("drawCntsImg.jpg", threshed)
cv2.imshow("drawCntsImg1.jpg", z)
cv2.waitKey(0)