import cv2
import numpy as np
import os
import random
import sys
import math
from collections import deque

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0 :#or div >= max_w or div >= max_h:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def dist(p1, p2):
   return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
def getNeibors(p, arr, trd, take_first=False):
    result = []

    for x, y in arr:
        if p[0]!=x or p[1]!=y:
            z = dist(p, (x, y))
            if z>=0 and z<=trd and not (x, y, z) in result:
                result.append((x, y, z))
    result = sorted(result, key=lambda s: s[2])
    return result

def get_relevant_points(neibors, un_x_ptr,ok_arr):
    result = []
    for n in neibors:
        if n[0] in un_x[:un_x_ptr] and not (n[0],n[1]) in ok_arr:
            result.append(n)
    return result

# img = cv2.imread('f:\Project\map.jpg',-1)
# img = cv2.imread('f:\Project\map\mymap_2.jpg',-1)
img = cv2.imread('f:\Project\map\\r1.jpg')#

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,threshed = cv2.threshold(gray, 188, 255, cv2.THRESH_BINARY)

def correct_conturs(src_img, offset=5):
    tmp_img = np.zeros(shape=(src_img.shape[0]+10,src_img.shape[1]+10),dtype=int)
    tmp_img[:,:]=255
    x_offset=y_offset=offset

    tmp_img[y_offset:y_offset+src_img.shape[0], x_offset:x_offset+src_img.shape[1]] = src_img

    edges = cv2.Canny( np.uint8(tmp_img),0, 255, 1)

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )
    for idx, val in enumerate(contours):
        for idx1, val1 in enumerate(val):
            for idx2, val2 in enumerate(val1):
                contours[idx][idx1][idx2][0]=contours[idx][idx1][idx2][0]-x_offset
                if contours[idx][idx1][idx2][0]<0:
                    contours[idx][idx1][idx2][0] = 0
                contours[idx][idx1][idx2][1]=contours[idx][idx1][idx2][1]-y_offset
                if contours[idx][idx1][idx2][1]<0:
                    contours[idx][idx1][idx2][1] = 0
    return contours
contours = correct_conturs(threshed, 5)
# np.append(contours[0],contours[0][len(contours)-1])

# contours2=np.empty(shape=[0, 2],dtype=int)
# for c in contours:
#     a=np.empty(shape=[0, 2],dtype=int)
#     for c1 in c:
#         b=np.empty(shape=[0, 2],dtype=int)
#         for c2 in c1:
#             b=np.append(b, (c2[0]-5,c2[1]-5))
#         a=np.append(a,b)
#     contours2=np.append(contours2,a)
canvas = img.copy()
# arr=np.empty(shape=[0, 2],dtype=int)

sq_border = 5
sq_size = (sq_border*2)+1

for c in contours:
    # arclen = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, arclen * 0.0001, True)
    # cv2.drawContours(canvas, [approx], -1, (0,0,215), 1, cv2.LINE_8)
    canvas = cv2.polylines(canvas, [c], True, (255, 0, 0) , 1)

    # for a in approx:
    #     cv2.circle(canvas, (a[0][0],a[0][1]), 2, [0,0,0], thickness=1, lineType=8, shift=0)
    
    
# for c in contours:
#     arr = np.concatenate((arr,np.array(c).reshape(-1,2)))

# GRID_SIZE = 10
# border_in = 0
# int_points = []
# y_min = min(arr[:,1:])[0]
# y_max = max(arr[:,1:])[0]+GRID_SIZE

# x_min = min(arr[:,:1])[0]
# x_max = max(arr[:,:1])[0]+GRID_SIZE
# prev = (None,None)

# height, width, channels = canvas.shape
# for x in range(x_min, x_max, GRID_SIZE):
#     # cv2.line(canvas, (x, 0), (x, height), (255, 0, 0), 1, 1)
#     for y in range(y_min, y_max, GRID_SIZE):
#         # cv2.line(canvas, (0, y), (width, y), (255, 0, 0), 1, 1)
#         vvv = line_intersection(((x, 0), (x, height)),((0, y), (width, y)))
#         if vvv:
#             for c in contours:
#                 is_in = cv2.pointPolygonTest(c, (vvv[0], vvv[1]), True)
#                 if is_in >= border_in:
#                     int_points.append((vvv[0], vvv[1]))
#                     cv2.circle(canvas, (vvv[0], vvv[1]),1, (0,0,255), -1)

# if len(int_points)==0:
#     raise Exception('int_points is empty, something go wrong')
# #OK!!!!!!!!!!!!!!!!!!
# un_x = np.unique(np.array(int_points)[:,:1], axis=0)
# un_x_ptr = 0
# neibor_distance = 8
# k=0
# ok_arr = []
# # for xy in int_points:
# xy = int_points[0]
# while True:
#     x=xy[0]
#     y=xy[1]

#     cv2.circle(canvas, (x,y),1, (255,255,255), -1)

#     neibors = getNeibors(xy,int_points,neibor_distance)
#     relevant_points = get_relevant_points(neibors,un_x_ptr,ok_arr)


#     if len(relevant_points)>0:
#         relevant_points_fst = sorted(relevant_points, key=lambda s: s[2])[0]
#         if not relevant_points_fst in ok_arr:
#             ok_arr.append((relevant_points_fst[0],relevant_points_fst[1]))
#             cv2.line(canvas, (x,y), (relevant_points_fst[0],relevant_points_fst[1]), (0, 0, 0), 1, 1)
#             xy = relevant_points_fst
#         else:
#             un_x_ptr+=1
#             if len(un_x)<=un_x_ptr:
#                 break
#     else:
#         all_neibors = [n for n in getNeibors(xy,int_points,x_max+y_max,True) if not (n[0],n[1]) in ok_arr]
#         if len(all_neibors)>0 and all_neibors[0][2] > neibor_distance:
#             #just to first
#             ok_arr.append((all_neibors[0][0],all_neibors[0][1]))
#             cv2.line(canvas, (x,y), (all_neibors[0][0],all_neibors[0][1]), (0, 0, 0), 1, 1)
#             xy = all_neibors[0]
#         else:
#             un_x_ptr+=1
#             if len(un_x)<=un_x_ptr:
#                 break
#     # for n in neibors:
#     #     cv2.line(canvas, (x,y), (x,y), (0, 0, 0), 1, 1)
#     if k>1117111:
#         break
#     k+=1

cv2.imshow("drawCntsImg.jpg", canvas)
cv2.waitKey(0)