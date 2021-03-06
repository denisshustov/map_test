import cv2
import numpy as np
import os
import random
import sys
import math

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
from skimage.measure import regionprops

class PathFinder:

    def __init__(self, contours, src_image, GRID_SIZE = 10, border_in = 5):
        if src_image.shape[0] == 0 or src_image.shape[1] == 0:
            raise Exception('src_image has 0 shape')
        if len(contours)==0:
            raise Exception('contours len is 0')
        if GRID_SIZE==0:
            raise Exception('GRID_SIZE is 0!')
        
        self.GRID_SIZE = GRID_SIZE
        self.border_in = border_in
        self.src_image = src_image
        self.contours = contours
        self.array_of_contours = np.empty(shape=[0, 2],dtype=int)
        for c in contours:
            self.array_of_contours = np.concatenate((self.array_of_contours,np.array(c).reshape(-1,2)))
    
    def line_intersection(self, line1, line2):
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

    def dist(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        
    def getNeibors(self, p, arr, trd, take_first=False):
        result = []

        for x, y in arr:
            if p[0]!=x or p[1]!=y:
                z = self.dist(p, (x, y))
                if z>=0 and z<=trd and not (x, y, z) in result:
                    result.append((x, y, z))
        result = sorted(result, key=lambda s: s[2])
        return result

    def get_relevant_points(self, neibors,un_x, un_x_ptr,ok_arr):
        result = []
        for n in neibors:
            if n[0] in un_x[:un_x_ptr] and not (n[0],n[1]) in ok_arr:
                result.append(n)
        return result
    
    def get_route(self, visualize=False,visualize_grid=False,):
        int_points = []
        y_min = min(self.array_of_contours[:,1:])[0]
        y_max = max(self.array_of_contours[:,1:])[0]+self.GRID_SIZE

        x_min = min(self.array_of_contours[:,:1])[0]
        x_max = max(self.array_of_contours[:,:1])[0]+self.GRID_SIZE
        prev = (None,None)

        height = self.src_image.shape[0]
        width  = self.src_image.shape[1]
        for x in range(x_min, x_max, self.GRID_SIZE):
            if visualize_grid:
                cv2.line(self.src_image, (x, 0), (x, height), (255, 0, 0), 1, 1)
            for y in range(y_min, y_max, self.GRID_SIZE):
                if visualize_grid:
                    cv2.line(self.src_image, (0, y), (width, y), (255, 0, 0), 1, 1)
                vvv = self.line_intersection(((x, 0), (x, height)),((0, y), (width, y)))
                if vvv:
                    for c in self.contours:
                        is_in = cv2.pointPolygonTest(c, (vvv[0], vvv[1]), True)
                        if is_in >= self.border_in:
                            int_points.append((vvv[0], vvv[1]))
                            if visualize:
                                cv2.circle(self.src_image, (vvv[0], vvv[1]),1, (0,0,255), -1)

        if len(int_points)==0:
            return []
            #raise Exception('int_points is empty, something go wrong')
        #OK!!!!!!!!!!!!!!!!!!
        un_x = np.unique(np.array(int_points)[:,:1], axis=0)
        un_x_ptr = 0
        neibor_distance = 8
        k=0
        ok_arr = []
        # for xy in int_points:
        xy = int_points[0]
        while True:
            x=xy[0]
            y=xy[1]

            cv2.circle(self.src_image, (x,y),1, (255,255,255), -1)

            neibors = self.getNeibors(xy,int_points,neibor_distance)
            relevant_points = self.get_relevant_points(neibors,un_x, un_x_ptr,ok_arr)


            if len(relevant_points)>0:
                relevant_points_fst = sorted(relevant_points, key=lambda s: s[2])[0]
                if not relevant_points_fst in ok_arr:
                    ok_arr.append((relevant_points_fst[0],relevant_points_fst[1]))
                    cv2.line(self.src_image, (x,y), (relevant_points_fst[0],relevant_points_fst[1]), (0, 0, 0), 1, 1)
                    xy = relevant_points_fst
                else:
                    un_x_ptr+=1
                    if len(un_x)<=un_x_ptr:
                        break
            else:
                all_neibors = [n for n in self.getNeibors(xy,int_points,x_max+y_max,True) if not (n[0],n[1]) in ok_arr]
                if len(all_neibors)>0 and all_neibors[0][2] > neibor_distance:
                    #just to first
                    ok_arr.append((all_neibors[0][0],all_neibors[0][1]))
                    cv2.line(self.src_image, (x,y), (all_neibors[0][0],all_neibors[0][1]), (0, 0, 0), 1, 1)
                    xy = all_neibors[0]
                else:
                    un_x_ptr+=1
                    if len(un_x)<=un_x_ptr:
                        break
            # for n in neibors:
            #     cv2.line(self.src_image, (x,y), (x,y), (0, 0, 0), 1, 1)
            if k>1117111:
                break
            k+=1
        return ok_arr

def get_conturs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flag, image = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)

    distance = cv2.distanceTransform(image, cv2.DIST_C, 5)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((60, 60)), labels=image)
    markers = morphology.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=image)
    result=[]

    for label in np.unique(labels_ws):
        if label == 0:
            continue

        # draw label on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels_ws == label] = 255
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result.append(contours)

    return result  


# img = cv2.imread('f:\Project\map.jpg',-1)
# img = cv2.imread('f:\Project\map\mymap_2.jpg',-1)
img = cv2.imread('/home/pi/git/map_test/img/mymap_22.jpg')#
cnts = get_conturs(img)
i=0

for cnt in cnts:
    pth = PathFinder(cnt,img,10,5)
    qqq = pth.get_route(i==13)
    for c in cnt:
        if cv2.contourArea(c) > 100:
            cv2.drawContours(img, [c], -1, (0, 0, 255), 1, 1)
    i+=1

cv2.imshow("drawCntsImg.jpg", img)
cv2.imshow("drawCntsImg2.jpg", pth.src_image)
cv2.waitKey(0)

#https://stackoverflow.com/questions/60479830/extracting-objects-after-watershed-segmentation