import cv2
import numpy as np
import os
import random
from numpy.lib.stride_tricks import as_strided
import math
from collections import deque

obsticle = 0

class Node:
    def __init__(self,x,y,left,right,up,down,id):
        self.children = []
        self.parent = None
        self.x = x
        self.y = y
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.id = id
    
    def show_list(self, showChildren=True):
        for n in self.children:
            n.show(True, False)

    def get_coord(self):
        return (self.x-self.left, self.y-self.up, self.x+self.right, self.y+self.down)
        
    def show(self, showChildren=True, ignore_childern=False):
        if len(self.children)>0 or not ignore_childern: # or True:
            x1 = self.x-self.left
            y1 = self.y-self.up
            x2 = self.x+self.right
            y2 = self.y+self.down
            cv2.rectangle(img,(x1,y1),(x2,y2),(31,255,0),1)
            
            c1x = int(x1 + (abs(x1-x2)/2))
            c1y = int(y1 + (abs(y2-y1)/2))
            cv2.rectangle(img,(c1x,c1y),(c1x,c1y),(255,0,0),1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(img, str(self.id), (c1x,c1y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(img, str(self.x) + ':'+str(self.y), (c1x,c1y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if showChildren and len(self.children)>0:
                self.show_list()
    
    def get_central_point(self):
        x = self.x-self.left
        y = self.y-self.up
        x2 = self.x+self.right
        y2 = self.y+self.down

        c_x = int(x + (abs(x-x2)/2))
        c_y = int(y + (abs(y-y2)/2))
        return (c_x,c_y)
    
    @property
    def width(self):
        return self.left+self.right+1
    
    @property
    def height(self):
        return self.up+self.down+1

    def area(self):
        return (self.width)*(self.height)

    def dist(self, other):
        c_point = self.get_central_point()
        c_x = c_point[0]
        c_y = c_point[1]
        
        c_point_2 = other.get_central_point()
        o_c_x = c_point_2[0]
        o_c_y = c_point_2[1]
        
        if abs(c_x - o_c_x) <= (self.width + other.width):
            dx = 0
        else:
            dx = abs(c_x - o_c_x) - (self.width + other.width)
        
        if abs(c_y - o_c_y) <= (self.height + other.height):
            dy = 0
        else:
            dy = abs(c_y - o_c_y) - (self.height + other.height)
        return dx + dy

    def __eq__(self,other): 
        return other!=None and self.x==other.x and self.y==other.y and self.left==other.left and self.right==other.right \
            and self.up==other.up and self.down==other.down 

def getWindow(arr, x, y, border=0, left=0, right=0, up=0, down=0):
    if border!=0:
        left+=border
        right+=border
        up+=border
        down+=border
    i0=y-up if y-up>0 else 0
    i1=y+down+1
    j0=x-left if x-left>0 else 0
    j1=x+right+1
    return arr[i0:i1,j0:j1]

def getNeibors(node, used):
    result = []

    for x1, y1, left1, right1, up1, down1 in used:
        if node.x!=x1 and node.y!=y1:
            z = node.dist(Node(x1,y1,left1,right1,up1,down1,-1))
            if z>=0 and z<=1:
                result.append((x1, y1, left1, right1, up1, down1,z))
    return result

def getChain(used, node, id):
    id +=1
    neibors = getNeibors(node,used)
    if (node.x,node.y,node.left,node.right,node.up,node.down) in used:
        idx = used.index((node.x,node.y,node.left,node.right,node.up,node.down))
        del used[idx]
            
    if len(neibors)>0:
        for x_,y_,left_,right_,up_,down_,_ in neibors:
            child_node = Node(x_,y_,left_,right_,up_,down_,id)
                            
            child_node.parent = node
            node.children.append(child_node)

            id = getChain(used, child_node, id)
            if (x_,y_,left_,right_,up_,down_) in used:
                idx = used.index((x_,y_,left_,right_,up_,down_))
                del used[idx]

    return id

def get_rect(arr,x,y,left,right,up,down,order):
    cond_counter = 0

    while True:
        if cond_counter==order[0]:
            if (x - left-1) > 0:
                left+=1
            else:
                cond_counter+=1

        elif cond_counter==order[1]:
            if (x + right + 1) < arr.shape[1]:
                right+=1
            else:
                cond_counter+=1

        elif cond_counter==order[2]:
            if (y - up-1) > 0:
                up+=1
            else:
                cond_counter+=1

        elif cond_counter==order[3]:
            if (y + down + 1) < arr.shape[0]:
                down+=1
            else:
                cond_counter+=1
            
        w_out = getWindow(arr,x,y,0,left,right,up,down)

        if obsticle in w_out.flatten():
            if cond_counter==order[0]:
                left-=1
            elif cond_counter==order[1]:
                right-=1
            elif cond_counter==order[2]:
                up-=1
            elif cond_counter==order[3]:
                down-=1
            cond_counter+=1

        if cond_counter==4:
            break
    return x,y,left,right,up,down
    
def hz(arr):
    result = deque()
    k=0
    left=0
    right=0
    up=0
    down=0
    more_than = 1
    y,x = np.unravel_index(arr.argmax(), arr.shape)
    
    while True:
        if np.amax(arr)< 0.005:
            break
        
        w_out = getWindow(arr,x,y,1,left,right,up,down)
        
        if obsticle in w_out.flatten():
            w_out = getWindow(arr,x,y,0,left,right,up,down)

            if not obsticle in w_out.flatten():
                x1,y1,left1,right1,up1,down1 = get_rect(arr, x,y,left,right,up,down,[0,1,2,3])
                x2,y2,left2,right2,up2,down2 = get_rect(arr, x,y,left,right,up,down,[3,2,1,0])
                if (left1+right1+up1+down1)>(left2+right2+up2+down2):
                    x,y,left,right,up,down=x1,y1,left1,right1,up1,down1
                else:
                    x,y,left,right,up,down=x2,y2,left2,right2,up2,down2

                w_out = getWindow(arr,x,y,0,left,right,up,down)
                if left>more_than or right>more_than or up>more_than or down>more_than:
                    result.append((x, y, left, right, up, down))
                    w_out[:,:]=obsticle
                else:
                    # used.append((x, y, left, right, up, down))
                    w_out[:,:]=obsticle

            
            y,x = np.unravel_index(arr.argmax(), arr.shape)
            left=0
            right=0
            up=0
            down=0
        else:
            if (x - left-1) > 0:
                left+=1
            if (x + right + 1) < arr.shape[1]:
                right+=1
            if (y - up) > 0:
                up+=1
            if (y + down + 1) < arr.shape[0]:
                down+=1
        
        # if k>181:
        #     break
        k+=1
    return result

img = cv2.imread('f:\Project\map\mymap_22.jpg')#
#img = cv2.imread('f:\Project\map\mymap_222.jpg')#
#img = cv2.imread('f:\Project\map\mymap_223.jpg')#
#img = cv2.imread('f:\Project\map.jpg')
#img = cv2.imread('f:\Project\map_m.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,threshed = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)

kernel = np.ones((2,2),np.uint8)
dist_transform = cv2.distanceTransform(threshed, cv2.DIST_L1 ,3)
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

# threshed_2 = threshed.copy()
dist_transform_2 = dist_transform.copy()

used=hz(dist_transform)

nodes=[]
id = 0
while len(used)>0:
    node=Node(*used.pop(),id)
    getChain(used, node, id)
    if len(node.children)>0:
        nodes.append(node)

# i=0
for n in nodes:    
    n.show(True,True)#
    # if i>=1110:
    #     break
    # i+=1

# cv2.imshow("drawCntsImg.jpg", dist_transform)
cv2.imshow("drawCntsImg552.jpg", dist_transform_2)
cv2.imshow("drawCntsImg4.jpg", img)
# cv2.imshow("drawCntsImg1.jpg", threshed_2)
cv2.imshow("drawCntsImg2.jpg", threshed)
cv2.waitKey(0)