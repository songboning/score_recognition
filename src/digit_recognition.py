# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:16:32 2017

@author: Burning
"""

import cv2
import numpy as np

def cut_red(img, delta=13, maxval=255):
    channel = cv2.split(img.astype(int))
    red = channel[2]
    red = np.where(red>channel[0]+delta, red, 0)
    red = np.where(red>channel[1]+delta, maxval, 0)
    red = red.astype(np.uint8)
    #找出红色通道值比较大的像素
    kernel = np.ones((2,2), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    return red
#按照红色通道的突出成都取出红色区域

def bound(red):
    red, contours, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #只取最外层轮廓RETR_EXTERNAL,简单结构CHAIN_APPROX_SIMPLE,得到原图red,轮廓列表contours,包含关系信息hierarchy
    boundrects = np.mat([cv2.boundingRect(contour) for contour in contours])
    boundrects[:,2] += boundrects[:,0]
    boundrects[:,3] += boundrects[:,1]
    return boundrects
#得到红色数字的外接矩形

for i in range(5, 68+1):
    img = cv2.imread('./tmp/_cpp%d.jpg' %i)
    img = cut_red(img)
    cv2.imwrite('./tmp/__red%d.jpg' %i, img)
    print(i, 'ok')