# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:15:50 2017
从试卷截图中截取打分表格
@author: Burning
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def see_gray(img, gap=15, wait=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0, 255, gap):
        show = cv2.inRange(gray, i, min(i+gap, 255))
        cv2.imshow("show", show)
        print('gray:%d-%d' %(i,min(i+gap,255)))
        cv2.waitKey(wait)
    cv2.destroyAllWindows()
#查看一张图片的灰度分布,255以gap为区间,巨慢无比没法用

def sum_column(img, threshold=100, greater=False):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = img
    black = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)[1]
    sc = [sum(black[:,i]) for i in range(black.shape[1])]
    return sc
#统计图片每列累计特征

def cut_table(img, threshold=150):
    while img.size > 10000000:
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    black = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((5,5), np.uint8)
    black = cv2.dilate(black, kernel)
    kernel = np.ones((7,7), np.uint8)
    black = cv2.erode(black, kernel)
    kernel = np.ones((2,2), np.uint8)
    black = cv2.dilate(black, kernel)
    cum_col = [sum(black[:,i]) for i in range(black.shape[1])]
    cum_col = np.array(cum_col) * 100 / max(cum_col)
    pos_left = np.argmax(cum_col[:int(cum_col.size/4)])
    pos_left += np.argmin(cum_col[pos_left:pos_left+int(cum_col.size/10)])
    pos_right = int(cum_col.size / 2)
    step = int(cum_col.size / 20)
    pos = int(cum_col.size * 0.3)
    bestval = 1000
    a, b = 0.7, 0.3
    while(pos+step<cum_col.size*0.82):
        sub = cum_col[pos-step:pos+step]
        mean = np.mean(sub)
        var = np.sum(sub * sub) / sub.size - mean * mean
        if a * mean + b * var < bestval:
            bestval = a * mean + b * var
            pos_right = pos
        pos += step
    table = img[:, pos_left:pos_right]
    plt.plot(cum_col)
    return table

def _main(i=10):
    img = cv2.imread("../data/cpp (%d).jpg" %i)
    sc = sum_column(img)
    plt.subplot('211')
    plt.plot(sc)
    img = cv2.imread("../data/ee (%d).jpg" %i)
    sc = sum_column(img)
    plt.subplot('212')
    plt.plot(sc)

def extract():
    kernel = np.ones((5,5), np.uint8)
    tmp = cv2.dilate(black, kernel)
    kernel = np.ones((7,7), np.uint8)
    tmp = cv2.erode(tmp, kernel)
    kernel = np.ones((2,2), np.uint8)
    tmp = cv2.dilate(tmp, kernel)
    plt.imshow(tmp)
    cv2.imwrite('tmp.jpg', tmp)
    return tmp

#img = cv2.imread("../data/ee (37).jpg")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#black = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
#kernel = np.ones((5,5), np.uint8)
#tmp = cut_table(img)
#plt.imsave('tmp.jpg', tmp)
#plt.imshow(tmp)

def predeal_(gray):
    threshold = 150
    black = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((7,7), np.uint8)
    black = cv2.dilate(black, kernel)
    kernel = np.ones((7,7), np.uint8)
    black = cv2.erode(black, kernel)
    #kernel = np.ones((2,2), np.uint8)
    #black = cv2.dilate(black, kernel)
    return black

def predeal(gray):
    black = cv2.medianBlur(gray, 3)
    black = cv2.threshold(gray, 150, 1, cv2.THRESH_BINARY_INV)[1]
    return black

def cut_row(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 150
    #hreshold = np.mean(img)
    black = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((16,16), np.uint8)
    black = cv2.dilate(black, kernel)
    kernel = np.ones((19,19), np.uint8)
    black = cv2.erode(black, kernel)
    kernel = np.ones((3,3), np.uint8)
    black = cv2.dilate(black, kernel)
    cum_col = [sum(black[i,:]) for i in range(black.shape[0])]
    #cum_col = [sum(black[:,i]) for i in range(black.shape[1])]
    cum_col = np.array(cum_col) * 100 / max(cum_col)
    return cum_col
    pos_left = np.argmax(cum_col[:int(cum_col.size/6)] + np.linspace(4, 0, int(cum_col.size/6)))
    pos_left += np.argmin(cum_col[pos_left:pos_left+int(cum_col.size/20)])# + np.linspace(0, 2, int(cum_col.size/20)))
    pos_right = int(cum_col.size / 2)
    step = int(cum_col.size / 30)
    pos = int(cum_col.size * 0.3)
    bestval = 1000
    a, b = 0.7, 0.3
    #cum_col += np.linspace(0, 0, cum_col.size)
    while(pos+step<cum_col.size*0.8):
        sub = cum_col[pos-step:pos+step]
        mean = np.mean(sub)
        var = np.sum(sub * sub) / sub.size - mean * mean
        if a * mean + b * var < bestval:
            bestval = a * mean + b * var
            pos_right = pos
        pos += step
    table_col = img[:, pos_left:pos_right]
    return table_col

def main(i):
    img = cv2.imread('./tmp/cpp%d.jpg' %i)
    tc = cut_row(img)
    plt.plot(tc)

def get_line(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray,50,150,apertureSize = 3)
    edges = predeal(gray)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=int(img.shape[1]/4),maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    return img

def invmothed(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    cum_col = [sum(gray[i,:]) for i in range(gray.shape[0])]
    plt.plot(cum_col)

#img = cv2.imread('./tmp/ee55.jpg')
#tmp = get_line(img)
#plt.imshow(tmp)
#plt.imsave('tmp.jpg', tmp)

def cut_red_bgr(img, delta=12, maxval=255):
    channel = cv2.split(img.astype(int))
    red = channel[2]
    red = np.where(red>channel[0]+delta, red, 0)
    red = np.where(red>channel[1]+delta, red, 0)
    red = red.astype(np.uint8)
    kernel = np.ones((2,2), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    return red
#按照红色通道的突出成都取出红色区域

def cut_red_hsv(img, maxval=255):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = np.where((30<h) & (h<126), 0, maxval)
    red = np.where(s>43, red, 0)
    red = np.where(46<v, red, 0)
    return red
#基于HSV空间筛选红色,位运算要注意优先级

def bound(red, min_len=5):
    red, contours, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #只取最外层轮廓RETR_EXTERNAL,简单结构CHAIN_APPROX_SIMPLE,得到原图red,轮廓列表contours,包含关系信息hierarchy
    rects = [cv2.boundingRect(contour) for contour in contours]
    #rects = [rect for rect in rects if rect[2]>min_len or rect[3]>min_len]
    rects = [rect for rect in rects if rect[2] * rect[3] > 30]
    rects = np.mat(rects)
    rects[:,2] += rects[:,0]
    rects[:,3] += rects[:,1]
    return rects
#得到红色数字的外接矩形

#red, contours, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#boundrects = np.mat([cv2.boundingRect(contour) for contour in contours])
#boundrects[:,2] += boundrects[:,0]
#boundrects[:,3] += boundrects[:,1]
#for rect in boundrects:
#    rect[2] += rect[0]
#    rect[3] += rect[1]

def cut_digit(red, rects):
    digits = [red[rect[0]:rect[2]+1, rect[1]:rect[3]+1] for rect in rects]
    return digits
#从图像中切除数字部分

i = random.randint(10, 60)
img = cv2.imread('./tmp/_cpp%d.jpg' %i)
red = cut_red_bgr(img)
boundrects = bound(red).tolist()
baseline = np.zeros(img.shape[0])
for rect in boundrects:
    baseline[rect[1]:rect[3]] += 1
pos = np.argmax(baseline)
boundrects = [rect for rect in boundrects if rect[1]-8<=pos<=rect[3]+8]
#tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#tmp = img.copy()
tmp = red.copy()
for rect in boundrects:
        cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (0,255,0),2)
#cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
plt.imshow(img)
delta = 5
digits = [tmp[rect[1]-delta:rect[3]+delta, rect[0]-delta:rect[2]+delta] for rect in boundrects]
for i, digit in enumerate(digits):
    cv2.imwrite('./tmp_digit/%d.jpg' %i, digit)
