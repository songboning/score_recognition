# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:53:31 2017
从试卷图片中切出打分表格
@author: Burning
"""

import cv2
import numpy as np

def cut_column(img):
    while img.size > 10000000:
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = np.mean(img)
    black = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((6,6), np.uint8)
    black = cv2.dilate(black, kernel)
    kernel = np.ones((8,8), np.uint8)
    black = cv2.erode(black, kernel)
    kernel = np.ones((2,2), np.uint8)
    black = cv2.dilate(black, kernel)
    cum_col = [sum(black[:,i]) for i in range(black.shape[1])]
    cum_col = np.array(cum_col) * 100 / max(cum_col)
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
#切出图像中包含打分表的纵列

def cut_row(img):
    return img[int(0.1*img.shape[0]):int(0.6*img.shape[0]),:]

def main_col():
    for i in range(1, 68+1):
        img = cv2.imread('../data/cpp (%d).jpg' %i)
        tc = cut_column(img)
        cv2.imwrite('./tmp/cpp%d.jpg' %i, tc)
        print('cpp %d is ok' %i)
    for i in range(1, 213+1):
        img = cv2.imread('../data/ee (%d).jpg' %i)
        tc = cut_column(img)
        cv2.imwrite('./tmp/ee%d.jpg' %i, tc)
        print('ee %d is ok' %i)

def main():
    for i in range(1, 68+1):
        img = cv2.imread('./tmp/cpp%d.jpg' %i)
        tc = cut_row(img)
        cv2.imwrite('./tmp/_cpp%d.jpg' %i, tc)
        print('cpp %d is ok' %i)
    for i in range(1, 213+1):
        img = cv2.imread('./tmp/ee%d.jpg' %i)
        tc = cut_row(img)
        cv2.imwrite('./tmp/_ee%d.jpg' %i, tc)
        print('ee %d is ok' %i)

def rename(new_size=(500,500)):
    cnt = 0
    for i in range(5, 68+1):
        img = cv2.imread('./tmp/_cpp%d.jpg' %i)
        img = cv2.resize(img, new_size)
        cv2.imwrite('../voc_format/pic/%06d.jpg' %cnt, img)
        print('cpp %d is ok' %i)
        cnt += 1
    for i in range(8, 213+1):
        img = cv2.imread('./tmp/_ee%d.jpg' %i)
        img = cv2.resize(img, new_size)
        cv2.imwrite('../voc_format/pic/%06d.jpg' %cnt, img)
        print('ee %d is ok' %i)
        cnt += 1

rename()
#img = cv2.imread('../data/ee (37).jpg')
#tc = cut_column(img)
#plt.imshow(tc)