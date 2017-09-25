# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 23:18:50 2017
对数据做随机分配
@author: Burning
"""

import random
import math

def allot(total, cut_num):
    num = [int(total * rate) for rate in cut_num]
    if sum(num) > total:
        print ('划分比例不合理')
        num = [math.floor(total/cut_num)] * len(cut_num)
    num = [sum(num[:i]) for i in range(len(num) + 1)]
    num.append(total)
    seq = list(range(total))
    random.shuffle(seq)
    allot = [seq[num[i-1]:num[i]] for i in range(1, len(num))]
    return allot
#通过传入前N-1份比例组成的cut_num列表，得到0~total-1乱序的序列组

table = allot(251, [0.5])
file = {}
file[0] = open('../voc_format/ImageSets/Main/trainval.txt', 'w')
file[1] = open('../voc_format/ImageSets/Main/test.txt', 'w')
for i in range(len(file)):
    for n in sorted(table[i]):
        file[i].writelines('%06d\n' %n)
    file[i].close()