# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:12:03 2017

@author: Burning
"""

import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def data2txt(samples):
    fp = open('samples.txt', 'w')
    for sample in samples:
        for num in sample.reshape(784):
            fp.write('%d ' %num)
        fp.write('\n')

#mnist = datasets.fetch_mldata('MNIST Original')
#mnist_data = mnist.data
#mnist_target = mnist.target
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.97)
model = GradientBoostingClassifier(n_estimators=100)
model = LogisticRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predict = model.predict(x_test)
score = np.mean(predict == y_test)
print(score)
samples = [cv2.resize(digit, (28,28)) for digit in digits]
for i, sample in enumerate(samples):
    cv2.imwrite('./tmp_digit/_%d.jpg' %i, sample)
samples = np.array([ sample.reshape(784) for sample in samples])
predict = model.predict(samples)
print(predict)