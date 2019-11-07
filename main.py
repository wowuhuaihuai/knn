# -*- coding: utf-8 -*-
"""
@Project = KNN
@File = main.py
@Author = FengQi
@Mail = wowuhuaihuai@163.com
@Time = 2019/11/7 10:21 上午
@Software: PyCharm
"""
from kNN import kNNClassifier
import numpy as np

raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343853454, 3.368312451],
              [3.582294121, 4.679917921],
              [2.280362211, 2.866990212],
              [7.423436752, 4.685324231],
              [5.745231231, 3.532131321],
              [9.172112222, 2.511113104],
              [7.927841231, 3.421455345],
              [7.939831414, 0.791631213]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]# 设置训练组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

x = np.array([[8.90933607318, 3.365731514]])


knn_clf = kNNClassifier(k=6)
knn_clf.fit(X_train, y_train)
X_predict = x.reshape(1,-1)
y_predict = knn_clf.predict(X_predict)
print(y_predict)
