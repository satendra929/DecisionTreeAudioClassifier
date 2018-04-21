
from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
import math
from python_speech_features import mfcc
import os
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from graphviz import Source
from sklearn import tree
from sklearn.datasets import load_iris
import collections


def predict(features):
    if (features[19] <= 1080946688.0):
        if (features[7] <= 386723776.0):
            if (features[9] <= 1281349888.0):
                if (features[1] <= 1085200384.0):
                    if (features[17] <= 1214752000.0):
                        if (features[5] <= 191212352.0):
                            return [[0,1]]
                        else:
                            if (features[15] <= 1007127040.0):
                                return [[31, 0]]
                            else:
                                return [[4,2]]
                    else:
                        return [[0,1]]
                else:
                    if (features[6] <= 897229952.0):
                        return [[0,4]]
                    else:
                        if (features[9] <= 896530496.0):
                            return [[3,0]]
                        else:
                            return [[0,1]]
            else:
                return [[0,3]]
        else:
            if (features[11] <= 1336672256.0):
                if (features[2] <= 343647808.0):
                    if (features[19] <= 806112192.0):
                        return [[0,17]]
                    else:
                        if (features[8] <= 894574976.0):
                            if (features[3] <= 771246656.0):
                                return [[1,0]]
                            else:
                                return [[1,8]]
                        else:
                            return [[5,0]]
                else:
                    if (features[12] <= 236159072.0):
                        return [[8,0]]
                    else:
                        if (features[15] <= 293831104.0):
                            if (features[5] <= 1295487104.0):
                                return [[1,13]]
                            else:
                                return [[1,0]]
                        else:
                            if (features[11] <= 918677888.0):
                                return [[73,91]]
                            else:
                                return [[74,47]]
            else:
                if (features[15] <= 462198528.0):
                    return [[0,1]]
                else:
                    return [[12, 0]]
    else:
        if (features[0] <= 475709184.0):
            if (features[2] <= 1093208832.0):
                if (features[9] <= 300194176.0):
                    if (features[10] <= 1111915264.0):
                        return [[2,0]]
                    else:
                        return [[0,1]]
                else:
                    if (features[6] <= 198416896.0):
                        return [[1,0]]
                    else:
                        if (features[15] <= 1279640320.0):
                            return [[0,33]]
                        else:
                            if (features[16] <= 690079104.0):
                                return [[1,0]]
                            else:
                                return [[0,1]]
            else:
                if (features[20] <= 888885504.0):
                    return [[0,2]]
                else:
                    return [[4,0]]
        else:
            if (features[12] <= 591325568.0):
                if (features[16] <= 266438112.0):
                    return [[2,0]]
                else:
                    if (features[22] <= 279484224.0):
                        return [[2,0]]
                    else:
                        return [[0,18]]
            else:
                if (features[15] <= 1116518272.0):
                    if (features[1] <= 1052548736.0):
                        if (features[15] <= 320609600.0):
                            return [[0,2]]
                        else:
                            if (features[10] <= 410241408.0):
                                return [[1,2]]
                            else:
                                return [[20, 1]]
                    else:
                        if (features[7] <= 1003890944.0):
                            return [[0,6]]
                        else:
                            return [[3,0]]
                else:
                    return [[0,5]]

X = []
Y = []
# Getting Truck MAG values from text file
filepath = ['TRUCK_MAG_PHOTON_TEST.txt']

for f in filepath:
    with open(f) as fp:
        line = fp.readline()
        while line:
            mag_values = line.strip().split(",")
            if len(mag_values) != 1:
                X.append(list(map(int, mag_values[:len(mag_values)-1][:23])))
                Y.append(True)
            line = fp.readline()


# Getting Non-Truck MAG values from text file
filepath = ['NON_TRUCK_MAG_PHOTON_TEST.txt']

for f in filepath:
    with open(f) as fp:
        line = fp.readline()
        while line:
            mag_values = line.strip().split(",")
            if len(mag_values) != 1:
                X.append(list(map(int, mag_values[:len(mag_values)-1][:23])))
                Y.append(False)
            line = fp.readline()


data_feature_names = [(str)(x) for x in range(23)]
print(data_feature_names)

for index, value in enumerate(X):
    if Y[index] == True:
        actual = "This is a Truck Sound"
    else:
        actual = "This is not a Truck Sound"
    pred = predict(value)
    prediction = False
    if (pred[0][0] >= pred[0][1] ) :
        prediction = False
    else :
        prediction = True 
    print("Prediction :" + (str)(prediction) + " Actual :" + actual)
