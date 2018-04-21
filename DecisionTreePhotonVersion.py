from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
import math
from python_speech_features import mfcc
import os
#import pandas as pd   
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

X = []
Y = []
#Getting Truck MAG values from text file
filepath = ['TRUCK_MAG_PHOTON.txt','TRUCK_MAG_PHOTON_2.txt','TRUCK_MAG_PHOTON_3.txt']

for f in filepath :
    with open(f) as fp:  
       line = fp.readline()
       while line:
           mag_values = line.strip().split(",")
           if len(mag_values) != 1: 
               X.append(list(map(int, mag_values[:len(mag_values)-1][:23])))
               Y.append(True)
           line = fp.readline()

print(X[0])
#Getting Non-Truck MAG values from text file
filepath = ['NON_TRUCK_MAG_PHOTON.txt','NON_TRUCK_MAG_PHOTON_2.txt','NON_TRUCK_MAG_PHOTON_3.txt']

for f in filepath :
    with open(f) as fp:  
       line = fp.readline()
       while line:
           mag_values = line.strip().split(",")
           if len(mag_values) != 1: 
               X.append(list(map(int, mag_values[:len(mag_values)-1][:23])))
               Y.append(False)
           line = fp.readline()

print(len(X),len(Y))
data_feature_names = [(str)(x) for x in range (23)]
print (data_feature_names)
'''
#getting  the clip names
clip_names = []
directory = "."
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        clip_names.append(filename)

clip_number = 0
for clip in clip_names :
    #count in frequency range
    freq_count = [0 for x in range (24)]
    #freq_cnt_bool = [True for x in range (6)]
    #check = [0 for c in range(605)]
    fs_rate, signal = wavfile.read(clip)
    signal = signal[:1024]
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    secs = N / float(fs_rate)
    Ts = 1.0/fs_rate # sampling interval in time
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = scipy.fft(signal)
    
    abs_FFT = abs(FFT)[:24]
    for index,value in enumerate(abs_FFT) :
        freq_count[index] = value
        
    #print (clip+" "+(str)(freq_count))
    X.append(freq_count)
    if "non" in clip :
        Y.append(False)
    else :
        Y.append(True)
    #print (X_train[-1], Y_train)
    clip_number+=1
'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)  

classifier = DecisionTreeClassifier(max_depth=7)  
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test) 
print(confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))




#function to get if and else from tree
def recurse(left, right, threshold, features, node, value):
    if (threshold[node] != -2):
        print ("if ( \"" + features[node] +"\""+ " <= " + str(threshold[node]) + " ) :")
        if left[node] != -1:
            recurse (left, right, threshold, features,left[node], value)
        print (" else :")
        if right[node] != -1:
            recurse (left, right, threshold, features,right[node], value)
        #print ("}")
    else:
        print ("return " + str(value[node]))
def get_code(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    print (value)
    recurse(left, right, threshold, features, 0, value)


get_code(classifier, data_feature_names)

with open("dtree.txt", "w") as f:
    f = tree.export_graphviz(classifier, out_file=f)


#Not Useful

'''
    if value >= 0.01*max(abs_FFT[100:]) and index>=100 and index<=700 :
        if index>=100 and index <=200 :
            freq_count[0] += 1
        elif index>=201 and index <=300 :
            freq_count[1] += 1
        elif index>=301 and index <=400 :
            freq_count[2] += 1
        elif index>=401 and index <=500 :
            freq_count[3] += 1
        elif index>=501 and index <=600 :
            freq_count[4] += 1
        elif index>=601 and index <=700 :
            freq_count[5] += 1
    '''
'''    
for index, value in enumerate(freq_count) :
    if value >= 60 :
        freq_cnt_bool[index] = True
    else :
        freq_cnt_bool[index] = False
'''
