
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
    if (features["401-500"] <= 69.0):
        if (features["501-600"] <= 4.5):
            if (features["100-200"] <= 64.5):
                if (features["201-300"] <= 23.0):
                    if (features["100-200"] <= 27.0):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            if (features["401-500"] <= 42.5):
                if (features["201-300"] <= 32.5):
                    if (features["601-700"] <= 1.0):
                        return False
                    else:
                        if (features["601-700"] <= 18.0):
                            return True
                        else:
                            return False
                else:
                    return False
            else:
                if (features["501-600"] <= 75.0):
                    if (features["201-300"] <= 32.0):
                        return False
                    else:
                        if (features["201-300"] <= 78.0):
                            if (features["100-200"] <= 70.0):
                                if (features["601-700" ]<= 33.5):
                                    return False
                                else:
                                    if (features["201-300"] <= 67.5):
                                        return True
                                    else:
                                        return False
                            else:
                                return True
                        else:
                            return False
                else:
                    return False
    else:
        if (features["201-300"] <= 94.0):
            if (features["501-600"] <= 94.5):
                if (features["100-200"] <= 88.0):
                    if (features["301-400"] <= 63.0):
                        return False
                    else:
                        if (features["201-300"] <= 66.0):
                            if (features["501-600"] <= 66.5):
                                return True
                            else:
                                return False
                        else:
                            return True
                else:
                    if (features["601-700"] <= 52.5):
                        return False
                    else:
                        if (features["301-400"] <= 84.5):
                            if (features["201-300"] <= 79.5):
                                return True
                            else:
                                return False
                        else:
                            return True
            else:
                return False
        else:
            return False



X = []
Y = []
data_feature_names = ["100-200", "201-300",
                      "301-400", "401-500", "501-600", "601-700"]
# getting  the clip names
clip_names = []
directory = "test_samples"
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        clip_names.append("test_samples\\" + filename)

clip_number = 0
for clip in clip_names:
    # count in frequency range
    freq_count = [0 for x in range(6)]
    freq_cnt_bool = [True for x in range(6)]
    check = [0 for c in range(605)]
    fs_rate, signal = wavfile.read(clip)
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    secs = N / float(fs_rate)
    Ts = 1.0 / fs_rate  # sampling interval in time
    # time vector as scipy arange field / numpy.ndarray
    t = scipy.arange(0, secs, Ts)
    FFT = scipy.fft(signal)

    abs_FFT = abs(FFT)[:701]
    for index, value in enumerate(abs_FFT):
        if value >= 0.09 * max(abs_FFT[100:]) and index >= 100 and index <= 700:
            if index >= 100 and index <= 200:
                freq_count[0] += 1
            elif index >= 201 and index <= 300:
                freq_count[1] += 1
            elif index >= 301 and index <= 400:
                freq_count[2] += 1
            elif index >= 401 and index <= 500:
                freq_count[3] += 1
            elif index >= 501 and index <= 600:
                freq_count[4] += 1
            elif index >= 601 and index <= 700:
                freq_count[5] += 1
    for index, value in enumerate(freq_count):
        if value >= 60:
            freq_cnt_bool[index] = True
        else:
            freq_cnt_bool[index] = False
    print(clip + " " + (str)(freq_count) + " " + (str)(freq_cnt_bool))
    X.append(freq_count)
    if "non" in clip:
        Y.append(False)
    else:
        Y.append(True)
    # print (X_train[-1], Y_train)
    features = {}
    for index, value in enumerate(data_feature_names):
        print (index)
        features[value] = freq_count[index]
    if predict(features) :
    	print (clip+" is a Truck sound")
    else :
    	print (clip+" is a Non-Truck sound")    	
    clip_number += 1


