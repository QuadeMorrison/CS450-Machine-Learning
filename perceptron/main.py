import pandas as pd
#import numpy as np
from classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import algorithm

def perceptron():
    train_d, test_d, train_c, test_c = readCsv("iris.data")
    classifier = Classifier()
    classifier.fit(train_d, train_c)
    predicted = classifier.predict(test_d)
    print(calcAccuracy(predicted, test_c))

def calcAccuracy(predicted, actual):
    correctNum = sum([1 if p == a else 0 for (p, a) in zip(predicted, actual)])
    return correctNum / len(actual) * 100


def readCsv(fileName):
    df = pd.read_csv(fileName, header=None)
    class_index = len(list(df)) - 1
    inputs = df.drop(class_index, axis=1)
    classes = df[class_index]
    inputs_norm = (inputs - inputs.mean()) / (inputs.max() - inputs.min())
    train_d, test_d, train_c, test_c = train_test_split(inputs_norm, classes, test_size = 0.3)
    return train_d, test_d, train_c, test_c
    
perceptron()
