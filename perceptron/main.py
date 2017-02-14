import pandas as pd
#import numpy as np
from classifier import Classifier
from sklearn import preprocessing

def perceptron():
    inputs, classes = readCsv("iris.data")
    classifier = Classifier()
    classifier.fit(inputs, classes)
    predicted = classifier.predict(inputs)
    print(calcAccuracy(predicted, classes))

def calcAccuracy(predicted, actual):
    correctNum = sum([1 if p == a else 0 for (p, a) in zip(predicted, actual)])
    return correctNum / len(actual) * 100


def readCsv(fileName):
    #csv = np.genfromtxt(fileName, delimiter=",", dtype=None)
    #inputs = [list(data)[:-1] for data in csv]
    #classes = [data[-1] for data in csv]
    #return inputs, classes
    df = pd.read_csv(fileName, header=None)
    class_index = len(list(df)) - 1
    inputs = df.drop(class_index, axis=1)
    classes = df[class_index]
    inputs_norm = (inputs - inputs.mean()) / (inputs.max() - inputs.min())
    return inputs_norm, classes
    
perceptron()
