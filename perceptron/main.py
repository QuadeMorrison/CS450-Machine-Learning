import pandas as pd
import numpy as np
import random
from sklearn import preprocessing

#constant
BIAS_INPUT = [-1]

def perceptron():
    inputs, classes = readCsv("iris.data")
    input = list(inputs.iloc[[0]])
    nodes = createNodeLayer(2, len(input) + 1)
    doesFireArray = doesFire(input, nodes)
    print(doesFireArray)

def createWeights(featureNum):
    return [random.uniform(-1, 1) for i in range(featureNum)]

def createNodeLayer(nodeNum, featureNum):
    return [createWeights(featureNum) for i in range(nodeNum)]

def calc_threshold(features, weights):
    return sum([f * w for (f, w) in zip(features + BIAS_INPUT, weights)])

def doesFire(features, nodes):
    return [1 if calc_threshold(features, weights) >= 0 else 0 for weights in nodes]

def readCsv(fileName):
    df = pd.read_csv(fileName, header=None)
    class_index = len(list(df)) - 1
    inputs = df.drop(class_index, axis=1)
    classes = df[class_index]
    inputs_norm = (inputs - inputs.mean()) / (inputs.max() - inputs.min())
    return inputs_norm, classes
    

perceptron()
