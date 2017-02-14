from math import exp
import random

#constant
BIAS_INPUT = [-1]

def createWeights(inputNum):
    return [random.uniform(-0.5, 0.5) for i in range(inputNum)]

def createLayer(numInputs, nodeNum):
    return [createWeights(numInputs) for i in range(nodeNum)]

def createLayers(numInputs, *layerNodes):
    createLayerPartial = lambda x: [createLayer(numInputs + 1, x)]
    numNodes = layerNodes[0]
    return createLayerPartial(numNodes) + createLayers(numNodes, *layerNodes[1:]) \
        if len(layerNodes) > 1 else createLayerPartial(numNodes)

def calcThreshold(inputs, weights):
    return sum([i * w for (i, w) in zip(BIAS_INPUT + inputs, weights)])

def sigmoid(x):
    return 1 / (1 + exp(-1 * x))

def layerOutput(inputs, nodes):
    return [sigmoid(calcThreshold(inputs, weights)) for weights in nodes]

def feedForward(inputs, layers):
    return feedForward(layerOutput(inputs, layers[0]), layers[1:]) \
        if layers else inputs
