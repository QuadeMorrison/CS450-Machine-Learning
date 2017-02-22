from math import exp
import random
import numpy as np

#constant
BIAS_INPUT = [-1]

def createWeights(numInputs):
    return [random.uniform(-0.5, 0.5) for i in range(numInputs)]

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
    return 1 / (1 + 2.71828**-x)

def layerOutput(inputs, nodes):
    return [sigmoid(calcThreshold(inputs, weights)) for weights in nodes]

def feedForward(inputs, layers):
    output = layerOutput(inputs, layers[0])
    return [output] + feedForward(output, layers[1:]) \
        if len(layers) > 1 else [output]

def outputError(outputs, target):
    targetVector = [0] * len(outputs)
    targetVector[target] = 1
    return [o * (1 - o) * (o - t) for (o, t) in zip(outputs, targetVector)]

def hiddenError(activation, nodes, kError):
    def nodeError(weights):
        return sum([w * e for (w, e) in zip(weights, kError)])

    return [a * (1 - a) * nodeError(n) for (a, n) in zip(activation, nodes)]

def updateLayer(activations, layer, error):
    def updateWeight(node):
        return [w - 0.5 * e * a for (w, a, e) in zip(node, activations, error)]

    return [updateWeight(n) for n in layer]
    
def backPropagation(activations, layers, target):
    error = outputError(activations[-1], target)
    newLayers = []

    for i in range(1, len(layers) + 1):
        nodes = np.transpose(layers[-i]);
        error = hiddenError(activations[-i], nodes, error)
        newNodes = updateLayer(BIAS_INPUT + activations[-(i + 1)], nodes, error)
        newLayers = [np.transpose(newNodes).tolist()] + newLayers

    return newLayers

def perceptron(inputs, targets, iterations):
    layers = createLayers(inputs.shape[1], 2, len(targets))

    for i in range(len(inputs)):
        activations = feedForward()
    
