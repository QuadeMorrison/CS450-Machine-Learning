import numpy as np
from algorithm import make_tree

class Classifier(object):

    def __init__(self):
        pass

    def fit(self, data, classes):
        features = range(len(data[0]) - 1)
        self.tree = make_tree(data, classes, features)

    def predict(self, data):
        return [self.classify(d, self.tree) for d in data]

    def classify(self, data, tree):
        if isinstance(tree, dict):
            index = list(tree.keys())[0]
            treeVal = tree[index]
            dataVal = data[index]
            value = dataVal if dataVal in treeVal else list(treeVal.keys())[0]
            return self.classify(data, tree[index][value])
        else:
            return tree

