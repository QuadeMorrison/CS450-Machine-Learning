import algorithm
import numpy as np

class Classifier(object):
    def fit(self, inputs, target):
        self.classes = np.unique(target)
        self.layers = algorithm.createLayers(inputs.shape[1], 2, 3)

    def predict(self, inputs):
        return [self.classify(algorithm.feedForward(list(inputs.iloc[i]), self.layers)) \
                for i in range(inputs.shape[0])]
        
    def classify(self, outputs):
        return self.classes[outputs.index(max(outputs))]
