import algorithm as a
import numpy as np

class Classifier(object):
    def fit(self, inputs, target):
        self.classes = np.unique(target)
        self.layers = a.createLayers(inputs.shape[1], 3, len(self.classes))
        for i in range(1000):
            for i in range(len(inputs)):
                inputList = list(inputs.iloc[i])
                activations = [inputList] + a.feedForward(inputList, self.layers)
                self.layers = a.backPropagation(activations, self.layers,
                                                self.classes.tolist().index(target.iloc[0]))

    def predict(self, inputs):
        return [self.classify(a.feedForward(list(inputs.iloc[i]), self.layers)) \
                for i in range(inputs.shape[0])]
        
    def classify(self, outputs):
        return self.classes[outputs.index(max(outputs))]
