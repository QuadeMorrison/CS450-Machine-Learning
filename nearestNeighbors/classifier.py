from numpy import argsort, argmax

def knn(k, train_data, train_targets, test_instance):
    distances = ((train_data - test_instance) ** 2).sum(axis = 1)
    neighbors = [train_targets[i] for i in argsort(distances)[:k]]
    return max(set(neighbors), key=neighbors.count)

class Classifier(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, dataset, targets):
        self.train_data = dataset
        self.train_targets = targets

    def predict(self, dataset):
        return [knn(self.k, self.train_data, self.train_targets, instance)
                for instance in dataset]
