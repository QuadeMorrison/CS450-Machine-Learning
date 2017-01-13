class Classifier(object):

    def fit(self, dataset, targets):
        pass

    def predict(self, dataset):
        return [x*0 for x in range(len(dataset))]
