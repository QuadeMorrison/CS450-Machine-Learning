from sklearn import datasets
from sklearn import model_selection as ms
from dataset import Dataset
from sklearn.neighbors import KNeighborsClassifier as knc
from classifier import Classifier
from menu import Menu

def accuracy(target, predicted_target):
    data_length = len(target)
    accuracyArray = [1 if t == pt else 0 for (t, pt) in zip(target, predicted_target)]
    precent = round(accuracyArray.count(1) / data_length * 100, 2)
    return str(precent) + "%" 

def swapData(menu, value):
    return "car" if value == "iris" else "iris"

def splitPercent(menu, value):
    print("% size of train set: ")
    return input()

def howManyNeighbors(menu, value):
    print("How many neighbors? ")
    return input()

def swapAlgorithm(menu, value):
    return "sklearn" if value == "Quade" else "Quade"

def run(menu, value):
    dataset = Dataset(menu.getValue("d") + ".data")
    percent = int(menu.getValue("p")) / 100
    nNeighbors = int(menu.getValue("k"))
    algorithm = menu.getValue("a")
    te_data, tr_data, te_target, tr_target = ms.train_test_split(
        dataset.data, dataset.target, test_size=percent)

    classifier = Classifier(nNeighbors) if algorithm == "Quade" else knc(n_neighbors=nNeighbors)
    classifier.fit(tr_data, tr_target)
    predicted_target = classifier.predict(te_data)
    print("Accuracy of Algorithm:", accuracy(te_target, predicted_target))

def main():
   menu = Menu({ "r" : { "name" : "Run Algorithm",
                         "value" : None,
                         "handler" : run },
                 "d" : { "name" : "Swap dataset (iris or car)",
                         "value" : "iris",
                         "handler" : swapData },
                 "a" : { "name" : "Swap algorithm (Quade or sklearn)",
                         "value" : "Quade",
                         "handler" : swapAlgorithm },
                 "k" : { "name" : "Number of Neighbors",
                         "value" : "3",
                         "handler" : howManyNeighbors },
                 "p" : { "name" : "% size of train set",
                         "value" : "70",
                         "handler" : splitPercent }
   })

   menu.start()


if __name__ == "__main__":
    main()
