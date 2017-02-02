import numpy as np
from classifier import Classifier
from sklearn import model_selection as ms
from sklearn import tree
from sklearn import preprocessing as pp
from algorithm import print_tree

def read_csv(filename):
    csv = np.genfromtxt(filename, delimiter=",", dtype=None)
    data = [list(row)[:-1] for row in csv]
    binnedData = binData(data)
    classes = [str(row[-1]) for row in csv]
    return binnedData, classes

def binData(data):
    binnedData = []
    for row in np.transpose(data):
        if isinstance(row[0], float):
            rMin = min(row)
            rMax = max(row)
            bins = np.arange(rMin, rMax + 1, (rMax - rMin) / 9)
            intBinnedData = np.digitize(row, bins)
            binnedData.append([str(item) for item in intBinnedData])
        else:
            binnedData.append([str(item) for item in row])
    return np.transpose(binnedData)

def accuracy(predicted, real):
    correct = sum([1 if a == b else 0 for (a, b) in zip(predicted, real)])
    print("Accuracy: " + str(correct / len(predicted) * 100) + "%")

def convertDataToInt(data, classes):
    le = pp.LabelEncoder()
    intData = []
    for row in np.transpose(data):
        le.fit(row)
        intData.append(le.transform(row))
    le.fit(classes)
    return np.transpose(intData), le.transform(classes)

def whichDataSet():
    print("Which dataset? Iris [i], Lenses [l], House Votes [h]")
    option = input()

    return {
        "l": "lenses.data",
        "h": "house-votes-84.data",
    }.get(option, "iris.data")

def whichClassifier():
    print("Which classifier? Quade [q], SciKitLearn [s]")
    option = input()

    return {
        "s": tree.DecisionTreeClassifier()
    }.get(option, Classifier())

def displayTree():
    print("Print tree? yes [y], no [n]")
    option = input()

    return True if option == "y" else False

def main():
    filename = whichDataSet()
    data, classes = read_csv(filename)
    classifier = whichClassifier()

    # SciKitLearn won't take the house-votes-data as strings so
    # I have to change the data to ints for this case
    if not isinstance(classifier, Classifier):
        if filename == "house-votes-84.data":
            data, classes = convertDataToInt(data, classes)
            
    train_d, test_d, train_c, test_c = ms.train_test_split(
        data, classes, test_size=0.3)

    classifier.fit(train_d, train_c)
    predicted = classifier.predict(test_d)
    accuracy(predicted, test_c)

    if isinstance(classifier, Classifier):
        if displayTree(): print_tree(classifier.tree)

main()
