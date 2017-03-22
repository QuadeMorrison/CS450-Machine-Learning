from sklearn import preprocessing
from sklearn import model_selection as ms
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.neural_network import MLPClassifier
import numpy as np

def load_csv(filename):
    #get data from csv
    dataset = np.genfromtxt(filename, delimiter=',', dtype=str)

    #Make nominal data numeric because scikitlearn doesn't believe
    #in nominal data
    dataset = np.transpose(dataset)
    for i in range(len(dataset)):
        le = preprocessing.LabelEncoder()
        le = le.fit(np.unique(dataset[i]))
        dataset[i] = le.transform(dataset[i])
    dataset = np.transpose(dataset)

    #Seperate dataset into input and targets
    data = [row[:-1] for row in dataset]

    #Normalize the data otherwise the neural network classifier doesn't work
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    #No need to normalize target
    target = [row[-1] for row in dataset]

    return data, target

def accuracy(predictions, actual):
    correct = sum([1 if p == a else 0 for (p, a) in zip(predictions, actual)])
    return correct / len(actual) * 100

def runAlgo(filename):
    # Read a given dataset to csv and then run it through six different
    # classifiers. 3 Normal classifiers and 3 ensembles
    print(filename)
    d, t = load_csv(filename)
    runModel("Decision Tree", tree.DecisionTreeClassifier(), d, t)
    runModel("KNearesest Ne", knc(), d, t)
    runModel("Neural Networ", MLPClassifier(hidden_layer_sizes=(30,30,30)), d, t)
    runModel("Bagging      ", BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.9, max_features=0.9), d, t)
    runModel("Random Forest", RandomForestClassifier(n_estimators=100), d, t)
    runModel("AdaBoost     ", AdaBoostClassifier(n_estimators=100), d, t)

def runModel(modelName, classifier, d, t):
    # Build a model of a given classifier and print how accurate it is
    train_d, test_d, train_c, test_c = ms.train_test_split(d, t, test_size=0.3)
    classifier.fit(train_d, train_c)
    predictions = classifier.predict(test_d)
    print("\t" + modelName + ": %.2f" % accuracy(predictions, test_c))

runAlgo('house-votes-84.data')
runAlgo('pima-indians-diabetes.data')
runAlgo('crx.data')
