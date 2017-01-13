from sklearn import datasets
from random import randrange
from classifier import Classifier

iris = datasets.load_iris()

def knuth_shuffle(data, target):
    "http://rosettacode.org/wiki/Knuth_shuffle#Python"
    for i in range(len(data)-1, 0, -1):
        j = randrange(i + 1)
        data[i], data[j] = data[j], data[i]
        target[i], target[j] = target[j], target[i]

    return data, target

def split_list_percent(data, target, percent):
    split_until = int(round(percent / 100 * (len(iris.data) - 1)))
    shuffled_data, shuffled_target = knuth_shuffle(data, target)

    return (shuffled_data[:split_until], shuffled_data[split_until:],
            shuffled_target[:split_until], shuffled_target[split_until:])

def accuracy(target, predicted_target):
    correct_predictions, data_length = 0, len(target)

    for i in range(data_length):
        if (target[i] == predicted_target[i]):
            correct_predictions += 1

    precent = round(correct_predictions / data_length * 100, 2)
    return str(precent) + "%" 

IrisClassifier = Classifier()

train_data, train_target, test_data, test_target = split_list_percent(iris.data, iris.target, 70)
IrisClassifier.fit(train_data, train_target)
predicted_target = IrisClassifier.predict(test_data)
print(accuracy(test_target, predicted_target))
