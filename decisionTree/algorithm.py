import numpy as np
import json
from queue import Queue

def calc_entropy(*probabilities):
    return sum([-p * np.log2(p) if p != 0 else 0 for p in probabilities])

def calc_weighted_entropy(classCount, valueTotal):
    total = sum(classCount)
    probabilities = [ c / total if total else 0 for c in classCount]
    return total / valueTotal * calc_entropy(*probabilities)

def get_class_count(value, classSet):
    classList = [v[1] for v in value]
    return [classList.count(c) for c in np.unique(classSet)]

def calc_info_gain(data, classes, feature, entropy):
    values = [row[feature] for row in data]
    dataset = list(zip(values, classes))
    splitValues = [list(filter(lambda d: d[0] == value, dataset)) for value in np.unique(values)]
    classCounts = [get_class_count(value, np.unique(classes)) for value in splitValues]
    return entropy - sum([calc_weighted_entropy(c, len(classes)) for c in classCounts])

def filter_dataset(value, bestFeature, data, classes, features):
    dataset = list(zip(data, classes))
    filteredDataset = list(filter(lambda row: row[0][bestFeature] == value, dataset))
    filteredClasses = [dataset[1] for dataset in filteredDataset]
    filteredData = [dataset[0] for dataset in filteredDataset]
    filteredFeatures = list(filter(lambda f: f != bestFeature, features))
    return filteredData, filteredClasses, filteredFeatures

def make_tree(data, classes, features):
    probabilities = [classes.count(name) / len(classes) for name in np.unique(classes)]
    entropy = calc_entropy(*probabilities)
    nData = len(data)
    tree = {}

    if nData == 0 or len(features) == 0:
        classSet = np.unique(classes)
        commonIndice = np.argsort([classes.count(c) for c in classSet])[0]
        return classSet[commonIndice]
    elif classes.count(classes[0]) == nData:
        return classes[0]
    else:
        bestIndice = np.argsort([calc_info_gain(data, classes, f, entropy) for f in features])[-1]
        bestFeature = features[bestIndice]
        values = [row[bestFeature] for row in data]
        
        for value in np.unique(values):
            fData, fClasses, fFeature = filter_dataset(value, bestFeature, data, classes, features)
            subtree = make_tree(fData, fClasses, fFeature)
            if tree.get(bestFeature):
                tree[bestFeature][value] = subtree
            else:
                tree[bestFeature] = { value : subtree }

    return tree

def print_tree(tree):
    print(json.dumps(tree, indent=2))
        
def print_tree_aux(tree, leveled_tree, level):
    tree_num = 0
    if isinstance(tree, dict):
        leveled_tree[level] = {}
        for key in tree:
            leveled_tree[level][tree_num] += str(key)
            tree_num += 1
            leveled_tree = print_tree_aux(tree, leveled_tree, level + 1)

        return leveled_tree
    else:
        leveled_tree[level][tree_num] = tree
        return leveled_tree
    
