import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def gender_converter(data):
    # convert gender to one hot
    temp = np.zeros(shape=(len(data), 3))
    data = np.append(data, temp, axis=1)
    for sample in data:
        if sample[0] == 'I':
            sample[-3] = 1
        if sample[0] == 'F':
            sample[-2] = 1
        if sample[0] == 'M':
            sample[-1] = 1
    return np.delete(data, [0], axis=1)


# Classify Abalone age into three classes
def classify_age(labels):
    for label in range(len(labels)):
        if labels[label] < 7:
            labels[label] = 1
        if 7 <= labels[label] < 14:
            labels[label] = 2
        if 14 <= labels[label]:
            labels[label] = 3
    return labels


def prepare_data():
    # load data set
    data = np.genfromtxt("data/abalone_data", delimiter=",", dtype="str")
    last_column = np.size(data, 1) - 1
    labels = data[:, last_column]
    # remove labels from data
    data = np.delete(data, [last_column], axis=1)
    data=gender_converter(data)
    # convert the data to floats
    data = data.astype(float)
    labels = labels.astype(float)
    labels = classify_age(labels)
    preprocessing.normalize(data)
    # shuffle and split the data
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels)
    return train_data, train_labels, test_data, test_labels
