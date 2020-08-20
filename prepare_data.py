import numpy as np
from sklearn.model_selection import train_test_split

gender_converter = {"M": 1, "F": 2, "I": 3}


# Classify Abalone age into three classes
def classify_age(labels):
    for label in range(len(labels)):
        if labels[label] < 9:
            labels[label] = 1
        if 9 <= labels[label] < 11:
            labels[label] = 2
        if 11 <= labels[label]:
            labels[label] = 3
    return labels


def prepare_data():
    # load data set
    data = np.genfromtxt("data/abalone_data", delimiter=",", dtype="str")
    last_column = np.size(data, 1) - 1
    labels = data[:, last_column]
    # remove labels from data
    data = np.delete(data, [last_column], axis=1)
    # convert gender to numbers
    for sample in data:
        sample[0] = gender_converter[sample[0]]
    # convert the data to floats
    data = data.astype(float)
    labels = labels.astype(np.double)
    labels = classify_age(labels)
    # shuffle and split the data
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels)
    return train_data, train_labels, test_data, test_labels
