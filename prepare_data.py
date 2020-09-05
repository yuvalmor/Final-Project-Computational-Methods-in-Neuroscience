import numpy as np
from sklearn.model_selection import train_test_split


# Convert abalone gender to one hot encoding
def gender_converter(data):
    additional_columns = np.zeros(shape=(len(data), 3))
    data = np.append(data, additional_columns, axis=1)
    for sample in data:
        if sample[0] == 'I':
            sample[-3] = 1
        if sample[0] == 'F':
            sample[-2] = 1
        if sample[0] == 'M':
            sample[-1] = 1
    return np.delete(data, [0], axis=1)


# Classify abalone rings into three classes (ranges)
def classify_rings(labels):
    for label in range(len(labels)):
        if labels[label] < 9:
            labels[label] = 1
        if 9 <= labels[label] < 11:
            labels[label] = 2
        if 11 <= labels[label]:
            labels[label] = 3
    return labels


def prepare_data():
    # Load data set
    data = np.genfromtxt("data/abalone_data", delimiter=",", dtype="str")
    # Set the labels (number of rings - last attribute)
    last_column = np.size(data, 1) - 1
    labels = data[:, last_column]
    # Remove labels from data
    data = np.delete(data, [last_column], axis=1)
    # One-Hot encoding for the gender attribute
    data = gender_converter(data)
    # Convert the data to floats
    data = data.astype(float)
    labels = labels.astype(float)
    # Classify the rings number to three classes
    labels = classify_rings(labels)
    # Shuffle and split the data for: training and test set
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels)
    return train_data, train_labels, test_data, test_labels


def prepare_validation(data, labels):
    # Shuffle and split the data for: training and validation set
    train_data, validation_data, train_labels, validation_labels = train_test_split(data, labels)
    return train_data, train_labels, validation_data, validation_labels
