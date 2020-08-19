import numpy as np

SPLIT_PERCENTAGE = 0.8
UNDER_THRESHOLD = 1
ABOVE_THRESHOLD = 2

gender_converter = {"M": 1, "F": 2, "I": 3}


# Shuffle accordingly the data and the lables
def shuffle_data(data, labels):
    random_order = np.arange(data.shape[0])
    np.random.shuffle(random_order)
    return data[random_order], labels[random_order]


# Split Abalone age into three classes: young (age under 6, adult (age between 6 to 10 ), old (age above 10)
def split_age_to_classes(labels):
    for label in range(len(labels)):
        if labels[label] < 6:
            labels[label] = 1
        if 6 <= labels[label] < 11:
            labels[label] = 2
        if labels[label] >= 11:
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
    for row in data:
        row[0] = gender_converter[row[0]]
    # convert the data to floats
    data = data.astype(float)
    labels = labels.astype(np.double)
    labels = split_age_to_classes(labels)
    # shuffle the data
    data, labels = shuffle_data(data, labels)
    return data, labels


def split_data():
    data, labels = prepare_data()
    num_of_rows = int(SPLIT_PERCENTAGE * (len(data)))
    train_data, train_labels = data[:num_of_rows, :], labels[:num_of_rows]
    test_data, test_labels = data[num_of_rows:, :], labels[num_of_rows:]
    return train_data, train_labels, test_data, test_labels
