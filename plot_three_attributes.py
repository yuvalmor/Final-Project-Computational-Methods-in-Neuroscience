from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from prepare_data import prepare_data

color_converter = {1: "slategrey", 2: "lightsteelblue", 3: "cornflowerblue"}


def plot_three_attributes(train_data, train_labels):
    length = [sample[0] for sample in train_data]
    diameter = [sample[1] for sample in train_data]
    whole_weight = [sample[3] for sample in train_data]
    figure = plt.figure().add_subplot(1, 1, 1, projection='3d')
    for i in range(len(train_data)):
        figure.scatter(length[i], diameter[i], whole_weight[i], c=color_converter.get(train_labels[i]))

    figure.set_xlabel('Length')
    figure.set_ylabel('Diameter')
    figure.set_zlabel('Whole weight')
    plt.show()