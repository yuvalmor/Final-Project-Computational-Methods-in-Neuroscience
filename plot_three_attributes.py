from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import constants

def plot_three_attributes(train_data, train_labels):
    x_1 = []
    x_2 = []
    x_3 = []
    y_1 = []
    y_2 = []
    y_3 = []
    z_1 = []
    z_2 = []
    z_3 = []
    for sample in range(len(train_data)):
        if train_labels[sample] == 1:
            x_1.append(train_data[sample][constants.LENGTH])
            y_1.append(train_data[sample][constants.DIAMETER])
            z_1.append(train_data[sample][constants.WHOLE_WEIGHT])
        elif train_labels[sample] == 2:
            x_2.append(train_data[sample][constants.LENGTH])
            y_2.append(train_data[sample][constants.DIAMETER])
            z_2.append(train_data[sample][constants.WHOLE_WEIGHT])
        else:
            x_3.append(train_data[sample][constants.LENGTH])
            y_3.append(train_data[sample][constants.DIAMETER])
            z_3.append(train_data[sample][constants.WHOLE_WEIGHT])

    first_class = (x_1, y_1, z_1)
    second_class = (x_2, y_2, z_2)
    third_class = (x_3, y_3, z_3)

    data = (first_class, second_class, third_class)
    colors = ("slategrey", "lightsteelblue", "cornflowerblue")
    groups = ("class 1", "class 2", "class 3")

    figure = plt.figure()
    figure = figure.add_subplot(1, 1, 1, projection='3d')

    for data, colors, groups in zip(data, colors, groups):
        x, y, z = data
        figure.scatter(x, y, z, c=colors, label=groups, linewidths=0.3, edgecolors='black')

    figure.set_xlabel('Length')
    figure.set_ylabel('Diameter')
    figure.set_zlabel('Whole Weight')

    plt.title('Three Attributes Presentation')
    plt.legend(loc='upper left')
    plt.show()
