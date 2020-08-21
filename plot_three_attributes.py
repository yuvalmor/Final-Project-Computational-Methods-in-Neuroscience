from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from prepare_data import prepare_data

color_converter = {1: "slategrey", 2: "lightsteelblue", 3: "cornflowerblue"}

train_data, train_labels, test_data, test_labels = prepare_data()
print(train_data)
figure = plt.figure().add_subplot(1, 1, 1, projection='3d')

Length = [sample[0] for sample in train_data]
Diameter = [sample[1] for sample in train_data]
Whole_weight = [sample[3] for sample in train_data]
Labels = [label for label in train_labels]

for i in range(len(train_data)):
    figure.scatter(Length[i], Diameter[i], Whole_weight[i],c=color_converter.get(Labels[i]))

figure.set_xlabel('Length')
figure.set_ylabel('Diameter')
figure.set_zlabel('Whole weight')

plt.show()