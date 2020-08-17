from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from prepare_data import split_data

train_data, train_labels, test_data, test_labels = split_data()
print(train_data)
figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

Length = [sample[1] for sample in train_data]
Diameter = [sample[2] for sample in train_data]
Whole_weight = [sample[4] for sample in train_data]

print(train_data)
print(Length)
print(Diameter)
print(Whole_weight)
# ax.scatter(Length, Diameter, Whole_weight,s=100, edgecolor="r", facecolor="gold")
#
# ax.set_xlabel('Length')
# ax.set_ylabel('Diameter')
# ax.set_zlabel('Whole weight')
#
# plt.show()