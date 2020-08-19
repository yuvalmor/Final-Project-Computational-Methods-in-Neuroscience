from plot_validation_curve import plot_validation_curve as pvc
from sklearn.neural_network import MLPClassifier
from prepare_data import prepare_data
import matplotlib.pyplot as plt



if __name__ == '__main__':
    train_x, train_y, _, _ = prepare_data()
    title = "Learning Curves (NN)"
    # build the estimator
    estimator = nn = MLPClassifier(hidden_layer_sizes=(20, 8,20), max_iter=100,
                                   activation='relu', solver='adam', alpha=0.01,
                                   learning_rate='constant', learning_rate_init=0.001)
    pvc(estimator, title, train_x, train_y, ylim=(0.2, 0.4),cv=5,
                           n_jobs=-1)
    plt.show()
