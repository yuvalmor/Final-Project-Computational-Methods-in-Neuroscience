#from plot_validation_curve import plot_validation_curve as pvc
from sklearn.neural_network import MLPClassifier
from grid_parameter_search import grid_search
import numpy as np
import warnings
warnings.simplefilter("ignore")


def get_nn_estimator(alpha=0.0):
    return MLPClassifier(hidden_layer_sizes=(10, 5, 10), max_iter=70,
                         activation='relu', solver='adam', alpha=alpha,
                         learning_rate='constant', learning_rate_init=0.001, shuffle=True)


def learning_curve_nn(train_x, train_y, alpha):
    title = "Learning Curves (NN)"
    # build the estimator
    estimator = get_nn_estimator(alpha)
    pvc(estimator, title, train_x, train_y, ylim=(0.2, 0.5), cv=5,
        n_jobs=-1)


def get_best_parameters(train_x, train_y):
    param_grid = {'max_iter': [50, 100, 150, 200], 'learning_rate_init': [0.001, 0.01, 0.1],
                  'activation': ('identity', 'logistic', 'tanh', 'relu')}
    grid_search(get_nn_estimator(), param_grid, train_x, train_y)
