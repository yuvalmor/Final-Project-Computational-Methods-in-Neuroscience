from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def get_svm_model(**kwargs):
    return svm.SVC(**kwargs)


def get_decision_tree_model(**kwargs):
    return DecisionTreeClassifier(**kwargs)


def get_logistic_regression_model(**kwargs):
    return LogisticRegression(**kwargs)


# def get_logistic_regression_model(penalty='none', regularization_strength=1.0,
#                                   solver='newton-cg', max_iterations=100):
#     return LogisticRegression(penalty=penalty, C=regularization_strength,
#                               multi_class='multinomial', solver=solver, max_iter=max_iterations)



# from sklearn.tree import DecisionTreeClassifier
# from plot_validation_curve import plot_validation_curve as pvc
#
#
#
# def get_dt_estimator():
#     return DecisionTreeClassifier(random_state=0, max_depth=5)
#
# def lerning_curve_dt(train_x, train_y):
#     title = "Learning Curves (Decision Tree)"
#     # build the estimator
#     estimator = get_dt_estimator()
#     pvc(estimator, title, train_x, train_y, ylim=(0.0, 1.0), cv=5, n_jobs=-1)
#
#


#
# #from plot_validation_curve import plot_validation_curve as pvc
# from sklearn.neural_network import MLPClassifier
# from grid_parameter_search import grid_search
# import numpy as np
# import warnings
# warnings.simplefilter("ignore")
#
#
# def get_nn_estimator(alpha=0.0):
#     return MLPClassifier(hidden_layer_sizes=(10, 5, 10), max_iter=70,
#                          activation='relu', solver='adam', alpha=alpha,
#                          learning_rate='constant', learning_rate_init=0.001, shuffle=True)
#
#
# def learning_curve_nn(train_x, train_y, alpha):
#     title = "Learning Curves (NN)"
#     # build the estimator
#     estimator = get_nn_estimator(alpha)
#     pvc(estimator, title, train_x, train_y, ylim=(0.2, 0.5), cv=5,
#         n_jobs=-1)
#
#
# def get_best_parameters(train_x, train_y):
#     param_grid = {'max_iter': [50, 100, 150, 200], 'learning_rate_init': [0.001, 0.01, 0.1],
#                   'activation': ('identity', 'logistic', 'tanh', 'relu')}
#     grid_search(get_nn_estimator(), param_grid, train_x, train_y)
