from sklearn.model_selection import GridSearchCV
import numpy as np
import constants


def grid_search(estimator, param_grid, train_x, train_y):
    clf = GridSearchCV(estimator, param_grid)
    clf.fit(train_x, train_y)
    print("The optimal parameters are:")
    print(clf.best_params_)
    print("The best accuracy is:")
    print(clf.best_score_)


def get_svm_param_grid():
    return {'kernel': constants.KERNEL_PARAMETERS,
            'gamma': constants.GAMMA_PARAMETERS,
            'C': constants.C_PARAMETERS}


def get_decision_tree_grid():
    return {'criterion':('gini','entropy'),
            'max_depth':[2,5,0,15,20],
            'min_samples_split':[5,10,15,20,30,50]}


def get_logitic_regression_grid():
    return {'penalty':('l1','l2'),
            'solver':('newton-cg','lbfgs','liblinear','sag','saga'),
            'C': np.logspace(-4,4,20)}