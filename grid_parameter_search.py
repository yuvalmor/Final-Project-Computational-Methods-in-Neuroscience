from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def grid_search(estimator, param_grid, train_x, train_y):
    clf = GridSearchCV(estimator, param_grid)
    clf.fit(train_x, train_y)
    print("The best parameters are:")
    print(clf.best_params_)
    print(clf.best_score_)

def get_svm_param_grid():
    return {'kernel': ('linear','poly','rbf','sigmoid'),'max_iter': [50, 100, 150, 200,500,700,1000],
            'C': np.logspace(-4,4,20)}

def get_decision_tree_grid():
    return {'criterion':('gini','entropy'),
            'max_depth':[2,5,0,15,20],
            'min_samples_split':[5,10,15,20,30,50]}

def get_logitic_regression_grid():
    return {'penalty':('l1','l2'),
            'solver':('newton-cg','lbfgs','liblinear','sag','saga'),
            'C': np.logspace(-4,4,20)}