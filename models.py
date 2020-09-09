from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def get_svm_model(**kwargs):
    return svm.SVC(**kwargs)


def get_decision_tree_model(**kwargs):
    return DecisionTreeClassifier(**kwargs)


def get_logistic_regression_model(**kwargs):
    return LogisticRegression(**kwargs)

