import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate

from prepare_data import prepare_data
from plot_validation_curve import plot_validation_curve

def set_model(penalty='none', regularization_strength=1.0,
              solver='newton-cg', max_iterations=100):
    return LogisticRegression(penalty=penalty, C=regularization_strength,
    multi_class='multinomial', solver=solver, max_iter=max_iterations)


def lening_curve_lr(train_x, train_y):
    model = set_model()
    # model.fit(train_x,train_y)
    # y_pred= model.predict(train_x)
    # print(accuracy_score(train_y,y_pred))
    # scores = cross_val_score(model,train_x,train_y,cv=5)
    # print(scores)
    plot_validation_curve(model, "logistic regression", train_x, train_y, ylim=(0.1, 1.0))