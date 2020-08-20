import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from prepare_data import prepare_data
from plot_validation_curve import plot_validation_curve


# not prepare !!! 

def set_model(penalty='none', regularization_strength=1.0, solver='lbfgs', max_iterations=1000):
    return LogisticRegression(penalty=penalty, C=regularization_strength, solver=solver, max_iter=max_iterations)


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = prepare_data()
    model = set_model(penalty='l2', solver='')
    train_data = preprocessing.scale(train_data)
    model.fit(train_data, train_labels)
    plot_validation_curve(model, "logistic regression", train_data, train_labels, ylim=(0.1, 0.4), cv=5)