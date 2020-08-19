from sklearn.neural_network import MLPClassifier
from prepare_data import prepare_data
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


# plot validation curve
def plot_Validation_curve(estimator, title, X, y, ylim, cv=None,
                          n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")

    train_sizes, train_scores, test_scores, _, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Validation score")
    plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    train_x, train_y, _, _ = prepare_data()
    title = "Learning Curves (NN)"
    # build the estimator
    estimator = nn = MLPClassifier(hidden_layer_sizes=(25, 15, 25), max_iter=300,
                                   activation='relu', solver='adam', alpha=0.0001,
                                   learning_rate='constant', learning_rate_init=0.001)
    plot_Validation_curve(estimator, title, train_x, train_y, ylim=(0.0, 1.01),cv=5,
                           n_jobs=-1)
    plt.show()
