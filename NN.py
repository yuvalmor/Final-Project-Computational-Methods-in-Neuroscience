from sklearn.model_selection import KFold
from mlxtend.plotting import plot_learning_curves
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from prepare_data import split_data
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import validation_curve


# plot validation curve
def plot_Validation_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                          n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Validation score")
    axes[0].legend(loc="best")
    return plt


if __name__ == '__main__':
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    X, y, test_X, test_y = split_data()
    title = "Learning Curves (NN)"
    # build the estimator- default cv: 5 cross validation
    estimator = nn = MLPClassifier(hidden_layer_sizes=(25, 15, 25), max_iter=300,
                                   activation='relu', solver='adam', alpha=0.0001,
                                   learning_rate='constant', learning_rate_init=0.001)
    plot_Validation_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.0, 1.01),
                           n_jobs=-1)
    plt.show()
