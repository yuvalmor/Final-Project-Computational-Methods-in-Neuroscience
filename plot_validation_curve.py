# from sklearn.model_selection import learning_curve, validation_curve
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # plot validation curve
# def plot_validation_curve(estimator, title, x, y, ylim=None):
#     train_sizes, train_scores, validation_scores = \
#         learning_curve(estimator, x, y, train_sizes=[0.2, 0.4, 0.6, 0.8,1.0]
#                        , cv=5, shuffle=True)
#     train_error_scores_mean = 1 - np.mean(train_scores, axis=1)
#     validation_error_scores_mean = 1 - np.mean(validation_scores, axis=1)
#
#     # Plot learning curve
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Error")
#     plt.style.use('seaborn')
#     # set x axis values
#     plt.xticks([500, 1000, 1500, 2000, 2500])
#     plt.plot(train_sizes, train_error_scores_mean, label="Training error score")
#     plt.plot(train_sizes, validation_error_scores_mean, label="Validation error score")
#     plt.legend(loc="best")
#     plt.show()