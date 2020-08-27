from sklearn.linear_model import LogisticRegression


def get_logistic_regression_model(penalty='none', regularization_strength=1.0,
                                  solver='newton-cg', max_iterations=100):
    return LogisticRegression(penalty=penalty, C=regularization_strength,
                              multi_class='multinomial', solver=solver, max_iter=max_iterations)