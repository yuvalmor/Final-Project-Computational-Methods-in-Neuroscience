from sklearn.model_selection import GridSearchCV
import constants


def grid_search(estimator, param_grid, train_data, train_labels):
    clf = GridSearchCV(estimator, param_grid)
    clf.fit(train_data, train_labels)
    print("The optimal parameters are:")
    print(clf.best_params_)
    print("The best accuracy is:")
    print(clf.best_score_)


def get_svm_param_grid():
    return {'kernel': constants.KERNEL_PARAMETERS,
            'gamma': constants.GAMMA_PARAMETERS,
            'C': constants.C_PARAMETERS}


def get_dt_param_grid():
    return {'criterion': constants.CRITERION_PARAMETERS,
            'max_depth': constants.MAX_DEPTH_PARAMETERS,
            'min_samples_split': constants.MIN_SAMPLE_SPLIT_PARAMETERS}


def get_logistic_regression_grid():
    return {'penalty': constants.PENALTY_PARAMETERS,
            'solver': constants.SOLVER_PARAMETER,
            'C': constants.C_PARAMETERS}
