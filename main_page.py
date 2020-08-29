import numpy as np
from sklearn import tree
from calculate_errors import calculate_error_for_different_training_sizes, calculate_error_with_cross_validation
from models import get_svm_model, get_logistic_regression_model, get_decision_tree_model
from plot_error_curve import plot_error_curve
from prepare_data import prepare_data

if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = prepare_data()

    decision_tree_model = get_decision_tree_model(max_depth=5, min_samples_split=20)

    decision_tree_model = decision_tree_model.fit(train_data, train_labels)
    tree.plot_tree(decision_tree_model)


    train_error, validation_error = calculate_error_for_different_training_sizes(
        train_data, train_labels, validation_data, validation_labels, decision_tree_model)
    plot_error_curve(train_error, validation_error)
    train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
        train_data, train_labels, decision_tree_model)

    print(np.mean(train_error))
    print(np.mean(validation_error))
    print(np.mean(train_error_cv))
    print(np.mean(validation_error_cv))

    # svm_model = get_svm_model(C=1, kernel='rbf')
    # train_error, validation_error = calculate_error_for_different_training_sizes(
    #     train_data, train_labels, validation_data, validation_labels, svm_model)
    # plot_error_curve(train_error, validation_error)
    # train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
    #     train_data, train_labels, svm_model)
    #
    # print(train_error)
    # print(np.mean(validation_error))
    # print(train_error_cv)
    # print(np.mean(validation_error_cv))
    #
    # svm_model_without_regularization = get_svm_model(C=10000, kernel='rbf')
    # train_error, validation_error = calculate_error_for_different_training_sizes(
    #     train_data, train_labels, validation_data, validation_labels, svm_model_without_regularization)
    # plot_error_curve(train_error, validation_error)
    # train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
    #     train_data, train_labels, svm_model_without_regularization)
    #
    # print(train_error)
    # print(np.mean(validation_error))
    # print(train_error_cv)
    # print(np.mean(validation_error_cv))



    # logistic_regression_model_with_regularization = get_logistic_regression_model(penalty='l2', C=0.1)
    # train_error, validation_error = calculate_error_for_different_training_sizes(
    #     train_data, train_labels, validation_data, validation_labels, logistic_regression_model_with_regularization)
    # plot_error_curve(train_error, validation_error)
    # train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
    #     train_data, train_labels, logistic_regression_model_with_regularization)
