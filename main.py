from prepare_data import prepare_data, prepare_validation
from plot_three_attributes import plot_three_attributes
from accuracy_presentation import represent_accuracy

from calculate_errors import calculate_error_for_different_training_sizes, calculate_error_with_cross_validation
from models import get_svm_model, get_decision_tree_model, get_logistic_regression_model
from plot_error_curve import plot_error_curve
import grid_parameter_search as grid_earch
from grid_parameter_search import grid_search, get_svm_param_grid, get_decision_tree_grid, get_logitic_regression_grid

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = prepare_data()
    plot_three_attributes(train_data, train_labels)
    train_data, train_labels, validation_data, validation_labels = prepare_validation(train_data, train_labels)

    # SVM model with regularization
    svm_model = get_svm_model(C=1, kernel='rbf', gamma=1)
    train_error, validation_error = calculate_error_for_different_training_sizes(
        train_data, train_labels, validation_data, validation_labels, svm_model)
    plot_error_curve(train_error, validation_error)
    train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
        train_data, train_labels, svm_model)
    represent_accuracy("SVM model", train_error, validation_error, train_error_cv, validation_error_cv)

    # SVM model without regularization
    svm_without_regularization = get_svm_model(C=10000, kernel='rbf', gamma=1)
    train_error, validation_error = calculate_error_for_different_training_sizes(
        train_data, train_labels, validation_data, validation_labels, svm_without_regularization)
    plot_error_curve(train_error, validation_error)
    train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
        train_data, train_labels, svm_without_regularization)
    represent_accuracy("SVM without regularization", train_error, validation_error, train_error_cv, validation_error_cv)

    print(grid_search(get_svm_model(), get_svm_param_grid(), train_data, train_labels))


    # decision_tree_model = get_decision_tree_model()
    # train_error, validation_error = calculate_error_for_different_training_sizes(
    #     train_data, train_labels, validation_data, validation_labels, decision_tree_model)
    # plot_error_curve(train_error, validation_error)
    # train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
    #     train_data, train_labels, decision_tree_model)


    # decision_tree_model = get_decision_tree_model(criterion="entropy", max_depth=5, min_samples_split=20)
    # decision_tree_model = get_decision_tree_model()
    # train_error, validation_error = calculate_error_for_different_training_sizes(
    #     train_data, train_labels, validation_data, validation_labels, decision_tree_model)
    # plot_error_curve(train_error, validation_error)
    # train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
    #     train_data, train_labels, decision_tree_model)



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
