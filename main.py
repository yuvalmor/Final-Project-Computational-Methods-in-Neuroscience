from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from prepare_data import prepare_data, prepare_validation
from plot_three_attributes import plot_three_attributes
from accuracy_presentation import represent_accuracy
from calculate_errors import calculate_error_for_different_training_sizes, calculate_error_with_cross_validation
from models import get_svm_model, get_decision_tree_model, get_logistic_regression_model
from plot_error_curve import plot_error_curve
from grid_parameter_search import grid_search, get_svm_param_grid, get_dt_param_grid, get_logistic_regression_grid
from test_models_and_ensemble import test_models_accuracy, test_ensemble
import warnings

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

    # Decision Tree model
    decision_tree_model = get_decision_tree_model(criterion="entropy", max_depth=5, min_samples_split=200)
    train_error, validation_error = calculate_error_for_different_training_sizes(
        train_data, train_labels, validation_data, validation_labels, decision_tree_model)
    plot_error_curve(train_error, validation_error)
    train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
        train_data, train_labels, decision_tree_model)

    represent_accuracy("Decision tree", train_error, validation_error, train_error_cv, validation_error_cv)

    # Decision Tree model without regularization
    decision_tree_without_regularization = get_decision_tree_model(criterion="entropy")
    train_error, validation_error = calculate_error_for_different_training_sizes(
        train_data, train_labels, validation_data, validation_labels, decision_tree_without_regularization)
    plot_error_curve(train_error, validation_error)
    train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
        train_data, train_labels, decision_tree_without_regularization)

    represent_accuracy("Decision tree without regularization", train_error, validation_error,
                       train_error_cv, validation_error_cv)

    print(grid_search(get_decision_tree_model(), get_dt_param_grid(), train_data, train_labels))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # Logistic regression model
        logistic_regression_model = get_logistic_regression_model(penalty='l2', solver='lbfgs', C=1)
        train_error, validation_error = calculate_error_for_different_training_sizes(
            train_data, train_labels, validation_data, validation_labels, logistic_regression_model)
        plot_error_curve(train_error, validation_error)
        train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
            train_data, train_labels, logistic_regression_model)

        represent_accuracy("Logistic Regression model", train_error, validation_error,
                           train_error_cv, validation_error_cv)

        # Logistic regression model without regularization
        logistic_regression_without_regularization = get_logistic_regression_model(penalty='none', solver='lbfgs')
        train_error, validation_error = calculate_error_for_different_training_sizes(
            train_data, train_labels, validation_data, validation_labels, logistic_regression_without_regularization)
        plot_error_curve(train_error, validation_error)
        train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
            train_data, train_labels, logistic_regression_without_regularization)

        represent_accuracy("Logistic Regression without regularization", train_error, validation_error,
                           train_error_cv, validation_error_cv)

        print(grid_search(get_logistic_regression_model(), get_logistic_regression_grid(), train_data, train_labels))

    # Evaluating the best models on the test set
    test_models_accuracy(test_data, test_labels, test_data, test_labels)
    # Create and test ensemble model
    test_ensemble(test_data, test_labels, test_data, test_labels)
