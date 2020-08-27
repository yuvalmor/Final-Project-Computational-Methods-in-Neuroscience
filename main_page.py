from calculate_errors import calculate_error_for_different_training_sizes, calculate_error_with_cross_validation
from plot_error_curve import plot_error_curve
from prepare_data import prepare_data
from logistic_regression import get_logistic_regression_model

if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = prepare_data()
    logistic_regression_model_with_regularization = get_logistic_regression_model(penalty='l2',
                                                                                  regularization_strength=0.1)
    train_error, validation_error = calculate_error_for_different_training_sizes(
        train_data, train_labels, validation_data, validation_labels, logistic_regression_model_with_regularization)
    plot_error_curve(train_error, validation_error)
    train_error_cv, validation_error_cv = calculate_error_with_cross_validation(
        train_data, train_labels, logistic_regression_model_with_regularization)
