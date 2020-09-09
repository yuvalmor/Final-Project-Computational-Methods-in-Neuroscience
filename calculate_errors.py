from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import constants


def calculate_error_for_different_training_sizes(train_data, train_labels, validation_data, validation_labels, model):
    train_error_list = []
    validation_error_list = []

    for size in constants.TRAIN_SIZES:
        error_train, error_validation = calculate_error(train_data[:size], train_labels[:size],
                                                        validation_data, validation_labels, model)
        train_error_list.append(error_train)
        validation_error_list.append(error_validation)
    return train_error_list, validation_error_list


def calculate_error(train_data, train_labels, validation_data, validation_labels, model):
    # Train the model
    model.fit(train_data, train_labels)
    # Prediction on training set
    train_prediction = model.predict(train_data)
    # Prediction on validation set
    validation_prediction = model.predict(validation_data)
    error_train = 1 - accuracy_score(train_prediction, train_labels)
    error_validation = 1 - accuracy_score(validation_prediction, validation_labels)
    return error_train, error_validation


def calculate_error_with_cross_validation(train_data, train_labels, model):
    cv_results = cross_validate(model, train_data, train_labels, return_train_score=True, cv=5)
    return 1-cv_results['train_score'], 1-cv_results['test_score']