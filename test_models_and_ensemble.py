from tabulate import tabulate
from models import get_svm_model, get_decision_tree_model, get_logistic_regression_model
from sklearn.metrics import accuracy_score
import numpy as np


def prepare_models_and_prediction(train_data, train_labels, test_data):
    # Creating the best models
    best_svm = get_svm_model(kernel='rbf', gamma=1, C=100)
    best_decision_tree = get_decision_tree_model(criterion='gini', max_depth=5, min_samples_split=200)
    best_logistic_regression = get_logistic_regression_model(penalty='l2', solver='newton-cg', C=100)

    # Train all the models
    best_svm.fit(train_data, train_labels)
    best_decision_tree.fit(train_data, train_labels)
    best_logistic_regression.fit(train_data, train_labels)

    # Predictions on the test set
    svm_prediction = best_svm.predict(test_data)
    decision_tree_prediction = best_decision_tree.predict(test_data)
    logistic_regression_prediction = best_logistic_regression.predict(test_data)

    return svm_prediction, decision_tree_prediction, logistic_regression_prediction


def test_models_accuracy(train_data, train_labels, test_data, test_labels):
    # Get models prediction
    svm_prediction, decision_tree_prediction, logistic_regression_prediction = \
        prepare_models_and_prediction(train_data, train_labels, test_data)

    # Evaluation of the models
    svm_accuracy = accuracy_score(svm_prediction, test_labels)
    decision_tree_accuracy = accuracy_score(decision_tree_prediction, test_labels)
    logistic_regression_accuracy = accuracy_score(logistic_regression_prediction, test_labels)

    # Represent results
    table = [["SVM", str(svm_accuracy * 100)], ["Decision Tree", str(decision_tree_accuracy * 100)],
             ["Logistic Regression", str(logistic_regression_accuracy * 100)]]
    headers = ["Model", "Accuracy %"]
    print(tabulate(table, headers, tablefmt="github"))


def test_ensemble(train_data, train_labels, test_data, test_labels):
    # Get models prediction
    svm_predictions, decision_tree_predictions, logistic_regression_predictions = \
        prepare_models_and_prediction(train_data, train_labels, test_data)

    ensemble_predictions = []

    for sample in range(len(test_data)):
        models_predictions = [svm_predictions[sample], decision_tree_predictions[sample],
                              logistic_regression_predictions[sample]]
        # Takes the commonly value in the array and that's the prediction
        ensemble_predictions.append(np.bincount(models_predictions).argmax())

    ensemble_acc = accuracy_score(ensemble_predictions, test_labels)
    # Represent accuracy
    print("Ensemble Accuracy (%) is:", ensemble_acc*100)
