import math

import sns as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from prepare_data import split_data
import matplotlib.pyplot as plt


def confusion_matrix_data(conf_matrix):
    fix, ax = plt.subplots(figsize=(16, 12))
    plt.suptitle('Confusion Matrix  on Data Set')
    for ii, values in conf_matrix.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(2, 2, ii)  # starts from 1
        plt.title(title);
        sns.heatmap(matrix, annot=True, fmt='');

if __name__ == '__main__':
    # neural network
    scaler = StandardScaler()
    # Fit only to the training data
    train_X, train_y, test_X, test_y = split_data()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    neural_network_class = MLPClassifier(hidden_layer_sizes=(20, 10, 20))
    neural_network_class.fit(train_X, train_y)
    predictions = neural_network_class.predict(test_X)
    y_pred = predictions

    # calculate ROC curve
    # preds = neural_network_class.predict_proba(test_X)[:,1]
    # calculate_roc_curve(test_y, y_pred,3)


    # calculate Confusion Matrix
    # print("Confusion Matrix")
    # calculate_confusion_matrix(test_y, y_pred)

    print("Accuracy of Neural Networks is")
    print(accuracy_score(test_y, y_pred) * 100)

    # Mean Absolute Error

    mae = mean_absolute_error(test_y, y_pred);
    print("MAE:" + str(mae))
    # RMSE
    rmse = math.sqrt(mean_squared_error(test_y, y_pred))
    print("RMSE:" + str(rmse))
    # Median Absolute error
    Medae = median_absolute_error(test_y, y_pred)
    print("Median Absolute Error:" + str(Medae))

    print("Classification report for Test data %s:\n%s\n\n"
          % (scaler, metrics.classification_report(test_y, y_pred)))
