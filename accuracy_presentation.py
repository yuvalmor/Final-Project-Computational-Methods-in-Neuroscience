import numpy as np
from tabulate import tabulate


# Represent the model accuracy
def represent_accuracy(model_name, train_error, validation_error, train_error_cv, validation_error_cv):
    # calculate the mean error and calculate the accuracy (the opposite from the error score)
    train_acc = 1 - np.mean(train_error)
    validation_acc = 1 - np.mean(validation_error)
    train_acc_cv = 1 - np.mean(train_error_cv)
    validation_acc_cv = 1 - np.mean(validation_error_cv)
    table = [["Train", str(train_acc)], ["Validation", str(validation_acc)],
             ["Train CV", str(train_acc_cv)], ["Validation CV", str(validation_acc_cv)]]
    headers = [model_name, "Accuracy"]
    print(tabulate(table, headers, tablefmt="github"))
