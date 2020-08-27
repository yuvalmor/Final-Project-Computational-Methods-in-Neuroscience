from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_sizes = [50, 250, 500, 750, 1000, 1500, 2000]


def calculate_error(train_x, train_y, validation_x, validation_y, model):
    model.fit(train_x, train_y)
    train_y_hat = model.predict(train_x)
    validation_y_hat = model.predict(validation_x)
    error_train = 1 - accuracy_score(train_y_hat, train_y)
    error_validation = 1 - accuracy_score(validation_y_hat, validation_y)
    return error_train, error_validation


def plot_error_curve(train_x, train_y, model):
    train_data, validation_data, train_labels, validation_labels = train_test_split(train_x, train_y)
    train_error_list = []
    validation_error_list = []
    for size in train_sizes:
        error_train, error_validation = calculate_error(train_data[:size], train_labels[:size],
                                                        validation_data, validation_labels, model)
        train_error_list.append(error_train)
        validation_error_list.append(error_validation)

    # Plot learning curve
    plt.title("Error score as function of train size")
    plt.xlabel("Train size")
    plt.ylabel("Error score")
    plt.style.use('seaborn')
    # set x axis values
    #plt.xticks(train_sizes)
    #plt.ylim(0.1, 1)
    plt.plot(train_sizes, train_error_list, label="Training error score")
    plt.plot(train_sizes, validation_error_list, label="Validation error score")
    plt.legend(loc="best")
    plt.show()






