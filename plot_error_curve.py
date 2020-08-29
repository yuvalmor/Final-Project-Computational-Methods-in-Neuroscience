import matplotlib.pyplot as plt
import constants


def plot_error_curve(training_error_score, validation_error_score):
    plt.title("Error score as function of train size")
    plt.xlabel("Train size")
    plt.ylabel("Error score")
    plt.ylim(0.0, 1.0)
    plt.plot(constants.TRAIN_SIZES, training_error_score, label="Training error score", color="slategrey")
    plt.plot(constants.TRAIN_SIZES, validation_error_score, label="Validation error score", color="lightsteelblue")
    plt.scatter(constants.TRAIN_SIZES, training_error_score, color="slategrey")
    plt.scatter(constants.TRAIN_SIZES, validation_error_score,  color="lightsteelblue")
    plt.legend(loc="best")
    plt.show()





