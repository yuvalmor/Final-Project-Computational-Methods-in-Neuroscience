from sklearn.tree import DecisionTreeClassifier
from plot_validation_curve import plot_validation_curve as pvc



def get_dt_estimator():
    return DecisionTreeClassifier(random_state=0, max_depth=5)

def lerning_curve_dt(train_x, train_y):
    title = "Learning Curves (Decision Tree)"
    # build the estimator
    estimator = get_dt_estimator()
    pvc(estimator, title, train_x, train_y, ylim=(0.0, 1.0), cv=5, n_jobs=-1)


