from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import decision_tree as dt
from prepare_data import prepare_data
from NN import lerning_curve_nn, get_best_parameters
from grid_parameter_search import grid_search
from sklearn import tree


if __name__ == '__main__':
    train_x, train_y, test_x,test_y=prepare_data()
    lerning_curve_nn(train_x,train_y,0.1)
    #get_best_parameters(train_x,train_y)

    # dt.lerning_curve_dt(train_x,train_y)

