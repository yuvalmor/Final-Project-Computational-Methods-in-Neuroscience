from sklearn.model_selection import GridSearchCV


def grid_search(estimator, param_grid, train_x, train_y):
    clf = GridSearchCV(estimator, param_grid)
    clf.fit(train_x, train_y)
    print("The best parameters are:")
    print(clf.best_params_)
    print(clf.best_score_)

def get_svm_param_grid():
    return {'kernel': ('liner','poly','rbf','sigmoid'),'max_iter': [50, 100, 150, 200,500,700,1000],
            'decision_function_shape': ('ovo','ovr')}

def get_decision_tree_grid():
    return {'criterion':('gini','entropy'), 'max_depth':[2,5,0,15,20], 'min_samples_split':[5,10,15,20,30,50]}
