from sklearn.model_selection import GridSearchCV

def grid_search(estimator,param_grid,train_x,train_y):
    clf= GridSearchCV(estimator, param_grid)
    clf.fit(train_x,train_y)
    print("The best parameters are:")
    print(clf.best_params_)