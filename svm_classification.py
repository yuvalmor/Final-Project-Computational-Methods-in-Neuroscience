from sklearn.model_selection import cross_val_score
from prepare_data import split_data
from sklearn import svm

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = split_data()
    # create svm with C=1- no regularization
    clf = svm.LinearSVC(loss='squared_hinge', C=1, max_iter=10)
    scores = cross_val_score(clf, train_data, train_labels, cv=5)
    print(scores)
    print(2);