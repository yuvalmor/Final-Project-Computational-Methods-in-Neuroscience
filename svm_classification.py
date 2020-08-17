#from sklearn import svm
import prepare_data
#from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = prepare_data.split_data()
    print(2);
   # clf = svm.SVC(kernel='linear', C=1)
    #scores = cross_val_score(clf, X, y, cv=5)