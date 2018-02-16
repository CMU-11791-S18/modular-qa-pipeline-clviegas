from Classifier import Classifier
from sklearn import svm

# This is a subclass that extends the abstract class Classifier.


class SVM(Classifier):

    # The abstract method from the base class is implemeted here to return a svm classifier
    def buildClassifier(self, X_features, Y_train):
        clf = svm.SVC().fit(X_features, Y_train)
        return clf
