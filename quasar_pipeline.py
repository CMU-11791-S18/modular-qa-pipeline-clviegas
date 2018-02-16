# usr/bin/python3
import sys
import json
from sklearn.externals import joblib
import os
from datetime import datetime
from Retrieval import Retrieval
from Featurizer import Featurizer
from CountFeaturizer import CountFeaturizer
from TfidfFeaturizer import TfidfFeaturizer
from Classifier import Classifier
from MultinomialNaiveBayes import MultinomialNaiveBayes
from SVM import SVM
from MLP import MLP
from Evaluator import Evaluator
import pdb
import numpy as np


class Pipeline(object):
    def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance):
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance
        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        trainfile.close()
        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()
        self.question_answering()
        self.PATH = os.path.join('./Results', datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.mkdir(self.PATH)

    def makeXY(self, dataQuestions):
        X = []
        Y = []
        for question in dataQuestions:

            long_snippets = self.retrievalInstance.getLongSnippets(question)
            short_snippets = self.retrievalInstance.getShortSnippets(question)

            X.append(short_snippets)
            Y.append(question['answers'][0])

        return X, Y

    def question_answering(self):
        # pdb.set_trace()
        print('Loading data...')
        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates']
        X_train, Y_train = self.makeXY(self.trainData['questions'][0:30000])  # 31049 questions
        X_val, Y_val_true = self.makeXY(self.valData['questions'])
        # pdb.set_trace()
        # featurization
        print('Feature Extraction...')
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(
            X_train, X_val)
        self.clf = self.classifierInstance.buildClassifier(X_features_train, Y_train)

        # Prediction
        print('Prediction...')
        Y_val_pred = self.clf.predict(X_features_val)

        self.evaluatorInstance = Evaluator()
        a = self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
        p, r, f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)
        print("Accuracy: " + str(a))
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        print("F-measure: " + str(f))

        # Correctly answered questions
        correct_questions_indices = np.where(np.equal(Y_val_pred, Y_val_true))
        correct_questions = X_val[correct_questions_indices]

        # Save predictions in json
        results = {'feature': self.featurizerInstance.__class__.__name__,
                   'classifier': self.classifierInstance.__class__.__name__,
                   'accuracy': a,
                   'precision': p,
                   'recall': r,
                   'F-measure': f,
                   'Correct questions': correct_questions}
        file = open(os.path.join(self.PATH, self.featurizerInstance.__class__.__name__ +
                                 self.classifierInstance.__class__.__name__), 'w', encoding='utf-8')
        json.dump(results, file, ensure_ascii=False)


if __name__ == '__main__':
    trainFilePath = sys.argv[1]  # please give the path to your reformatted quasar-s json train file
    valFilePath = sys.argv[2]  # provide the path to val file
    retrievalInstance = Retrieval()
    # featurizerInstance = CountFeaturizer()
    # classifierInstance = MultinomialNaiveBayes()
    # trainInstance = Pipeline(trainFilePath, valFilePath, retrievalInstance,
    # featurizerInstance, classifierInstance)

    featurizerInstance = TfidfFeaturizer()
    classifierInstance = SVM()
    trainInstance = Pipeline(trainFilePath, valFilePath, retrievalInstance,
                             featurizerInstance, classifierInstance)
