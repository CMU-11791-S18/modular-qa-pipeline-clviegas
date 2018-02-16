from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer


#This is a subclass that extends the abstract class Featurizer
class TfidfFeaturizer(Featurizer):

	#The abstract method from the base class is implemented here to return tf-idf features
	def getFeatureRepresentation(self, X_train, X_val):
		tfidf_vect = TfidfVectorizer()
		X_train_tfidf = tfidf_vect.fit_transform(X_train)
		X_val_tfidf = tfidf_vect.transform(X_val)

		return X_train_tfidf, X_val_tfidf
