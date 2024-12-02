from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class SecondaryClassifier_SVM:
    def __init__(self):
        # SVM with linear kernel
        self.model = Pipeline([
            ('scaler', StandardScaler()), 
            ('svm', SVC(kernel='linear', probability=True))
        ])

    def train(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.model.predict(test_features)

    def evaluate(self, test_features, test_labels):
        predictions = self.model.predict(test_features)
        accuracy = (predictions == test_labels).mean()
        return accuracy
