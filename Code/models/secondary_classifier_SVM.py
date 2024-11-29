from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SecondaryClassifier_SVM:
    def __init__(self, kernel = "linear", C=1.0):
        """
        Initializes a secondary classifier using a Support Vector Machine (SVM).

        Args:
            kernel (str): Kernel type for the SVM.
            C (float): Penalty parameter of the error term.
        """
        self.classifier = SVC(kernel=kernel, C=C)
        
    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
            """
            Configures the model for training.

            Args:
                optimizer (str): Name of the optimizer.
                loss (str): Name of the loss function.
                metrics (list): List of evaluation metrics.
            """
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self,features, labels):
        """
        Trains the secondary classifier on the given features and labels.

        Args:
            features (np.ndarray): Feature vectors for training.
            labels (np.ndarray): Labels for training.
        """
        self.classifier.fit(features, labels)

    def evaluate(self, features, labels):
        """
        Evaluates the secondary classifier on the given features and labels.

        Args:
            features (np.ndarray): Feature vectors for evaluation.
            labels (np.ndarray): Labels for evaluation.
        Returns:
            float: Accuracy of the classifier on the given data.
        """
        predictions = self.classifier.predict(features)
        return accuracy_score(labels, predictions)
