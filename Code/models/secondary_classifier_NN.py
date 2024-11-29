import tensorflow as tf
from tf.keras.models import Sequential,layers

class SecondaryClassifier_NN:
    def __init__(self, input_shape = (7,7,2048), num_classes=5):
        """
        Initializes a secondary classifier using a NN.

        Args:
            input_shape (tuple): Shape of the input featuremap
            num_classes (int): Number of classes to predict.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):

        model = Sequential([
            layers.GlobalAveragePooling2D(input_shape=self.input_shape),
            layers.Dense(1024, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
         ])
        
        return model

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
            #Configures the model for training.
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_features, train_labels, val_features, val_labels, epochs=10, batchsize = 32, callbacks=None):
        #Trains the secondary classifier on the given features and labels.
        #train_labels needs to be one-hot encoding
        history = self.model.fit(
            x=train_features,
            y=train_labels,
            validation_data=(val_features, val_labels),
            epochs=epochs,
            batch_size=batchsize,
            callbacks=callbacks
        )
        return history

    def save(self, filepath):
        """
        Saves the model to the given filepath.

        Args:
            filepath (str): Filepath to save the model.
        """
        self.model.save(filepath)

    def load(self, filepath):
        """
        Loads the model from the given filepath.

        Args:
            filepath (str): Filepath to load the model.
        """
        self.model = tf.keras.models.load_model(filepath)

    def evaluate(self, features, labels):
        """
        Evaluates the secondary classifier on the given features and labels.

        Args:
            features (np.ndarray): Feature vectors for evaluation.
            labels (np.ndarray): Labels for evaluation.
        Returns:
            float: Accuracy of the classifier on the given data.
        """
        predictions = self.model.predict(features)
        return accuracy_score(labels, predictions)
