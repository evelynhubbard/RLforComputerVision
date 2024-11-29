import tensorflow as tf

class SecondaryClassifier_NN:
    #def __init__(self, kernel = "linear", C=1.0):
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """
        Initializes a secondary classifier using a NN.

        Args:
            input_shape (tuple): Shape of the input images.
            num_classes (int): Number of classes to predict.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(self.num_classes,activation='softmax'), #30 is number of classes
        ])
        return model

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
            """
            Configures the model for training.

            Args:
                optimizer (str): Name of the optimizer.
                loss (str): Name of the loss function.
                metrics (list): List of evaluation metrics.
            """
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_features, train_labels, val_features, val_labels, epochs=10, batchsize = 32, callbacks=None):
        """
        Trains the secondary classifier on the given features and labels.

        Args:
            features (np.ndarray): Feature vectors for training.
            labels (np.ndarray): Labels for training.
        """
        self.model.fit(
            x= train_features, 
            y= train_labels, 
            validation_data=(val_features, val_labels),
            epochs=epochs,
            callbacks=callbacks
            )
    
    def save(self, filepath):
        """
        Saves the model to the given filepath.

        Args:
            filepath (str): Filepath to save the model.
        """
        self.model.save(filepath)

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
