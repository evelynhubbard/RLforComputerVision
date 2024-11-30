import tensorflow as tf

class SecondaryClassifier_NN:
    def __init__(self, input_shape = (7,7,2048), num_classes=4, trainable = True):
        """
        Initializes a secondary classifier using a NN.

        Args:
            input_shape (tuple): Shape of the input featuremap
            num_classes (int): Number of classes to predict.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.trainable = trainable
    
    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name = "feature_input")  # Define the input layer
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Pass inputs to the first layer
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)  # Final layer
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)  # Define the model
        return model
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.GlobalAveragePooling2D(Input(shape=self.input_shape)),
        #     tf.keras.layers.Dense(1024, activation='relu'),
        #     tf.keras.layers.Dense(1024, activation='relu'),
        #     tf.keras.layers.Dense(1024, activation='relu'),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.Dense(self.num_classes, activation='softmax')
        #  ])   
        # return model

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
            #Configures the model for training.
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_feature_set, val_feature_set, epochs=10, batchsize = 32, callbacks=None):
        #Trains the secondary classifier on the given features and labels.
        #train_labels needs to be one-hot encoding
        history = self.model.fit(
            # x=train_features,
            # y=train_labels,
            # validation_data=(val_features, val_labels),
            train_feature_set,
            validation_data = val_feature_set,
            epochs=epochs,
            batch_size=batchsize,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def save(self, filepath):
        """
        Saves the model to the given filepath.

        Args:
            filepath (str): Filepath to save the model.
        """
        self.model.save_weights(filepath)

    def load(self, filepath):
        """
        Loads the model from the given filepath.

        Args:
            filepath (str): Filepath to load the model.
        """
        self.model.load_weights(filepath)

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
