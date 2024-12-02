import tensorflow as tf
from functions.visualization import plot_training_curves

class SecondaryClassifier_NN(tf.keras.Sequential):
    def __init__(self, input_shape = (1024,), num_classes=4, trainable = True, weights_path = None):
        super(SecondaryClassifier_NN, self).__init__()
        """
        Initializes a secondary classifier using a NN.

        Args:
            input_shape (tuple): Shape of the input featuremap
            num_classes (int): Number of classes to predict.
        """
        #self.input_shape = input_shape
        self.num_classes = num_classes
        #self.model = self._build_model()
        self.trainable = trainable

        #build the model
        self.add(tf.keras.layers.InputLayer(input_shape))
        self.add(tf.keras.layers.GlobalAveragePooling2D())
        self.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.add(tf.keras.layers.Dense(512, activation='relu'))
        self.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

    def load(self, weights_path):
        if weights_path:
            resnet_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
            ])
            resnet_dense.load_weights(weights_path)
        
            # Copy weights into current model layers
            for src_layer, target_layer in zip(resnet_dense.layers, self.layers):
                target_layer.set_weights(src_layer.get_weights())
    
 

    
    #def _build_model(self):
        # inputs = tf.keras.layers.Input(shape=self.input_shape)  # Define the input layer
        # #x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Pass inputs to the first layer
        # x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
        # # x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # # x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # x = tf.keras.layers.Dense(512, activation='relu')(x)
        # outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)  # Final layer
        # model = tf.keras.models.Model(inputs=inputs, outputs=outputs)  # Define the model
        # return model
        # model = tf.keras.models.Sequential([
        #     # tf.keras.layers.GlobalAveragePooling2D(Input(shape=self.input_shape)),
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
        plot_training_curves(history)
        return history

    def save(self, filepath):
        """
        Saves the model to the given filepath.

        Args:
            filepath (str): Filepath to save the model.
        """
        self.model.save_weights(filepath)

    #def load(self, filepath):
        """
        Loads the model from the given filepath.

        Args:
            filepath (str): Filepath to load the model.
        """
        #self.model.load_weights(filepath)

    def predict(self, test_feature):
        if len(test_feature.shape) == 1:  # Shape is (H, W, C)
            test_feature = tf.expand_dims(test_feature, axis=0)  # Shape becomes (1, H, W, C)

        predictions = self.model.predict(test_feature)
        return predictions

    def evaluate(self, test_feature_set):
        """
        Evaluates the secondary classifier on the given features and labels.

        Args:
            features (np.ndarray): Feature vectors for evaluation.
            labels (np.ndarray): Labels for evaluation.
        Returns:
            float: Accuracy of the classifier on the given data.
        """
        return self.model.evaluate(test_feature_set)

