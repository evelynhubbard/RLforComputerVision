import tensorflow as tf
from functions.visualization import save_training_curves

class SecondaryClassifier_NN:
    def __init__(self, input_shape, num_classes, trainable = True):
        super(SecondaryClassifier_NN, self).__init__()
        """
        Initializes a secondary classifier using a NN.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.trainable = trainable

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(self.input_shape),
            tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])
        return model

    def load_from_resnet(self, resnet_weights_path):
        # Load weights from a pre-trained ResNet model.
        if resnet_weights_path:
            resnet_classifier_model = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
            resnet_classifier_model.load_weights(resnet_weights_path)

            for classifier_layer, resnet_classifier_layer in zip(self.model.layers, resnet_classifier_model.layers):
                classifier_layer.set_weights(resnet_classifier_layer.get_weights())
                classifier_layer.trainable = False

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
            #Configures the model for training.
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_feature_set, val_feature_set, epochs=10, batchsize = 32, callbacks=None):
        #Trains the secondary classifier on the given features and labels.
        #Train labels needs to be one-hot encoded
        history = self.model.fit(
            train_feature_set,
            validation_data = val_feature_set,
            epochs=epochs,
            steps_per_epoch= 1532//batchsize,
            validation_steps= 788//batchsize,
            batch_size=batchsize,
            callbacks=callbacks,
            verbose=1
        )
        return history
        
    def load_selftrained_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_selftrained_weights(self, filepath):
        #Saves the weights of the secondary classifier to a file.
        self.model.save_weights(filepath)

