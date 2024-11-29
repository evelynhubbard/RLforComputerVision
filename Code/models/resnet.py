import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore

class ResNetModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4, trainable=False):
        """
        Initializes a ResNet50 model for transfer learning.

        Args:
            input_shape (tuple): Shape of the input images.
            num_classes (int): Number of classes to predict.
            trainable (bool): If True, the ResNet layers will be trainable.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.trainable = trainable
        self.model = self._build_model()

    def _build_model(self):
        """
    Returns a ResNet50 model without the top (fully connected) layer.
    Args:
        input_shape (tuple): Shape of the input images.
        trainable (bool): If False, the ResNet layers will be frozen.
    Returns:
        keras.Model: ResNet50 base model without top.
    """
        base_model = ResNet50(
            include_top=True,
            weights='imagenet',
            input_shape=self.input_shape,
        )
        base_model.trainable = self.trainable
        # realised this doesn't go here
        # model = tf.keras.Sequential([
        #     base_model,
        #     tf.keras.layers.GlobalAveragePooling2D(),
        #     tf.keras.layers.Dense(1024,activation='relu'),
        #     tf.keras.layers.Dense(1024,activation='relu'),
        #     tf.keras.layers.Dense(1024,activation='relu'),
        #     tf.keras.layers.Dense(512,activation='relu'),
        #     tf.keras.layers.Dense(self.num_classes,activation='softmax'), #30 is number of classes
        # ])

        return base_model
    
    def get_layer(self, layer_name):
        return self.model.get_layer(layer_name)

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """
        Configures the model for training.

        Args:
            optimizer (str): Name of the optimizer.
            loss (str): Name of the loss function.
            metrics (list): List of evaluation metrics.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, train_data, val_data, epochs=10, callbacks=None):
        """
        Trains the model on the given data.

        Args:
            train_data (tf.data.Dataset): Training data.
            val_data (tf.data.Dataset): Validation data.
            epochs (int): Number of epochs to train.
            callbacks (list): List of Keras callbacks.
        """
        self.model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)

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