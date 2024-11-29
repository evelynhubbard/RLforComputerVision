from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

class QNetwork:
    def __init__(self, num_actions=10, input_shape=(224, 224, 3)): 
        """
        Creates a custom Q-network using ResNet-50 as the base feature extractor.

        Parameters:
        - num_actions (int): Number of Q-values to predict (one per action).
        - input_shape (tuple): Shape of the input images (height, width, channels).

        Returns:
        - Model: A TensorFlow Keras model for Q-value prediction.
        """
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Builds a Q-network using ResNet-50 as the base feature extractor.

        Returns:
        - Model: A TensorFlow Keras model for Q-value prediction.
        """
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
        )
        base_model.trainable = False

        x = layers.Flatten()(base_model.output)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(self.num_actions, activation='linear')(x)

        model = Model(base_model.input, x)
        
        return model
    
    def compile(self, optimizer='adam', loss='mse', metrics=['mse']):
        """
        Configures the model for training.

        Parameters:
        - optimizer (str): Name of the optimizer.
        - loss (str): Name of the loss function.
        - metrics (list): List of evaluation metrics.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_data, val_data, epochs=10, callbacks=None):
        """
        Trains the Q-network on the given data.

        Parameters:
        - train_data (tf.data.Dataset): Dataset for training.
        - val_data (tf.data.Dataset): Dataset for validation.
        - epochs (int): Number of epochs to train.
        - callbacks (list): List of Keras callbacks for training.
        """
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
        )

    def save(self, filepath):
        """ Saves the Q-network to the given filepath. """
        self.model.save(filepath)

    def load(self, filepath):
        """ Loads the Q-network from the given filepath. """
        self.model = Model.load_model(filepath)