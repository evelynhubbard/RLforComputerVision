import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore

class ResNetModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5, trainable=False, output_layer_name = "conv5_block3_out"):
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
        self.output_layer_name = output_layer_name
        self.model = self._build_model()

    def _build_model(self): 
        #ResNet50 model without the top (fully connected) layer.
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
        )
        base_model.trainable = self.trainable
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(self.num_classes,activation='softmax') 
        ])
        return model
    
    def get_layer(self, layer_name):
        return self.model.get_layer(layer_name)

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        #configures the model for training.
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_dataset, val_dataset, epochs=10, batchsize=32, callbacks=None):
        #Trains the model on the given dataset.

        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks)

    
    def save(self, filepath):
        #Saves the model to the given filepath.
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        #Loads the model from the given filepath.
        self.model.load_weights(filepath)