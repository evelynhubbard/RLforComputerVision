import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore

class ResNetModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5, trainable=True):
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
        self.base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
        )
        self.base_model.trainable = False
        self.model = self._build_model()

    def _build_model(self): 
        #ResNet50 model without the top (fully connected) layers
        model = tf.keras.Sequential([
            self.base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(self.num_classes,activation='softmax') 
        ])
        return model
    
    def extract_features(self, image, layer_name):
        feature_extractor = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=self.base_model.get_layer(layer_name).output
        )
        return feature_extractor(image, training=False)

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

    def evaluate(self, dataset):
        #Evaluates the model on the given dataset.
        return self.model.evaluate(dataset)