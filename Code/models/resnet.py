import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore

class ResNetModel:
    def __init__(self, input_shape=(150, 150, 3), num_classes=5, trainable=True):
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
    
    # def extract_features(self, image, layer_name):
    #     feature_extractor = tf.keras.Model(
    #         inputs=self.base_model.input,
    #         outputs=self.base_model.get_layer(layer_name).output
    #     )
    #     return feature_extractor(image, training=False)
    
    def extract_features(self, image, layer_name):
    # # Ensure the input is a tensor

        # If it's a single image, expand dimensions to create a batch of 1
        if len(image.shape) == 3:  # Shape is (H, W, C)
            image = tf.expand_dims(image, axis=0)  # Shape becomes (1, H, W, C)

        # Create the feature extractor
        feature_extractor = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=self.base_model.get_layer(layer_name).output
        )

        # Extract features
        feature_maps = feature_extractor(image, training=False)

        feature_maps = tf.keras.layers.GlobalAveragePooling2D()(feature_maps)
        # # If it was a single image, remove the batch dimension
        # if feature_maps.shape[0] == 1:
        #     feature_maps = tf.squeeze(feature_maps, axis=0)  # Remove batch dimension

        return feature_maps


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

    
    # def save(self, filepath):
    #     #Saves the model to the given filepath.
    #     self.model.save_weights(filepath)

    def save_dense_weights(self, filepath):
        """
        Saves the weights of the dense layers to the given filepath.

        Args:
            filepath (str): Path to save the weights.
        """
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Sequential):
                layer.save_weights(filepath)
                break

    
    def load(self, filepath):
        #Loads the model from the given filepath.
        self.model.load_weights(filepath)

    def evaluate(self, dataset):
        #Evaluates the model on the given dataset.
        return self.model.evaluate(dataset)