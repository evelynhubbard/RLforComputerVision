import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore

class ResNetModel:
    def __init__(self, input_shape, num_classes, layer_name, trainable=True):
        """
        Initializes a ResNet50 model for transfer learning.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layer_name = layer_name
        self.trainable = trainable
        self.base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
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
    
    def extract_features(self, image):
        # Create the feature extractor
        feature_extractor = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=self.base_model.get_layer(self.layer_name).output,
        
        )

        # Extract features
        feature_maps = feature_extractor(image, training=False)
        return feature_maps

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        #configures the model for training.
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_dataset, val_dataset, epochs=10, batchsize=32, callbacks=None):
        #Trains the model on the given dataset.
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch= 1532//batchsize,
            validation_steps= 788//batchsize,
            callbacks=callbacks)
        
        return history

    def save_dense_weights(self, filepath):
        self.model.save_weights(f"{filepath}")

    def load_dense_weights(self, filepath):
        self.model.load_weights(filepath)