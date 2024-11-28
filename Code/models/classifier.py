import tensorflow as tf
from tensorflow.keras import layers, models

def secondary_classifier(input_dim):
    """
    Returns a neural network classifier operating on the feature map.
    """
    secondary_nn = models.Sequential([
        Flatten(input_shape=input_dim),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(30, activation='softmax')  # Number of classes
    ])
    secondary_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
