import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras import layers, Model
#from tensorflow.keras import layers, Sequential

ResNet50 = tf.keras.applications.ResNet50
# layers = tf.keras.layers

def get_tall_resnet_CNN(input_shape=(224, 224, 3), trainable=False):
    """
    Returns a ResNet50 model without the top (fully connected) layer.
    Args:
        input_shape (tuple): Shape of the input images.
        trainable (bool): If False, the ResNet layers will be frozen.
    Returns:
        keras.Model: ResNet50 base model without top.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    # Freeze the layers if trainable is set to False
    base_model.trainable = trainable

    tall_model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(4,activation='softmax'), #30 is number of classes
    ])

    return tall_model



def custom_q_network(num_actions=10, input_shape=(224, 224, 3)):
    """
    Creates a custom Q-network using ResNet-50 as the base feature extractor.

    Parameters:
    - num_actions (int): Number of Q-values to predict (one per action).
    - input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
    - Model: A TensorFlow Keras model for Q-value prediction.
    """
    # Load the pre-trained ResNet-50 model
    base_model = ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    classifier_activation='softmax')

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom Q-network layers on top
    x = layers.Flatten()(base_model.output)  # Flatten feature maps
    x = layers.Dense(256, activation='relu')(x)  # Dense layer with 256 units
    q_values = layers.Dense(num_actions)(x)  # Output layer with Q-values

    # Define the final model
    model = Model(inputs=base_model.input, outputs=q_values)
    return model
