import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def apply_action(image, action):
    """
    Apply a random action to an image.
    """
    if action == 0:
        perm_image = tf.image.rot90(image, k=3)
        
        return perm_image
    elif action == 1:
        return tf.image.rot90(image, k=2)
    elif action == 2:
        transform = tf.convert_to_tensor([[1.0, 0.0, 15.0, 0.0, 1.0, 15.0, 0.0, 0.0]], dtype=tf.float32)
        perm_image = tf.raw_ops.ImageProjectiveTransformV2(
            images = tf.expand_dims(image, axis=0),
            transforms = transform,
            output_shape = tf.shape(image)[:2],
            interpolation = 'BILINEAR'
        )[0]
        return perm_image
    else:
        raise ValueError(f"Invalid action: {action}")

def getM(secondary_classifier, f):
    predictions = secondary_classifier.predict(f)
    predictions = tf.squeeze(predictions, axis=0)

    return np.std(predictions)

def getReward(current_M, new_M):
    if new_M > current_M:
        return 1
    elif new_M == current_M:
        return 0
    else:
        return -1