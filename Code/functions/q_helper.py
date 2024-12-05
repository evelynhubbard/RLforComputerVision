import numpy as np
import tensorflow as tf

def apply_action(image, action):
    """
    Apply a random action to an image.
    """
    if action == 0:
        perm_image = tf.image.rot90(image, k=3)
    elif action == 1:
        perm_image =  tf.image.rot90(image, k=2)
    elif action == 2:
        transform = tf.convert_to_tensor([[1.0, 0.0, 15.0, 0.0, 1.0, 15.0, 0.0, 0.0]], dtype=tf.float32)
        transform = tf.reshape(transform[:2],[-1])
        perm_image = tf.raw_ops.ImageProjectiveTransformV2(
            images = tf.expand_dims(image, axis=0), # Add batch dimension
            transforms = tf.expand_dims(transform, axis = 0),
            output_shape = tf.shape(image)[:2],
            interpolation = 'BILINEAR'
        )
        perm_image = tf.squeeze(perm_image, axis=0)  # Remove batch dimension

    else:
        raise ValueError(f"Invalid action: {action}")
    
    return perm_image

def getM(secondary_classifier, f):
    # Get the standard deviation of the predictions

    predictions = secondary_classifier.model.predict(f, verbose=0, steps = 1)
    predictions = tf.squeeze(predictions, axis=0)

    return np.std(predictions)

def getReward(current_M, new_M):
    # Compute the reward based on the change in the metric M

    if new_M > current_M:
        return 1
    elif new_M == current_M:
        return 0
    else:
        return -1
    
def get_marked_hard_images(resnet, test_dataset):
    """
    Get a vector of marked hard images from the test dataset. Marked hard images are images that were incorrectly classified by the CNN.
    """
    marked_hard_vector = []

    for images, labels in test_dataset:
        # Predict class probabilities
        predictions = resnet.model.predict(images, verbose=0, steps = 1)

        # Get predicted and true classes
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = tf.argmax(labels, axis=1).numpy()

        # Append correctness 
        batch_marked_hard_vector = np.not_equal(predicted_classes, true_classes).astype(int)
        
        marked_hard_vector.extend(batch_marked_hard_vector)

    return marked_hard_vector