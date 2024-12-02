import numpy as np
import tensorflow as tf

def get_marked_hard_images(resnet, test_dataset):
    correctness_vector = []

    for images, labels in test_dataset:
        # Predict class probabilities
        predictions = resnet.model.predict(images, verbose=0)

        # Get predicted and true classes
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = tf.argmax(labels, axis=1).numpy()

        # Append correctness (1 for correct, 0 for incorrect)
        correctness_vector.extend(predicted_classes == true_classes)

    return np.array(correctness_vector)