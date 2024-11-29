import numpy as np
import tensorflow as tf

def get_marked_hard_images(resnet_model, test_dataset):
    correctness_vector = []

    for images, labels in test_dataset:
        # Predict class probabilities
        predictions = resnet_model.model.predict(images)

        # Get predicted and true classes
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = tf.argmax(labels, axis=1).numpy()

        # Append correctness (1 for correct, 0 for incorrect)
        correctness_vector.extend(predicted_classes == true_classes)

    return np.array(correctness_vector)
    # misclassified_images = []
    # misclassified_labels = []

    # for images, labels in test_dataset:
    #     predictions = resnet_model.model.predict(images)
    #     predicted_classes = np.argmax(predictions, axis=1)
    #     true_classes = tf.argmax(labels, axis=1).numpy()

    #     misclassified_indices = np.where(predicted_classes != true_classes)[0]

    #     for idx in misclassified_indices:
    #         misclassified_images.append(images[idx].numpy())
    #         misclassified_labels.append(true_classes[idx].numpy())
    
    # # Convert to NumPy arrays
    # hard_images = np.array(misclassified_images)
    # hard_labels = np.array(misclassified_images)

    # hard_dataset = tf.data.Dataset.from_tensor_slices((hard_images, hard_labels))

    # return hard_dataset