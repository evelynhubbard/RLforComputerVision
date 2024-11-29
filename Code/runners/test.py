from models import ResNetModel, SecondaryClassifier_NN, SecondaryClassifier_SVM
from functions.feature_extraction import extract_feature_maps
import tensorflow as tf
import numpy as np

def evaluate_model(model_path, dataset, is_primary=True):
    """
    Evaluate a model (primary CNN or secondary classifier) on a dataset.

    Args:
        model_path (str): Path to the saved model.
        dataset (tf.data.Dataset): Dataset for evaluation.
        is_primary (bool): Whether the model is a primary CNN or secondary classifier.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    if is_primary:
        # Load and evaluate primary CNN
        resnet_model = ResNetModel()
        resnet_model.load(model_path)
        _, accuracy = resnet_model.model.evaluate(dataset)
        print(f"Primary Model Accuracy: {accuracy:.2f}")
        return accuracy
    else:
        # Load and evaluate secondary classifier
        # Extract features using the primary CNN
        features, labels = extract_feature_maps(resnet_model.model, dataset)

        # Load the secondary classifier
        import pickle
        with open(model_path, "rb") as f:
            secondary_classifier = pickle.load(f)

        # Evaluate the secondary classifier
        predictions = secondary_classifier.predict(features)
        accuracy = np.mean(predictions == labels)
        print(f"Secondary Classifier Accuracy: {accuracy:.2f}")
        return accuracy
    
    import numpy as np
import os

def basic_classify_test(model, test_dataset):
    """
    Calculate the classification accuracy of a model on a test dataset.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        test_dataset (tf.data.Dataset): The test dataset.

    Returns:
        float: The classification accuracy of the model on the test dataset.
    """
    num_correct = 0
    num_total = 0

    for images, labels in test_dataset:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)

        num_correct += np.sum(predicted_classes == true_classes)
        num_total += len(true_classes)

    accuracy = num_correct / num_total
    return accuracy

def RL_classify_test(resnet_model, classifier_model, q_classifier, marked_hard_dataset, test_dataset):
    """
    Calculate the classification accuracy of a Q-learning model on a test dataset.

    Args:
        model (QClassifier): The Q-learning model to evaluate.
        test_dataset (tf.data.Dataset): The test dataset.

    Returns:
        float: The classification accuracy of the Q-learning model on the test dataset.
    """
    num_correct = 0
    num_total = 0

    for images, labels in test_dataset:
        true_classes = np.argmax(labels, axis=1)
        if marked_hard_dataset:
            predicted_classes = np.argmax(resnet_model.model.predict(images), axis=1)
        else:
            predicted_classes = q_classifier.classify_image(images, classifier_model)
        
        num_correct += np.sum(predicted_classes == true_classes)
        num_total += len(true_classes)
    
    accuracy = num_correct / num_total
    return accuracy




