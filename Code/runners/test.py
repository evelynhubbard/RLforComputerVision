#from models import ResNetModel
from functions.feature_extraction import extract_feature_maps
import tensorflow as tf
import numpy as np

#def evaluate_model(model_path, dataset, is_primary=True):
    # """
    # Evaluate a model (primary CNN or secondary classifier) on a dataset.

    # Args:
    #     model_path (str): Path to the saved model.
    #     dataset (tf.data.Dataset): Dataset for evaluation.
    #     is_primary (bool): Whether the model is a primary CNN or secondary classifier.

    # Returns:
    #     float: Accuracy of the model on the dataset.
    # """
    # if is_primary:
    #     # Load and evaluate primary CNN
    #     resnet_model = ResNetModel()
    #     resnet_model.load(model_path)
    #     _, accuracy = resnet_model.model.evaluate(dataset)
    #     print(f"Primary Model Accuracy: {accuracy:.2f}")
    #     return accuracy
    # else:
    #     # Load and evaluate secondary classifier
    #     # Extract features using the primary CNN
    #     features, labels = extract_feature_maps(resnet_model.model, dataset)

    #     # Load the secondary classifier
    #     import pickle
    #     with open(model_path, "rb") as f:
    #         secondary_classifier = pickle.load(f)

    #     # Evaluate the secondary classifier
    #     predictions = secondary_classifier.predict(features)
    #     accuracy = np.mean(predictions == labels)
    #     print(f"Secondary Classifier Accuracy: {accuracy:.2f}")
    #     return accuracy
    
    # import numpy as np

def basic_classify_test(resnet, test_dataset):
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
        predictions = resnet.model.predict(images, verbose=0)
        predicted_classes = tf.argmax(predictions, axis=1).numpy()
        true_classes = tf.argmax(labels, axis=1).numpy()
        num_correct += (predicted_classes == true_classes).sum()
        num_total += len(true_classes)
    
    accuracy1 = num_correct / num_total

    accuracy2 = resnet.evaluate(test_dataset)

    return accuracy1, accuracy2

def RL_classify_test(resnet, classifier_model, q_classifier, marked_hard_dataset, test_dataset):
    """
    Calculate the classification accuracy of a Q-learning model on a test dataset.

    Args:
        resnet_model (tf.keras.Model): The ResNet model to evaluate.
        classifier_model (SecondaryClassifier): The secondary classifier model.
        q_classifier (QClassifier): The Q-learning classifier.
        test_dataset (tf.data.Dataset): The test dataset.

    Returns:
        float: The classification accuracy of the Q-learning model on the test dataset.
    """
    num_correct = 0
    num_q_correct = 0
    num_total_1 = 0
    num_total_2 = 0
    
    for idx, (image_batch, labels) in enumerate(test_dataset):
        for i, image in enumerate(image_batch):
            true_class = tf.argmax(labels[i]).numpy()
            global_idx = idx * len(image_batch) + i
            #predictions = resnet.model.predict(tf.expand_dims(image, axis=0), verbose=0)
            #predicted_classes = tf.argmax(predictions, axis=1).numpy()
            feature_map = resnet.extract_features(image, 'conv5_block3_out')
            feature_map = tf.keras.layers.Dense(1024, activation='relu')(feature_map)
            predictions = classifier_model.predict(feature_map)
            predicted_class = np.argmax(predictions)
            
            num_correct += (predicted_class == true_class)
            num_total_1 += 1
        
        # for i, image in enumerate(image_batch):
        #     true_class = tf.argmax(labels[i]).numpy()
        #     global_idx = idx * len(image_batch) + i
            if marked_hard_dataset[global_idx]:
                #feature_getter = resnet.model.predict(tf.expand_dims(image, axis=0), verbose=0)
                feature_map_prime = resnet.extract_features(image, 'conv5_block3_out')
                feature_map_prime = tf.keras.layers.Dense(1024, activation='relu')(feature_map_prime)
                predictions_prime = classifier_model.predict(feature_map_prime)
                #predicted_classes = tf.argmax(predictions, axis=1).numpy()
                predicted_class_prime = np.argmax(predictions_prime)
            else:
                feature_map_prime = q_classifier.classify_image(image, global_idx, resnet, classifier_model)
                secondary_predictions = classifier_model.predict(feature_map_prime)
                #predicted_classes.append(np.argmax(secondary_predictions))
                predicted_class_prime = np.argmax(secondary_predictions)
            
            num_q_correct += (predicted_class_prime == true_class)
            num_total_2 += 1
    
    accuracy = num_correct / num_total_1
    q_accuacy = num_q_correct / num_total_2
    return accuracy, q_accuacy

 
    

     




