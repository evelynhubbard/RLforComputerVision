from functions import get_marked_hard_images
import tensorflow as tf
import numpy as np
import os

def CNN_classify_test(resnet, test_dataset):
    """
    Calculate the classification accuracy of a model on a test dataset.
    """
    num_correct = 0
    num_total = 0

    # accuracy1 = resnet.evaluate(test_dataset)
    # print(accuracy1)

    for images, labels in test_dataset:
        predictions = resnet.model.predict(images, verbose=0, steps = 1)
        predicted_classes = tf.argmax(predictions, axis=1).numpy()
        true_classes = tf.argmax(labels, axis=1).numpy()
        num_correct += (predicted_classes == true_classes).sum()
        num_total += len(true_classes)
    
    accuracy2 = num_correct / num_total

    num_incorrect = sum(get_marked_hard_images(resnet, test_dataset))

    print("\n=== CNN Classification Test Results ===")
    print(f"Total Images Evaluated : {num_total}")
    print(f"Correctly Classified   : {num_correct}")
    print(f"Incorrectly Classified : {num_incorrect}")
    print(f"Accuracy               : {accuracy2:.2%}")  # Display as percentage
    print("=======================================")

def RL_resnet_classify_test(CNN, q_classifier, marked_hard_dataset, test_dataset,  test_feature_set):
    """
    Calculate the classification accuracy of a Q-learning model on a test dataset using the resnet model as the classifier.
    """
    num_correct = 0
    num_q_correct = 0
    num_total = 0
    num_learning = 0
    just_RL_correct = 0
    j = 0
    # Q-Improvement test
    for idx, (image_batch, labels) in enumerate(test_dataset):
        for batch_idx, image in enumerate(image_batch):
            global_idx = idx * image_batch.shape[0] + batch_idx
            true_class = tf.argmax(labels[batch_idx]).numpy()
            if marked_hard_dataset[global_idx]: # Get the learned best feature map of the image
                image_star = q_classifier.get_learned_image(image, global_idx)

                # Get the predicted class of the image using the Q-Classifier
                second_predictions = CNN.model.predict(tf.expand_dims(image_star,axis=0), verbose=0, steps = 1)
                predicted_class_prime = np.argmax(second_predictions)
                num_learning = num_learning + 1
                just_RL_correct += (predicted_class_prime == true_class)
                
            else: # Get the predicted class of the image using only the CNN
                predictions_new = CNN.model.predict(tf.expand_dims(image,axis=0), verbose=0, steps = 1)
                predicted_class_prime = np.argmax(predictions_new)

            num_q_correct += (predicted_class_prime == true_class)
            num_total += 1
            
    num_incorrect = sum(marked_hard_dataset)
    q_accuracy = num_q_correct / num_total

    # accuracy2 = CNN.evaluate(test_dataset)
    # print(accuracy2)
    print(f"Length of marked_hard_dataset: {len(marked_hard_dataset)}")
    print(f"Sample from marked_hard_dataset: {marked_hard_dataset[:10]}")
    print(f"Total test images: {sum(1 for _ in test_dataset)}")
    print(f"Marked hard images: {sum(marked_hard_dataset)}")


    print("\n=== Classification Test Results for CNN Classifier with Q-Learning ===")
    print(f"Total Images Evaluated : {num_total}")
    print(f"Correctly Classified   : {num_q_correct}")
    print(f"Incorrectly Classified : {num_total-num_q_correct}")
    print(f"Accuracy               : {q_accuracy:.2%}")  # Display as percentage

    print(f"Total Hard Images         : {sum(marked_hard_dataset)}")
    print(f"Correctly Classified by RL: {just_RL_correct}")
    #print(f"RL Improvement            : {q_accuracy - accuracy2:.2%}")  # Display as percentage
    print("=====================================================")

def RL_classify_test(CNN, classifier, q_classifier, marked_hard_dataset, test_dataset, test_feature_set, result_path):
    """
    Calculate the classification accuracy of a Q-learning model on a test dataset.
    """
    num_correct = 0
    num_q_correct = 0
    num_total = 0
    num_learning = 0
    just_RL_correct = 0
    
    for idx, (image_batch, labels) in enumerate(test_dataset):
        for batch_idx, image in enumerate(image_batch):
            global_idx = idx * image_batch.shape[0] + batch_idx

            # Get the true class of the image
            true_class = tf.argmax(labels[batch_idx]).numpy()
            
            # Get the predicted class of the image using the Classifier
            
            feature_map = CNN.extract_features(tf.expand_dims(image,axis=0))
            predictions = classifier.model.predict(feature_map, verbose=0, steps = 1)
            predicted_class = np.argmax(predictions)
            
            num_correct += (predicted_class == true_class)
            num_total += 1

            # Q-Improvement test
            if marked_hard_dataset[global_idx]: # Get the learned best feature map of the image
                if global_idx % 10==0:
                    show_diff = True
                else: show_diff = False

                feature_map_star = q_classifier.get_learned_feature_map(image, global_idx, CNN, show_diff, result_path)

                # Get the predicted class of the image using the Q-Classifier
                second_predictions = classifier.model.predict(feature_map_star, verbose=0, steps = 1)
                predicted_class_prime = np.argmax(second_predictions)
                num_learning = num_learning + 1
                just_RL_correct += (predicted_class_prime == true_class)
                
            else: # Get the predicted class of the image using only the Classifier
                feature_map_star = CNN.extract_features(tf.expand_dims(image,axis=0))
                predictions_new = classifier.model.predict(feature_map_star, verbose=0, steps = 1)
                predicted_class_prime = np.argmax(predictions_new)

            num_q_correct += (predicted_class_prime == true_class)
    
    # num_incorrect = sum(get_marked_hard_images(classifier,test_feature_set))
    # print(num_incorrect)
    accuracy1 = num_correct / num_total
    q_accuracy = num_q_correct / num_total

    # accuracy2 = classifier.evaluate(test_feature_set)
    # print(accuracy2)
    print("\n== Classification Test Results for NN Classifier =")
    print(f"Total Images Evaluated : {num_total}")
    print(f"Correctly Classified   : {num_correct}")
    print(f"Incorrectly Classified : {num_total-num_correct}")
    print(f"Accuracy               : {accuracy1:.2%}")  # Display as percentage
    print("=======================================")
    print("\n=== Classification Test Results for NN Classifier with Q-Learning ===")
    print(f"Total Images Evaluated : {num_total}")
    print(f"Correctly Classified   : {num_q_correct}")
    print(f"Incorrectly Classified : {num_total-num_q_correct}")
    print(f"Accuracy               : {q_accuracy:.2%}")  # Display as percentage

    print(f"Total Hard Images         : {sum(marked_hard_dataset)}")
    print(f"Correctly Classified by RL: {just_RL_correct}")
    print(f"RL Improvement            : {q_accuracy - accuracy1:.2%}")  # Display as percentage
    print("=====================================================")

    return accuracy1, q_accuracy