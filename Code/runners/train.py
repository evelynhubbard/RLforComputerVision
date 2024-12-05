import os
import numpy as np
from functions import q_helper, visualization
import tensorflow as tf

def train_CNN(tall_resnet, train_dataset, val_dataset, resnet_save_path, results_path, epochs=10, batch_size=32):
    """
    Fine tunes pre-trained ResNet-50 (just the added layers) model on the dataset.
    """
    history = tall_resnet.train(train_dataset, val_dataset, epochs=epochs, batchsize=batch_size)

    # Save the training curves
    visualization.save_training_curves(history, results_path, title = "Training Curves for CNN")

    tall_resnet.save_dense_weights(resnet_save_path)

def train_secondary_classifier(secondary_NN, train_feature_set, val_feature_set, classifier_save_path, results_path, epochs=10, batch_size=32):
    """
    Trains the secondary classifier on the given features.
    """
    history = secondary_NN.train(
        train_feature_set,
        val_feature_set,
        epochs=epochs, 
        batchsize=batch_size
    )

    # Save the training curves
    visualization.save_training_curves(history, results_path, title = "Training Curves for Classifying NN")

    secondary_NN.save_selftrained_weights(classifier_save_path)

def train_q_tables(q_classifier, CNN, classifier, dataset, marked_hard_images, output_dir):
    """
    Trains Q-tables for each hard image in the dataset.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each batch in the dataset
    max_converge_index = 0
    for idx, (image_batch, _) in enumerate(dataset):
        for batch_idx, image in enumerate(image_batch):
            global_idx = idx * len(image_batch) + batch_idx   
            converge_index = 0
            # Skip easy images
            if marked_hard_images[global_idx]: 

                # Initialize Q-table and initial state
                #q_table = np.zeros((2, q_classifier.num_actions)) 
                q_table = np.random.uniform(low=-0.01, high=0.01, size=(2, q_classifier.num_actions))
                current_state = 0

                # Compute the initial metric M
                image = tf.expand_dims(image, axis=0) # Add batch dimension
                f_sample = CNN.extract_features(image)
                image = tf.squeeze(image, axis=0) # Remove batch dimension
                
                initial_M = q_helper.getM(classifier, f_sample)

                # Start updating the Q-table
                for i in range(q_classifier.num_iterations):

                    # Select a random action
                    action = np.random.randint(0, q_classifier.num_actions)

                    # Apply the action to the image
                    perm_image = q_helper.apply_action(image, action)

                    perm_image = tf.expand_dims(perm_image, axis=0) # Add batch dimension
                    f_prime_sample = CNN.extract_features(perm_image)
                    perm_image = tf.squeeze(perm_image, axis=0)

                    # Compute the new metric M
                    new_M = q_helper.getM(classifier, f_prime_sample)

                    # Compute the reward and next state
                    reward = q_helper.getReward(initial_M, new_M)
                    next_state = 0 if new_M <= initial_M else 1

                    # Update the Q-table
                    old_q_table = np.copy(q_table)
                    q_table[current_state, action] += q_classifier.learning_rate * (
                        reward + q_classifier.discount_rate * np.max(q_table[next_state]) - q_table[current_state, action]
                    )

                    # Check for convergence
                    max_q_change = np.max(np.abs(q_table - old_q_table))
                    if max_q_change < 0.0001:
                        if converge_index > max_converge_index: 
                            max_converge_index = converge_index
                        break

                    # Update the current state
                    current_state = next_state
                    converge_index +=1
                # After all iterations, save the Q-table to a file
                q_table_filepath = os.path.join(output_dir, f"q_table_{global_idx}.npy")
                np.save(q_table_filepath, q_table)
    print(f"Maximum number of iterations needed for convergence: {max_converge_index}")

    # Save the output directory to the QClassifier object
    q_classifier.q_table_dir = output_dir
    

