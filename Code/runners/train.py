import os
import numpy as np
from functions import q_helper
import tensorflow as tf

def train_CNN(tall_resnet, train_dataset, val_dataset, output_model_path, resnet_weights_path, epochs=10, batch_size=23, augment=True):
    """
    Fine tunes pre-trained ResNet-50 model on the dataset.
    """
    tall_resnet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    tall_resnet.train(train_dataset, val_dataset, epochs=epochs, batchsize=batch_size)

    #tall_resnet.save(output_model_path)
    tall_resnet.save_dense_weights(resnet_weights_path)
    #print(f"Model saved at {output_model_path}")
    print(f"ResNet weights saved at {resnet_weights_path}")

    # os.makedirs('checkpoints', exist_ok=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras',  # Save location and filename pattern
    #     save_best_only=True,  # Only save if the model improves on validation loss
    #     monitor='val_loss',  # Metric to monitor
    #     mode='min',          # Minimize the validation loss
    #     verbose=1            # Print a message when saving
    # )

    #return history


def train_secondary_classifier(
    secondary_NN, train_feature_set, val_feature_set, output_model_path, epochs=10, batch_size=32, augment=False
):
    secondary_NN.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    # train_labels = tf.keras.utilsto_categorical(train_labels, num_classes=5)
    # val_labels = tf.keras.utilsto_categorical(val_labels, num_classes=5)

    secondary_NN.train(
        # train_features,
        # train_labels,
        # val_features,
        # val_labels,
        train_feature_set,
        val_feature_set,
        epochs=epochs, 
        batchsize=batch_size,
        callbacks = []
    )
    
    secondary_NN.save(output_model_path)
    print(f"Model saved at {output_model_path}")

    # os.makedirs('checkpoints', exist_ok=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras',  # Save location and filename pattern
    #     save_best_only=True,  # Only save if the model improves on validation loss
    #     monitor='val_loss',  # Metric to monitor
    #     mode='min',          # Minimize the validation loss
    #     verbose=1            # Print a message when saving
    # )

    #return history

def train_q_tables(q_classifier, resnet, secondary_classifier, dataset, marked_hard_images, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (image_batch, _) in enumerate(dataset):
        for batch_idx, image in enumerate(image_batch):
            global_idx = idx * len(image_batch) + batch_idx   # Iterate over images in the dataset
            if not marked_hard_images[global_idx]: # Skip easy images
                q_table = np.zeros((2, q_classifier.num_actions))  # Initialize Q-table
                current_state = 0

                # Compute the initial metric M
                f_sample = resnet.extract_features(image, 'conv5_block3_out')
                f_sample = tf.keras.layers.Dense(1024, activation='relu')(f_sample)
                #f_sample= tf.squeeze(f_sample, axis=0)
                
                initial_M = q_helper.getM(secondary_classifier, f_sample)
                #print(f"Initial M: {initial_M}")

                for _ in range(q_classifier.num_iterations):
                # Select a random action
                    action = np.random.randint(0, q_classifier.num_actions)
                    perm_image = q_helper.apply_action(image, action)

                    f_prime_sample = resnet.extract_features(perm_image, 'conv5_block3_out')
                    f_prime_sample = tf.keras.layers.Dense(1024, activation='relu')(f_prime_sample)
                    f_prime_sample = tf.squeeze(f_prime_sample, axis=0)

                    # original_predictions = secondary_classifier.predict(f_sample)
                    # transformed_predictions = secondary_classifier.predict(f_prime_sample)
                    # print(f"Original Predictions: {original_predictions}")
                    # print(f"Transformed Predictions: {transformed_predictions}")
                    # initial_M = np.std(original_predictions)
                    # new_M = np.std(transformed_predictions) 
                    # print(f"Initial M: {initial_M}")
                    # print(f"New M: {new_M}")
                    new_M = q_helper.getM(secondary_classifier, f_prime_sample)
                    print(new_M)

                    reward = q_helper.getReward(initial_M, new_M)
                    next_state = 0 if new_M <= initial_M else 1
                    q_table[current_state, action] += q_classifier.learning_rate * (
                        reward + q_classifier.discount_rate * np.max(q_table[next_state]) - q_table[current_state, action]
                    )
                    current_state = next_state
                # Save the Q-table to a file
                q_table_filepath = os.path.join(output_dir, f"q_table_{global_idx}.npy")
                np.save(q_table_filepath, q_table)
                print(f"Trained Q-table saved for image {global_idx} at {q_table_filepath}")
    
    q_classifier.q_table_dir = output_dir
    

