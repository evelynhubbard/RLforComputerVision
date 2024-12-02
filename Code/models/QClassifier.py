import numpy as np
from functions import q_helper
import os
import tensorflow as tf

class QClassifier:
    #initiate 2-state Q-learning class
    def __init__(self, q_table_dir, num_actions = 3, learning_rate = 0.3, discount_rate = 0.4):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_iterations = self.num_actions * 20
        self.q_table_dir = q_table_dir #2 states

    def load_q_table(self, image_idx):
        #load Q-table from file
        q_table_filepath = os.path.join(self.q_table_dir, f"q_table_{image_idx}.npy")
        if not os.path.exists(q_table_filepath):
            raise FileNotFoundError(f"Q-table for image {image_idx} not found at {q_table_filepath}")
        return np.load(q_table_filepath)

    
    def classify_image(self, image, image_idx, resnet, secondary_classifier):
        #image is batch...
        # for image in images:
            # current_state = 0

            # f_sample = resnet.extract_features(image, 'conv5_block3_out')
            # initial_M = q_helper.getM(secondary_classifier, f_sample)

            # for i in range(self.num_iterations):
            #     action_i = np.random.randint(0,self.num_actions)
            #     perm_image= q_helper.apply_action(image, action_i)
            #     f_prime_sample = resnet.extract_features(perm_image, 'conv5_block3_out')
            #     new_M = q_helper.getM(secondary_classifier, f_prime_sample)
            #     reward = q_helper.getReward(initial_M, new_M)
            #     next_state = 0 if new_M <= initial_M else 1
            #     self.update_q_table(current_state, action_i, reward, next_state)
            #     current_state = next_state
            # Load the pre-trained Q-table for this image
            q_table = self.load_q_table(image_idx)
            current_state = 0  # Start with the initial state

            # Choose the best action based on the Q-table
            best_action = np.argmax(np.max(q_table))
            print(f"Best action for image {image_idx}: {best_action}")

            # Apply the best action to the image
            best_image = q_helper.apply_action(image, best_action)

            # Extract and return the feature map of the best-transformed image
            feature_map = resnet.extract_features(best_image, 'conv5_block3_out')
            feature_map = tf.keras.layers.Dense(1024, activation='relu')(feature_map)
            return feature_map