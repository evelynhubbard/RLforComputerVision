import numpy as np
from functions import q_helper, visualization
import os
import tensorflow as tf

class QClassifier:
    def __init__(self, q_table_dir, num_actions, learning_rate, discount_rate, iteration_constant):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_iterations = self.num_actions * iteration_constant
        self.q_table_dir = q_table_dir #2 states

    def load_q_table(self, image_idx):
        # Load Q-table from file

        q_table_filepath = os.path.join(self.q_table_dir, f"q_table_{image_idx}.npy")
        if not os.path.exists(q_table_filepath):
            raise FileNotFoundError(f"Q-table for image {image_idx} not found at {q_table_filepath}")
        return np.load(q_table_filepath)
    
    def get_learned_feature_map(self, image, image_idx, CNN, show_diff, result_path):
        # Extract the feature map of the learned best image
        
            # Load the pre-trained Q-table for this image
            q_table = self.load_q_table(image_idx)
            x_0 = 1  # Start with the initial state

            # Choose the best action based on the Q-table
            best_action = np.argmax(q_table[x_0])

            # Apply the best action to the image
            best_image = q_helper.apply_action(image, best_action)
            
            if show_diff and best_action > 0:
                visualization.show_image_difference(image, best_image, image_idx, best_action, result_path)
                
            # Extract and return the feature map of the best-transformed image
            feature_map = CNN.extract_features(tf.expand_dims(best_image, axis=0))
            
            return feature_map
    
    def get_learned_image(self, image, image_idx):
        # Extract the feature map of the learned best image
        
            # Load the pre-trained Q-table for this image
            q_table = self.load_q_table(image_idx)
            x_0 = 1  # Start with the initial state

            # Choose the best action based on the Q-table
            best_action = np.argmax(max(q_table[x_0]))

            # Apply the best action to the image
            best_image = q_helper.apply_action(image, best_action)
        
            return best_image