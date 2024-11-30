import numpy as np
import tensorflow as tf

def extract_feature_maps(model_name, resnet, layer_name, dataset):
    """
    Extract feature maps from a pre-trained model.
    """
    def map_fn(images, labels):
        
        feature_maps = resnet.extract_features(images, layer_name)  # Feature maps for the batch
        
        return feature_maps, labels  # Return feature maps and labels

    # Apply the mapping function to transform the dataset
    feature_map_dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    for feature_maps, labels in feature_map_dataset.take(5):
        print("Transformed Labels:", labels.numpy())


    return feature_map_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)