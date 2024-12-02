import numpy as np
import tensorflow as tf

# def extract_feature_maps(model_name, resnet, layer_name, dataset):
#     """
#     Extract feature maps from a pre-trained model.
#     """
#     def map_fn(images, labels):
        
#         feature_maps = resnet.extract_features(images, layer_name)  # Feature maps for the batch
        
#         feature_maps = tf.keras.layers.GlobalAveragePooling2D()(feature_maps)  # Global average pooling 
#         feature_maps = tf.keras.layers.Dense(1024, activation='relu')(feature_maps)

#         return feature_maps, labels  # Return feature maps and labels

#     # Apply the mapping function to transform the dataset
#     feature_map_dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

#     return feature_map_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def extract_feature_maps(model_name, resnet, layer_name, dataset):
    """
    Extract feature maps from a pre-trained model and apply dimensionality reduction.
    """
    # Create reusable layers outside the map function
    # global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    # dense_layer = tf.keras.layers.Dense(1024, activation='relu')
    dense_layer = tf.keras.layers.Dense(1024, activation='relu')

    def map_fn(images, labels):
        # Extract feature maps for the batch
        feature_maps = resnet.extract_features(images, layer_name)
        
        # Apply global average pooling
        # feature_maps = global_avg_pool(feature_maps)
        
        # # Apply dimensionality reduction
        feature_maps = dense_layer(feature_maps)
        

        return feature_maps, labels  # Return feature maps and labels

    # Apply the mapping function to transform the dataset
    feature_map_dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return feature_map_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
