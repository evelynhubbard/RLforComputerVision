import tensorflow as tf

def extract_feature_maps(resnet, dataset):
    """
    Extract feature maps from a pre-trained model and apply dimensionality reduction.
    """
    def map_fn(images, labels):
        # Extract feature maps for the batch
        feature_maps = resnet.extract_features(images)
        
        # Return feature maps and labels    
        return feature_maps, labels  

    # Apply the mapping function to transform the dataset
    feature_map_dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return feature_map_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
