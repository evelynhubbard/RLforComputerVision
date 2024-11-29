import numpy as np
import tensorflow as tf

def extract_feature_maps(model_name, resnet, layer_name, dataset):
    """
    Extract feature maps from a pre-trained model.
    """
    feature_extractor = tf.keras.Model(
        inputs = resnet.model.input,
        outputs = resnet.model.get_layer(layer_name).output)

    feature_maps = []
    labels = []
    for images, lbls in dataset:
        feature_maps.append(feature_extractor.predict(images))
        labels.append(lbls.numpy())
    return np.vstack(feature_maps), np.hstack(labels)
