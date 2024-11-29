import tensorflow as tf

def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy metric.
    
    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted labels.
    
    Returns:
        tf.Tensor: Accuracy value.
    """
    y_pred_classes = tf.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(y_true == y_pred_classes, tf.float32))
    return accuracy

def compute_precision(y_true, y_pred):
    """
    Compute precision metric.
    """
    y_pred_classes = tf.argmax(y_pred, axis=1)
    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred_classes)
    return precision.result().numpy()

def compute_recall(y_true, y_pred):
    """
    Compute recall metric.
    """
    y_pred_classes = tf.argmax(y_pred, axis=1)
    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred_classes)
    return recall.result().numpy()
