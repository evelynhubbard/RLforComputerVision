import tensorflow as tf

def custom_loss_function(y_true, y_pred):
    """
    Custom loss function (example: Mean Squared Error with a scaling factor).
    
    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted labels.
    
    Returns:
        tf.Tensor: Computed loss.
    """
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred) * 0.5  # Example scaling factor
