import matplotlib.pyplot as plt

def plot_training_curves(history):
    """
    Plot training and validation accuracy and loss curves.
    
    Args:
        history (tf.keras.callbacks.History): Training history object.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

def plot_feature_maps(feature_maps):
    """
    Visualize feature maps extracted from a CNN.
    
    Args:
        feature_maps (np.ndarray): Array of feature maps.
    """
    num_features = feature_maps.shape[-1]
    size = feature_maps.shape[1]

    # Plot first set of feature maps
    plt.figure(figsize=(15, 15))
    for i in range(num_features):
        plt.subplot(8, 8, i + 1)  # Adjust grid size for your feature map
        plt.imshow(feature_maps[0, :, :, i], cmap="viridis")
        plt.axis("off")
    plt.show()
