import matplotlib.pyplot as plt
import os
import numpy as np

def save_training_curves(history, results_path, title):
    """
    Plots and saves training and validation loss and accuracy curves.
    """
    # Create the results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)

    # Extract metrics from the history object
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    accuracy = history.history.get("accuracy", [])
    val_accuracy = history.history.get("val_accuracy", [])
    epochs = range(1, len(loss) + 1)

    # Create the plot
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss, label="Train Loss", color = "blue", linewidth = 2)
    plt.plot(val_loss, label="Validation Loss", color = "red", linestyle="--", linewidth = 2)
    plt.title("Loss Curve", fontsize = 14, fontweight = "bold")
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Loss",fontsize = 12)
    plt.ylim(0, max(max(loss), max(val_loss)))
    plt.xticks(ticks = epochs, fontsize = 10)
    plt.tick_params(axis = "x", which = "major", labelsize = 10)
    plt.yticks(fontsize = 10)
    plt.legend(fontsize = 10, loc = "upper right")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label="Train Loss", color = "blue", linewidth = 2)
    plt.plot(val_accuracy, label="Validation Loss", color = "red", linestyle="--", linewidth = 2)
    plt.title("Accuracy Curve", fontsize = 14, fontweight = "bold")
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Loss",fontsize = 12)
    plt.ylim(0.5, 1) 
    plt.xticks(ticks = epochs, fontsize = 10)
    plt.tick_params(axis = "x", which = "major", labelsize = 10)
    plt.yticks(fontsize = 10)
    plt.legend(fontsize = 10, loc = "upper right")

    # Save the plot to file
    save_path = os.path.join(results_path, title.replace(" ", "_").lower() + ".png")
    plt.savefig(save_path)
    plt.close()  # Free memory

def save_feature_maps(feature_maps, results_path, title):
    """
    Visualize feature maps extracted from a CNN.
    """

    # Create the results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)

    for feature_map_batch, _ in feature_maps.take(1):  # Only take the first batch
        feature_map_batch = feature_map_batch.numpy()
        break

    feature_map_batch = feature_map_batch[0]

    feature_map_batch = (feature_map_batch - feature_map_batch.min()) / (
        feature_map_batch.max() - feature_map_batch.min()
    )

    num_features = min(feature_map_batch.shape[-1], 64)
    size = feature_map_batch.shape[0]

    grid_cols = int(np.ceil(np.sqrt(num_features)))
    grid_rows = int(np.ceil(num_features / grid_cols))
    
    # Plot first batch of feature maps
    plt.figure(figsize=(15, 15))
    plt.suptitle(f"{title}: Top {num_features} Feature Maps", fontsize=16, fontweight="bold")
    #plt.suptitle(title, fontsize=16, fontweight="bold")
    for i in range(num_features):
        plt.subplot(grid_rows, grid_cols, i + 1)  # Adjust grid size for your feature map
        plt.imshow(feature_map_batch[:, :, i], cmap="viridis")
        plt.axis("off")
    
    # Save the plot to file
    save_path = os.path.join(results_path, title.replace(" ", "_").lower() + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()  # Free memory

def show_image_difference(original_image, transformed_image, image_idx, action, picture_path):
    """
    Display the original and transformed images side by side.
    """
    # Reverse the normalization
    imagenet_mean = np.array([123.68, 116.779, 103.939])  # RGB mean values for ImageNet
    original_image_for_display = np.clip(original_image+imagenet_mean, 0, 255).astype('uint8')
    transformed_image_for_display = np.clip(transformed_image+imagenet_mean, 0, 255).astype('uint8')

    os.makedirs(picture_path, exist_ok=True)

    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_for_display, cmap='gray' if len(original_image_for_display.shape) == 2 else None)
    plt.title('Original Image')
    plt.axis('off')

    # Processed image
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image_for_display, cmap='gray' if len(transformed_image_for_display.shape) == 2 else None)
    plt.title('Processed Image')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot to file
    save_path = os.path.join(picture_path, str(image_idx) + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Free memory
    print(f"Image {image_idx} with action {action} saved to {save_path}")