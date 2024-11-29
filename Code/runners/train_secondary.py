from models import SecondaryClassifier_NN, SecondaryClassifier_SVM
import tensorflow as tf

from functions.feature_extraction import extract_feature_maps

def train_secondary_classifier(
    train_features, train_labels, val_features, val_labels, output_model_path,epochs=10, batch_size=32, augment=True
):
    """
    Train a secondary classifier on feature maps extracted from the primary CNN.

    Args:
        primary_model_path (str): Path to the pre-trained primary model.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        output_model_path (str): Path to save the secondary classifier.

    Returns:
        float: Validation accuracy of the secondary classifier.
    """
    # Load the pre-trained primary model
    secondary_NN = SecondaryClassifier_NN()
    secondary_NN.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = secondary_NN.train(
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=epochs, 
        batchsize=batch_size,
        callbacks = []
    )
    secondary_NN.save(output_model_path)
    print(f"Model saved at {output_model_path}")

    # os.makedirs('checkpoints', exist_ok=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras',  # Save location and filename pattern
    #     save_best_only=True,  # Only save if the model improves on validation loss
    #     monitor='val_loss',  # Metric to monitor
    #     mode='min',          # Minimize the validation loss
    #     verbose=1            # Print a message when saving
    # )

    #return history

    
