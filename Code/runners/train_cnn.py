from models import ResNetModel
import tensorflow as tf


def train_CNN(tall_resnet, train_dataset, val_dataset, output_model_path, epochs=10, batch_size=23, augment=True):
    """
    Fine tunes pre-trained ResNet-50 model on the dataset.
    """
    tall_resnet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    tall_resnet.train(train_dataset, val_dataset, epochs=epochs, batchsize=batch_size)

    tall_resnet.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[],
    )

    tall_resnet.save(output_model_path)
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


