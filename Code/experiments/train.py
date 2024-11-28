import tensorflow as tf
import os


def train_CNN(tall_resnet):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        "Code/data/processed/tiny-imagenet-200/structured_train",
        image_size=(224, 224),  # ResNet50 input size
        batch_size=32
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        "Code/data/processed/tiny-imagenet-200/structured_val",
        image_size=(224, 224),
        batch_size=32
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)


    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    os.makedirs('checkpoints', exist_ok=True)

    tall_resnet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras',  # Save location and filename pattern
        save_best_only=True,  # Only save if the model improves on validation loss
        monitor='val_loss',  # Metric to monitor
        mode='min',          # Minimize the validation loss
        verbose=1            # Print a message when saving
    )

    history = tall_resnet.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10, 
        callbacks = [checkpoint]
    )

    return history


