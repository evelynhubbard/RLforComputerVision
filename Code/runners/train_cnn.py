from models import ResNetModel
from datasets import load_tiny_imagenet
import tensorflow as tf


def train_CNN(train_dataset, val_dataset, output_model_path, epochs=10, batch_size=23, augment=True):
    """
    Trains a ResNet-50 model on the Tiny ImageNet dataset.
    """
    # Load the Tiny ImageNet dataset
    
    tall_resnet = ResNetModel(input_shape=(224, 224, 3), num_classes=200, trainable=False)
    tall_resnet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    # history = tall_resnet.train(
    #     train_dataset,val_dataset,epochs=epochs, 
    #     callbacks = []
    # )
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


