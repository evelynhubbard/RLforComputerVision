import tensorflow as tf
import numpy as np
from models.resnet import get_tall_resnet_CNN

#from tensorflow.keras.preprocessing import image

# # Load an image
# img_path = "Code/data/raw/cat.jpg"  # Replace with your image path
# img = image.load_img(img_path, target_size=(224, 224))

# # Convert to array and preprocess
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Preprocess for ResNet-50

# print("Image array shape:", img_array.shape)

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


tall_resnet = get_tall_resnet_CNN()

tall_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = tall_resnet.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  
)
tall_resnet.save('tall_resnet_trained_model.h5')


# feature_map = tall_resnet.predict(img_array)
# print("Feature map shape:", feature_map.shape)
#basic_resnet = tf.keras.applications.ResNet50(weights="imagenet")

#tall_resnet.summary()

# feature_map = tall_resnet.predict(img_array)

# # Decode predictions
# decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=3)  # Top 3 predictions
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
#     print(f"{i+1}: {label} ({score:.2f})")
