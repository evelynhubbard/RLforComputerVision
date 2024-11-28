import tensorflow as tf
import numpy as np
from models.resnet import get_tall_resnet_CNN
from experiments.train import train_CNN

#from tensorflow.keras.preprocessing import image

# # Load an image
# img_path = "Code/data/raw/cat.jpg"  # Replace with your image path
# img = image.load_img(img_path, target_size=(224, 224))

# # Convert to array and preprocess
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Preprocess for ResNet-50

# print("Image array shape:", img_array.shape)
if __name__ == "__main__":
    tall_resnet_model = get_tall_resnet_CNN()  # Call model creation function
    history = train_CNN(tall_resnet_model)
    tall_resnet_model.save('tall_resnet_trained_model.h5')


# feature_map = tall_resnet.predict(img_array)
# print("Feature map shape:", feature_map.shape)
#basic_resnet = tf.keras.applications.ResNet50(weights="imagenet")

#tall_resnet.summary()

# feature_map = tall_resnet.predict(img_array)

# # Decode predictions
# decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=3)  # Top 3 predictions
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
#     print(f"{i+1}: {label} ({score:.2f})")
