import argparse
from runners.train_cnn import train_CNN
from functions import *
from runners.train_secondary import train_secondary_classifier
from runners.evaluate import evaluate_model
from datasets.tiny_imagenet import *
import tensorflow as tf
from models.resnet import ResNetModel
from models.secondary_classifier_NN import SecondaryClassifier_NN
#FOR 4 CLASSES
SELECTED_CLASSES = ["n01443537", "n01629819", "n01641577", "n01644900"]
# Paths
RAW_TRAIN_PATH = "Code/data/raw/tiny-imagenet-200/train"
RAW_VAL_PATH = "Code/data/raw/tiny-imagenet-200/val"
STRUCTURED_TRAIN_PATH = "Code/data/processed/tiny-imagenet-200/structured_train"
STRUCTURED_VAL_PATH = "Code/data/processed/tiny-imagenet-200/structured_val"
VAL_ANNOTATIONS_FILE = "Code/data/raw/tiny-imagenet-200/val/val_annotations.txt"
PRIMARY_MODEL_PATH = "Code/models/resnet_model.h5"
SECONDARY_MODEL_PATH = "Code/models/secondary_classifier_NN.h5"

EPOCHS = 10
BATCH_SIZE = 32
AUGMENT = True
EXTRACT_LAYER_NAME = "conv5_block3_out"

def main(retrain = False):
    """
    Main function to execute the training and evaluation workflow.
    """
    print("Restructure the dataset...")
    restructure_tiny_dataset(
        selected_classes=SELECTED_CLASSES,
        train_data_path=RAW_TRAIN_PATH, 
        structured_train_path=STRUCTURED_TRAIN_PATH,
        val_data_path=RAW_VAL_PATH,
        structured_val_path=STRUCTURED_VAL_PATH,
        val_annotations_file=VAL_ANNOTATIONS_FILE,
    )
    print("Summarizing the dataset...")
    num_train_classes, num_train_images = summarize_dataset(STRUCTURED_TRAIN_PATH)
    num_val_classes, num_val_images = summarize_dataset(STRUCTURED_VAL_PATH)
    print(f"Structured Training Dataset: {num_train_classes} classes, {num_train_images} images.")
    print(f"Structured Validation Dataset: {num_val_classes} classes, {num_val_images} images.")

    print("Loading the dataset...")
    train_dataset, val_dataset = load_tiny_imagenet(
        train_data_path=STRUCTURED_TRAIN_PATH,
        val_data_path=STRUCTURED_VAL_PATH,
        batch_size = BATCH_SIZE, 
        augment=AUGMENT)
    
    # print("Training the primary CNN...")
    # train_CNN(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     output_model_path=PRIMARY_MODEL_PATH,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     augment=AUGMENT
    # )
    # print(f"Primary model saved to Code/models/resnet_model.h5")

    resnet_model = ResNetModel()
    resnet_model.load(PRIMARY_MODEL_PATH)
    train_features, train_labels = extract_feature_maps("RESNET", resnet_model, EXTRACT_LAYER_NAME, train_dataset)
    val_features, val_labels = extract_feature_maps("RESNET", resnet_model, EXTRACT_LAYER_NAME, val_dataset)

    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    # print("Training the secondary classifier...")
    # accuracy = train_secondary_classifier(
    #     train_features,
    #     train_labels,
    #     val_features,
    #     val_labels,
    #     output_model_path=SECONDARY_MODEL_PATH,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     augment=AUGMENT
    # # )
    # print(f"Secondary classifier trained and saved to Code/models/secondary_classifier.pkl")
    # print(f"Validation Accuracy of Secondary Classifier: {accuracy:.2f}")
    classifier_model = SecondaryClassifier_NN()
    classifier_model.load(SECONDARY_MODEL_PATH)

    # Step 4: Evaluate the primary CNN
    print("Evaluating the primary CNN...")
    evaluate_model(
        model_path="Code/models/resnet_model.h5",
        dataset=val_dataset,
        is_primary=True,
    )

    # print("Evaluating the secondary classifier...")
    # evaluate_model(
    #     model_path="Code/models/secondary_classifier.pkl",
    #     dataset=val_dataset,
    #     is_primary=False,
    # )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load the primary CNN model.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the model instead of using the saved model.",
    )
    args = parser.parse_args()
    main(retrain=args.retrain)


# # print("Image array shape:", img_array.shape)
# if __name__ == "__main__":
    
#     tall_resnet_model = get_tall_resnet_CNN()  # Call model creation function
#     history = train_CNN(tall_resnet_model, train_dataset, val_dataset)
#     tall_resnet_model.save('tall_resnet_trained_model.h5')

#     feature_extractor = tf.keras.Model(
#             inputs=tall_resnet_model.input,
#             outputs=tall_resnet_model.get_layer('global_average_pooling2d').output  # Change if needed
#     )
#     train_features, train_labels = extract_feature_maps(train_dataset, feature_extractor)
#     val_features, val_labels = extract_feature_maps(val_dataset, feature_extractor)


# # Initialize and train ResNet-based model
# resnet_model = ResNetModel(input_shape=(224, 224, 3), num_classes=4, trainable=False)
# resnet_model.compile(optimizer="adam")
# resnet_model.train(train_dataset, val_dataset, epochs=10)
# resnet_model.save("resnet_model.h5")

# # Initialize and compile Q-Network
# q_network = QNetwork(num_actions=10)
# q_network.compile(optimizer="adam")
# q_network.save("q_network_model.h5")

# # Initialize and train Secondary Classifier
# features = np.random.rand(100, 512)  # Example feature map
# labels = np.random.randint(0, 4, size=(100,))  # Example labels

# secondary_classifier = SecondaryClassifier(kernel="linear")
# secondary_classifier.train(features, labels)
# accuracy = secondary_classifier.evaluate(features, labels)
# print(f"Secondary Classifier Accuracy: {accuracy:.2f}")


# print(f"Train features shape: {train_features.shape}")
# print(f"Validation features shape: {val_features.shape}")




# feature_map = tall_resnet.predict(img_array)
# print("Feature map shape:", feature_map.shape)
#basic_resnet = tf.keras.applications.ResNet50(weights="imagenet")

#tall_resnet.summary()

# feature_map = tall_resnet.predict(img_array)

# # Decode predictions
# decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=3)  # Top 3 predictions
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
#     print(f"{i+1}: {label} ({score:.2f})")

