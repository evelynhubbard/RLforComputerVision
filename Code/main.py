from runners import *
import os
import numpy as np
from functions import *
from datasets.imageNet import *
from utils import add_date_time_to_path
from models import *

#FOR 4 CLASSES
#SELECTED_CLASSES = ["n01443537", "n01774384", "n02002724", "n07873807", "n04398044"]
# Paths
# RAW_TRAIN_PATH = "Code/data/raw/tiny-imagenet-200/train"
# RAW_VAL_PATH = "Code/data/raw/tiny-imagenet-200/val"
# STRUCTURED_TRAIN_PATH = "Code/data/processed/tiny-imagenet-200/structured_train"
# STRUCTURED_VAL_PATH = "Code/data/processed/tiny-imagenet-200/structured_val"
#VAL_ANNOTATIONS_FILE = "Code/data/raw/tiny-imagenet-200/val/val_annotations.txt"
# PRIMARY_MODEL_PATH = "Code/models/resnet_model/16batch5Class.h5"
# SECONDARY_MODEL_PATH = "Code/models/secondary_classifier_NN/16batch5Class.h5"


RAW_PATH = "Code/data/raw/ImageNet"
STRUCTURED_TRAIN_PATH = "Code/data/processed/ImageNet/structured_train"
STRUCTURED_VAL_PATH = "Code/data/processed/ImageNet/structured_val"
STRUCTURED_TEST_PATH = "Code/data/processed/ImageNet/structured_test"
LABEL_MAPPING_PATH = "Code/data/processed/ImageNet/labelmappins.txt"

NEW_PRIMARY_MODEL_PATH = add_date_time_to_path("Code/models/resnet_model/ImageNet",".weights.h5")
CURRENT_PRIMARY_MODEL_PATH = "Code/models/resnet_model/ImageNet/2024-11-29_23-18-07.weights.h5"

NEW_SECONDARY_MODEL_PATH = add_date_time_to_path("Code/models/secondary_classifier_NN/ImageNet",".weights.h5")
CURRENT_SECONDARY_MODEL_PATH = "Code/models/secondary_classifier_NN/ImageNet/2024-11-29_21-49-42.weights.h5"

NUM_CLASSES = 4
EPOCHS = 10
BATCH_SIZE = 32
AUGMENT = False
EXTRACT_LAYER_NAME = "conv5_block3_out"

def main(retrain_cnn, retrain_classifier):
    
    if retrain_cnn or retrain_classifier:
        print("Restructure the dataset...")
        # restructure_tiny_dataset(
        #     selected_classes=SELECTED_CLASSES,
        #     train_data_path=RAW_TRAIN_PATH, 
        #     structured_train_path=STRUCTURED_TRAIN_PATH,
        #     val_data_path=RAW_VAL_PATH,
        #     structured_val_path=STRUCTURED_VAL_PATH,
        #     val_annotations_file=VAL_ANNOTATIONS_FILE,
        #)
        restructure_imageNet(
            data_path=RAW_PATH,
            structured_train_path=STRUCTURED_TRAIN_PATH,
            structured_val_path=STRUCTURED_VAL_PATH,
            structured_test_path=STRUCTURED_TEST_PATH,
            label_mapping_path = LABEL_MAPPING_PATH
        )
        print("Summarizing the dataset...")
        num_train_classes, num_train_images = summarize_dataset(STRUCTURED_TRAIN_PATH)
        num_val_classes, num_val_images = summarize_dataset(STRUCTURED_VAL_PATH)
        num_test_classes, num_test_images = summarize_dataset(STRUCTURED_TEST_PATH)

        print(f"Structured Training Dataset: {num_train_classes} classes, {num_train_images} images.")
        print(f"Structured Validation Dataset: {num_val_classes} classes, {num_val_images} images.")
        print(f"Structured Test Dataset: {num_test_classes} classes, {num_test_images} images.")

        print("Loading the train and val dataset...")
        # train_dataset, val_dataset = load_tiny_imagenet(
        #     train_data_path=STRUCTURED_TRAIN_PATH,
        #     val_data_path=STRUCTURED_VAL_PATH,
        #     batch_size = BATCH_SIZE, 
        #     augment=AUGMENT)
        batched_train_dataset = load_imageNet(
            data_path=STRUCTURED_TRAIN_PATH,
            image_size=(224,224),
            batch_size=BATCH_SIZE,
            augment=AUGMENT
        )
        batched_val_dataset = load_imageNet(
            data_path=STRUCTURED_VAL_PATH,
            image_size=(224,224),
            batch_size=BATCH_SIZE,
            augment=AUGMENT
        )
    
    resnet_model = ResNetModel(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, trainable=True)

    if retrain_cnn:
        print("Training the primary CNN...")
        train_CNN(
            tall_resnet = resnet_model,
            train_dataset=batched_train_dataset,
            val_dataset=batched_val_dataset,
            output_model_path=NEW_PRIMARY_MODEL_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            augment=AUGMENT
        )
        print(f"Primary model saved to {NEW_PRIMARY_MODEL_PATH}")
    else:
        resnet_model.load(CURRENT_PRIMARY_MODEL_PATH)

    classifier_model = SecondaryClassifier_NN(input_shape = (7,7,2048), num_classes=NUM_CLASSES, trainable = True)

    if retrain_classifier:
        train_features_set = extract_feature_maps("RESNET", resnet_model, EXTRACT_LAYER_NAME, batched_train_dataset)
        val_features_set = extract_feature_maps("RESNET", resnet_model, EXTRACT_LAYER_NAME, batched_val_dataset)

        #print(f"Train features shape: {train_features_set.shape}")
        #print(f"Train labels shape: {train_labels.shape}")

        print("Training the secondary classifier...")
        train_secondary_classifier(
            classifier_model,
            train_features_set,
            val_features_set,
            # train_features,
            # train_labels,
            # val_features,
            # val_labels,
            output_model_path=NEW_SECONDARY_MODEL_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            augment=AUGMENT
        )

        print(f"Secondary classifier trained and saved to Code/models/secondary_classifier.pkl")
    else:
        classifier_model.load(CURRENT_SECONDARY_MODEL_PATH)
    
    # q_classifier = QClassifier(num_actions = 3, learning_rate = 0.3, discount_rate = 0.4)
    # if retrain_q-learning:
    #     pass
    # else:
    #     q_classifier.load(Q_LEARNING_MODEL)

    # ##BEGIN TESTING PHASE
    test_dataset = load_imageNet(
        data_path=STRUCTURED_TEST_PATH,
        image_size=(224,224),
        batch_size=BATCH_SIZE,
        augment=False
    )
    
    basic_accuracy = basic_classify_test(resnet_model.model, test_dataset)

    resnet_model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    accuracy = resnet_model.model.evaluate(test_dataset)

    print(f"Primary Model Accuracy: {accuracy}")
    print(f"Basic accuracy: {basic_accuracy}")

    marked_hard_vector = get_marked_hard_images(resnet_model, test_dataset)
    

    # Q_accuracy = RL_classify_test(resnet_model, classifier_model, q_classifier, marked_hard_dataset, test_dataset)
    #for image in hard_to_classify set:
    
        
        #RL based classification:
        #random actions selected
        #action applied
        #new feature map extracted
        # current state is decided after observing standard deviation of predictions scores
        #reward is given based on the change in prediction score
        #number of iterations for updating Q-table is axm = 20
        #alpha = 0.4
        #discount rate = 0.3
        #after 20 iterations, the action with the highest Q-value is selected
        #this action is applied to the image
        #the new feature map is extracted
        #the new feature map is classified
        
    #extract feature map
    

    #RL based classification:
        #random actions selected
        #action applied
        #new feature map extracted
        # current state is decided after observing standard deviation of predictions scores
        #reward is given based on the change in prediction score
        #number of iterations for updating Q-table is axm = 20
        #alpha = 0.4
        #discount rate = 0.3
        #after 20 iterations, the action with the highest Q-value is selected
        #this action is applied to the image
        #the new feature map is extracted
        #the new feature map is classified
    # Step 4: Evaluate the primary CNN
    # print("Evaluating the primary CNN...")
    # evaluate_model(
    #     model_path="Code/models/resnet_model.h5",
    #     dataset=single_val_dataset,
    #     is_primary=True,
    # )

    # print("Evaluating the secondary classifier...")
    # evaluate_model(
    #     model_path="Code/models/secondary_classifier.pkl",
    #     dataset=val_dataset,
    #     is_primary=False,
    # )

if __name__ == "__main__":
    print("Do you want to retrain or load the CNN? (t/l)")
    cnn_choice = input().strip().lower()
    
    if cnn_choice == "t":
        # Display details for retraining
        dataset = os.path.basename(RAW_PATH)
        print(f"Training on dataset:  '{dataset}'")
        print(f"Epochs: '{EPOCHS}'") 
        retrain_cnn = True
    elif cnn_choice == "l":
        print(f"CNN loaded from {CURRENT_PRIMARY_MODEL_PATH}")
        retrain_cnn = False
    else:
        print("Invalid choice. Please restart and choose 't' for retrain or 'l' for load.")
        exit(1)

    print("Do you want to retrain or load the classifying structure? (t/l)")
    classifier_choice = input().strip().lower()

    if classifier_choice == "t":
        print(f"Retraining NN on feature maps from CNN layer '{EXTRACT_LAYER_NAME}'")
        retrain_classifier = True
    elif classifier_choice == "l":
        print(f"Classifier loaded from {CURRENT_SECONDARY_MODEL_PATH}")
        retrain_classifier = False
    else:
        print("Invalid choice. Please restart and choose 't' for retrain or 'l' for load.")
        exit(1)
    # Call main with appropriate variables
    main(
        retrain_cnn=retrain_cnn,
        retrain_classifier=retrain_classifier,
    )