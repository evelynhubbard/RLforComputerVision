import os
import yaml
import warnings
import numpy as np
from runners import *
from functions import *
from datasets import *
from models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_pathsparams(config, test_name):
    paths = {}
    # ImageNet Data
    paths['RAW_PATH'] = config['raw_data_path']
    paths['STRUCTURED_TRAIN_PATH'] = config['structured_train_path']
    paths['STRUCTURED_VAL_PATH'] = config['structured_val_path']
    paths['STRUCTURED_TEST_PATH'] = config['structured_test_path']
    paths['LABEL_MAPPING_PATH'] = config['label_mapping_path']

    # Models
    paths['NEW_PRIMARY_MODEL_PATH'] = add_date_time_to_path(config['new_primary_model_path'], ".weights.h5")
    paths['CURRENT_PRIMARY_MODEL_PATH'] = config['current_primary_model_path']
    paths['NEW_SECONDARY_MODEL_PATH'] = add_date_time_to_path(config['new_secondary_model_path'], ".weights.h5")
    paths['CURRENT_SECONDARY_MODEL_PATH'] = config['current_secondary_model_path']
    paths['NEW_Q_LEARNING_MODEL_PATH'] = add_date_time_to_path(config['new_q_learning_model_path'], ".npy")
    paths['CURRENT_Q_LEARNING_MODEL_PATH'] = config['current_q_learning_model_path']

    # Results
    paths['RESULTS_PATH'] = os.path.join(config['results_path'], test_name)
    paths['TRAIN_RESULTS'] = os.path.join(paths['RESULTS_PATH'], "trainplots")
    paths['TRAIN_FEATURES'] = os.path.join(paths['RESULTS_PATH'], "trainfeatures")
    paths['PICTURES'] = os.path.join(paths['RESULTS_PATH'], "pictures")

    parameters = {}
    # Dataset Parameters
    parameters['NUM_CLASSES'] = config['num_classes']
    # parameters['SELECTED_CLASSES'] = config['selected_classes']

    # Deep learning Paramaters
    parameters['EPOCHS'] = config['epochs']
    parameters['IMAGE_SIZE'] = tuple(config['image_size'])
    parameters['EXTRACT_LAYER_NAME'] = config['extract_layer_name']
    parameters['FEATURE_SIZE'] = tuple(config['feature_size'])
    parameters['BATCH_SIZE'] = config['batch_size']
    parameters['AUGMENT'] = config['augment']
    parameters['SHUFFLE'] = config['shuffle']

    # Q-Learning Parameters
    parameters['NUM_ACTIONS'] = config['num_actions']
    parameters['LEARNING_RATE'] = config['learning_rate']
    parameters['DISCOUNT_RATE'] = config['discount_rate']
    parameters['ITERATION_CONSTANT'] = config['iteration_constant']

    return paths, parameters


def main(restructure_data, retrain_cnn, retrain_classifier, retrain_q_learning, test_name):
    config_path = "Code/configs/ImageNet.yml"
    config = load_config(config_path)
    paths, parameters = setup_pathsparams(config, test_name)

    if restructure_data:
        print("Restructure the dataset...")
        restructure_imageNet(
            data_path=paths['RAW_PATH'],
            structured_train_path=paths['STRUCTURED_TRAIN_PATH'],
            structured_val_path=paths['STRUCTURED_VAL_PATH'],
            structured_test_path=paths['STRUCTURED_TEST_PATH'],
            label_mapping_path = paths['LABEL_MAPPING_PATH']
        )
        num_train_classes, num_train_images = summarize_dataset(paths['STRUCTURED_TRAIN_PATH'])
        num_val_classes, num_val_images = summarize_dataset(paths['STRUCTURED_VAL_PATH'])
        num_test_classes, num_test_images = summarize_dataset(paths['STRUCTURED_TEST_PATH'])

        print("Dataset Summary: ImageNet")
        print(f"{'Split':<15}{'Classes':<10}{'Images':<10}")
        print("-" * 40)
        print(f"{'Training':<15}{num_train_classes:<10}{num_train_images:<10}")
        print(f"{'Validation':<15}{num_val_classes:<10}{num_val_images:<10}")
        print(f"{'Testing':<15}{num_test_classes:<10}{num_test_images:<10}")
        print("-" * 40)

    if retrain_cnn or retrain_classifier:
        # If any of the deep learning models are to be retrained, load the datasets
        batched_train_dataset = load_imageNet(
            data_path=paths['STRUCTURED_TRAIN_PATH'],
            image_size = parameters['IMAGE_SIZE'],
            batch_size=parameters['BATCH_SIZE'],
            augment=parameters['AUGMENT'],
            shuffle=parameters['SHUFFLE']
        )
        batched_val_dataset = load_imageNet(
            data_path=paths['STRUCTURED_VAL_PATH'],
            image_size = parameters['IMAGE_SIZE'],
            batch_size=parameters['BATCH_SIZE'],
            augment=parameters['AUGMENT'],
            shuffle = parameters['SHUFFLE']
        )

    # Initialize CNN
    resnet_model = ResNetModel(input_shape=(parameters['IMAGE_SIZE'][0],parameters['IMAGE_SIZE'][1],3),layer_name=parameters['EXTRACT_LAYER_NAME'], num_classes=parameters['NUM_CLASSES'], trainable=True)
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if retrain_cnn:
        # Train the CNN with training and validation datasets
        print(f"Training the CNN on dataset:  '{os.path.basename(paths['RAW_PATH'])}'...")
        print(f"Epochs: '{parameters['EPOCHS']}'") 
        train_CNN(
            tall_resnet = resnet_model,
            train_dataset=batched_train_dataset,
            val_dataset=batched_val_dataset,
            resnet_save_path = paths['NEW_PRIMARY_MODEL_PATH'],
            results_path = paths['TRAIN_RESULTS'],
            epochs=parameters['EPOCHS'],
            batch_size=parameters['BATCH_SIZE']
        )
        # Save the model
        print(f"CNN saved to {paths['NEW_PRIMARY_MODEL_PATH']}")
    else:
        # Load the weights from the current CNN
        print(f"CNN loaded from {paths['CURRENT_PRIMARY_MODEL_PATH']}")
        resnet_model.load_dense_weights(paths['CURRENT_PRIMARY_MODEL_PATH'])

    # Initialize Classifier NN
    classifier_model = SecondaryClassifier_NN(input_shape = parameters['FEATURE_SIZE'], num_classes=parameters['NUM_CLASSES'], trainable=True)
    classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if retrain_classifier:
        print("-" * 40)
        print(f"Training NN on feature maps from CNN layer '{parameters['EXTRACT_LAYER_NAME']}'...")
        # Extract feature maps from the CNN
        train_features_set = extract_feature_maps(resnet_model, batched_train_dataset)
        val_features_set = extract_feature_maps(resnet_model, batched_val_dataset)

        # Save the feature maps visualization
        save_feature_maps(train_features_set, paths['TRAIN_FEATURES'], title="Train Set Feature Maps Extracted from CNN")

        # Train the secondary classifier on the train set feature maps
        train_secondary_classifier(
            classifier_model,
            train_features_set,
            val_features_set,
            classifier_save_path = paths['NEW_SECONDARY_MODEL_PATH'],
            results_path = paths['TRAIN_RESULTS'],
            epochs=parameters['EPOCHS'],
            batch_size=parameters['BATCH_SIZE'],
        )
        print(f"Classifier NN saved to {paths['NEW_SECONDARY_MODEL_PATH']}")
    else:
        # Load the weights from the top of the CNN
        # print(f"Classifier NN weights loaded from CNN at {paths['CURRENT_PRIMARY_MODEL_PATH']}")
        # classifier_model.load_from_resnet(paths['CURRENT_PRIMARY_MODEL_PATH']) #load weights from RESNET

        # Or load the weights from the trained Classifier
        print(f"Classifier NN loaded from {paths['CURRENT_SECONDARY_MODEL_PATH']}")
        classifier_model.load_selftrained_weights(paths['CURRENT_SECONDARY_MODEL_PATH'])
    
    # BEGIN TESTING PHASE

    # Load the test dataset
    test_dataset = load_imageNet(
        data_path=paths['STRUCTURED_TEST_PATH'],
        image_size = parameters['IMAGE_SIZE'],
        batch_size= parameters['BATCH_SIZE'],
        augment= parameters['AUGMENT'],
        shuffle= parameters['SHUFFLE'],
        repeat = False
    )
    test_feature_set = extract_feature_maps(resnet_model, test_dataset)

    # Get the set of test images that were 'hard to classify'
    marked_hard_vector = get_marked_hard_images(resnet_model, test_dataset)

    # Initialize Q-learning model
    q_classifier = QClassifier(q_table_dir = "", num_actions = parameters['NUM_ACTIONS'], learning_rate = parameters['LEARNING_RATE'], discount_rate = parameters['DISCOUNT_RATE'], iteration_constant = parameters['ITERATION_CONSTANT'])
    
    if retrain_q_learning:
        # Train the Q-learning model: populate the Q-tables for each hard image
        print("-" * 40)
        print("\nTraining the Q-learning model...")
        train_q_tables(
            q_classifier,
            CNN = resnet_model,
            classifier = classifier_model,
            dataset = test_dataset,
            marked_hard_images = marked_hard_vector,
            output_dir = paths['NEW_Q_LEARNING_MODEL_PATH']
        )
        print(f"Q-learning model saved to {paths['NEW_Q_LEARNING_MODEL_PATH']}")
    else:
        # Load the Q-learning model: load the table directory where pre-trained Q-tables are stored
        print(f"Q-learning loaded from {paths['CURRENT_Q_LEARNING_MODEL_PATH']}")
        q_classifier.q_table_dir = paths['CURRENT_Q_LEARNING_MODEL_PATH']

    # Test the models
    print("-" * 40+"\n Testing Phase:")

    # Evaluate the models on the test dataset
    CNN_classify_test(resnet_model, test_dataset)

    # Evaluate the secondary classifier on the test dataset and the Q-learning model using the secondary classifier
    RL_classify_test(resnet_model, classifier_model, q_classifier, marked_hard_vector, test_dataset, test_feature_set, paths['PICTURES'])
    
    # Evaluate the Q-learning model using resnet
    RL_resnet_classify_test(resnet_model, q_classifier, marked_hard_vector, test_dataset,  test_feature_set)

if __name__ == "__main__":
    print("Do you need to restructure the data? (y/n)")
    data_choice = input().strip().lower()
    
    if data_choice == "y":
        restructure_data = True
    elif data_choice == "n":
        restructure_data = False
    else:
        print("Invalid choice. Please restart and choose 't' for retrain or 'l' for load.")
        exit(1)
    
    print("Do you want to retrain or load the CNN? (t/l)")
    cnn_choice = input().strip().lower()
    
    if cnn_choice == "t":
        # Display details for retraining
        retrain_cnn = True
    elif cnn_choice == "l":
        retrain_cnn = False
    else:
        print("Invalid choice. Please restart and choose 't' for retrain or 'l' for load.")
        exit(1)

    print("Do you want to retrain or load the classifying structure? (t/l)")
    classifier_choice = input().strip().lower()
    if classifier_choice == "t":
        retrain_classifier = True
    elif classifier_choice == "l":
        retrain_classifier = False
    else:
        print("Invalid choice. Please restart and choose 't' for retrain or 'l' for load.")
        exit(1)
    
    if not (restructure_data or retrain_cnn or retrain_classifier):
        print("Do you want to retrain or load the Q-learning model? (t/l)")
        q_learning_choice = input().strip().lower()
        if q_learning_choice == "t":
            retrain_q_learning = True
        elif q_learning_choice == "l":
            retrain_q_learning = False
        else:
            print("Invalid choice. Please restart and choose 't' for retrain or 'l' for load.")
            exit(1)
    else:
        retrain_q_learning = True

    print("Name for the results folder: ")
    test_name = input().strip().lower()
    if test_name == "":
            print("Name cannot be empty. Please restart and enter a valid name.")
            exit(1)

    #Call main with appropriate variables
    main(
        restructure_data = restructure_data,
        retrain_cnn= retrain_cnn,
        retrain_classifier= retrain_classifier,
        retrain_q_learning= retrain_q_learning,
        test_name = test_name
    )