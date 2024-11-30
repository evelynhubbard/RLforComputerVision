import os
import shutil
import tensorflow as tf
import random
import json

def summarize_dataset(dataset_path):
     """
     Summarize the dataset by counting the number of classes and images.
     Args:
          dataset_path (str): Path to the dataset.
     Returns: 
          num_classes (int): Number of classes in the dataset.
          total_images (int): Total number of images in the dataset.
     """
     num_classes = len(os.listdir(dataset_path))
     total_images = 0
     for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        if os.path.isdir(class_path):  # Ensure it's a directory
            total_images += len(os.listdir(class_path))
     return num_classes, total_images

def restructure_imageNet(
    data_path,
    structured_train_path,
    structured_val_path,
    structured_test_path,
    label_mapping_path,
    train_count=383,
    val_count=197,
    test_count=187
    ):
    """
    Restructure the Tiny ImageNet dataset into a structured format.
    Args:
        data_path (str): Path to the Tiny ImageNet dataset.
        structured_train_path (str): Path to save the restructured train dataset.
        structured_val_path (str): Path to save the restructured validation dataset.
        structured_test_path (str): Path to save the restructured test dataset.
    """

     # Create structured train directory
    for path in [structured_train_path, structured_val_path, structured_test_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    synset_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    synset_dirs.sort()
    class_to_index = {synset: i for i, synset in enumerate(synset_dirs)}
    
    with open(label_mapping_path, 'w') as f:
        json.dump(class_to_index, f)
     

    for i, synset in enumerate(synset_dirs):
        synset_path = os.path.join(data_path, synset)
        synset_images = os.listdir(synset_path)
        random.shuffle(synset_images)

        total_needed = train_count + val_count + test_count
        if len(synset_images) < total_needed:
            raise ValueError(
                f"Not enough images in synset '{synset}' to split. "
                f"Required: {total_needed}, Found: {len(synset_images)}"
            )
        train_images = synset_images[:train_count]
        val_images = synset_images[train_count:train_count+val_count]
        test_images = synset_images[train_count+val_count:train_count+val_count+test_count]

     #    # Create class directories
     #    os.makedirs(os.path.join(structured_train_path, synset), exist_ok=True)
     #    os.makedirs(os.path.join(structured_val_path, synset), exist_ok=True)
     #    os.makedirs(os.path.join(structured_test_path, synset), exist_ok=True)

     #    for img in train_images:
     #        shutil.copy(os.path.join(synset_path, img), os.path.join(structured_train_path, synset, img))
     #    for img in val_images:
     #        shutil.copy(os.path.join(synset_path, img), os.path.join(structured_val_path, synset, img))
     #    for img in test_images:
     #        shutil.copy(os.path.join(synset_path, img), os.path.join(structured_test_path, synset, img))
          
          # Create class directories in target paths
     
        integer_label = class_to_index[synset]

        for subset, subset_images in zip(
            [structured_train_path, structured_val_path, structured_test_path],
            [train_images, val_images, test_images]
        ):
            class_dir = os.path.join(subset, str(integer_label))
            os.makedirs(class_dir, exist_ok=True)
            for img in subset_images:
                src = os.path.join(synset_path, img)
                dst = os.path.join(class_dir, img)
                shutil.copy(src, dst)

            print(f"Processed synset '{synset}' - Train: {len(train_images)}, "
              f"Val: {len(val_images)}, Test: {len(test_images)}.")

    print("ImageNet dataset restructured successfully!")

def load_imageNet(data_path, image_size = (224,224), batch_size = 32, shuffle = False, augment = False):
     
     """
     Load the Tiny ImageNet dataset from disk.
     Args:
          dataset_path (str): Path to the Tiny ImageNet dataset.
          image_size (tuple): Size to resize the images to.
          batch_size (int): Number of images per batch.
          augment (bool): If True, apply data augmentation.
     Returns:
          train_dataset tf.data.Dataset: A tf.data.Dataset object for the Tiny ImageNet train dataset.
          test_dataset tf.data.Dataset: A tf.data.Dataset object for the Tiny ImageNet value dataset.
     """

     dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        label_mode='categorical',
        image_size=image_size,  # ResNet50 input size
        batch_size=batch_size,
        shuffle = True
     )
     #dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

     # Normalize pixel values to be between 0 and 1
     #normalization_layer = tf.keras.layers.Rescaling(1./255)
     preprocess_layer = tf.keras.applications.resnet50.preprocess_input
     
     dataset = dataset.map(lambda x, y: (preprocess_layer(x), y))

     # Apply data augmentation
     if augment:
        augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2)
        ])
        dataset = dataset.map(lambda x, y: (augmentation_layer(x), y))

     return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def load_tiny_dataset(train_data_path, val_data_path, image_size = (224,224), batch_size = 32, augment = False):
     """
     Load the Tiny ImageNet dataset from disk.
     Args:
          dataset_path (str): Path to the Tiny ImageNet dataset.
          image_size (tuple): Size to resize the images to.
          batch_size (int): Number of images per batch.
          augment (bool): If True, apply data augmentation.
     Returns:
          train_dataset tf.data.Dataset: A tf.data.Dataset object for the Tiny ImageNet train dataset.
          test_dataset tf.data.Dataset: A tf.data.Dataset object for the Tiny ImageNet value dataset.
     """
     train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
        image_size=image_size,  # ResNet50 input size
        batch_size=batch_size
    )

     val_dataset = tf.keras.utils.image_dataset_from_directory(
          val_data_path,
        image_size=image_size,
        batch_size=batch_size
    )
     # Normalize pixel values to be between 0 and 1
     normalization_layer = tf.keras.layers.Rescaling(1./255)
     train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
     val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

     # Apply data augmentation
     if augment:
        augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ])
        train_dataset = train_dataset.map(lambda x, y: (augmentation_layer(x), y))

     return train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE), val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def restructure_tiny_dataset(selected_classes, train_data_path, structured_train_path, val_data_path, structured_val_path, val_annotations_file):
     """
     Restructure the Tiny ImageNet dataset to include only selected classes for training and validation
     Args:
          selected_classes (list): List of class names to include.
          train_data_path (str): Path to the original training dataset.
          structured_train_path (str): Path to save the restructured training dataset.
          val_data_path (str): Path to the original validation dataset.
          structured_val_path (str): Path to save the restructured validation dataset.
          val_annotations_file (str): Path to the validation annotations file.
     """

     # Create structured train directory
     if os.path.exists(structured_train_path):
        shutil.rmtree(structured_train_path)
     os.makedirs(structured_train_path, exist_ok=True)

     # Loop through each class directory
     for class_dir in os.listdir(train_data_path):
        if class_dir in selected_classes:
            class_path = os.path.join(train_data_path, class_dir, "images")
            if os.path.exists(class_path):
                dest_class_path = os.path.join(structured_train_path, class_dir)
                os.makedirs(dest_class_path, exist_ok=True)
                # Move images from "images/" to the class directory
                for img in os.listdir(class_path):
                    shutil.copy(os.path.join(class_path, img), dest_class_path)
     print("Training dataset restructured successfully!")


     # Create new val directory structure
     if os.path.exists(structured_val_path):
          shutil.rmtree(structured_val_path)
     os.makedirs(structured_val_path, exist_ok=True)

     # Parse annotations file and move images into class folders
     with open(val_annotations_file, "r") as file:
          annotations = file.readlines()

     for line in annotations:
          parts = line.strip().split("\t")
          img_name, class_name = parts[0], parts[1]

          if class_name in selected_classes:
               class_path = os.path.join(structured_val_path, class_name)
               os.makedirs(class_path, exist_ok=True)
               # Copy the image to the appropriate class folder
               src = os.path.join(val_data_path, "images", img_name)
               dest = os.path.join(class_path, img_name)

               if os.path.exists(src):  # Ensure the source file exists
                    shutil.copy(src, dest)

     print("Validation dataset restructured successfully!")
