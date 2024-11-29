import os
import shutil
import tensorflow as tf

def load_tiny_imagenet(train_data_path, val_data_path, image_size = (224,224), batch_size = 32, augment = False):
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
