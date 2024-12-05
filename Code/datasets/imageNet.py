import os
import shutil
import tensorflow as tf
import random
import json
import numpy as np

seed = 42
random.seed(seed)          # For Python's random module
np.random.seed(seed)       # For NumPy
tf.random.set_seed(seed)  # For TensorFlow

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
     processed/
     ├──ImageNet/
     │    ├── structured_test/  
     │    │    ├── 0/
     │    |    │    ├── image1.JPEG
     │    |    │    ├── image2.JPEG
     │    |    │    ├── ...
     │    │    ├── 1/
     │    |    │    ├── ...
     │    │    ├── ...      
     │    ├── structured_train/
     │    ├── structured_val/    
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
    print("-" * 40)
    
def load_imageNet(data_path, image_size, batch_size, shuffle, augment = False, repeat = True):
     """
     Load the Tiny ImageNet dataset from disk.
     """

     dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        label_mode='categorical',
        image_size=image_size,  # ResNet50 input size
        batch_size=batch_size,
        shuffle = shuffle,
        seed = 42,
     )
     dataset = dataset.unbatch().batch(batch_size, drop_remainder=True)
     if repeat:
          dataset = dataset.repeat()  
     

     # Normalize pixel values to be between 0 and 1
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