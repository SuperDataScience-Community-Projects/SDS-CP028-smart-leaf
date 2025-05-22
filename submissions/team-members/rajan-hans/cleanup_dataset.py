import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def main():
    # 1. Set reproducibility seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 2. Define dataset directory and parameters
    DATA_DIR = r"C:/MyWork/Tech-Work/SDS/Data/CropDisease"
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 25 #25 to start with 

    # 3. Collect all image paths and class names from sub-subfolders
    all_image_paths = glob(os.path.join(DATA_DIR, '*', '*', '*.jpg'))  # Recursively collect images

    # Extract unique class names from sub-subfolder names
    all_classes = sorted(list({os.path.basename(os.path.dirname(p)) for p in all_image_paths}))
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
    print(f"Discovered {len(all_classes)} classes:\n{all_classes}")

    # Assign images to class indices
    image_paths = []
    labels = []
    for img_path in all_image_paths:
        class_name = os.path.basename(os.path.dirname(img_path))
        image_paths.append(img_path)
        labels.append(class_to_index[class_name])

    print(f"Total images found: {len(image_paths)}")
    print(f"Sample: {image_paths[0]} => Label {labels[0]} ({all_classes[labels[0]]})")

    # 4. Shuffle data to prevent order bias
    image_paths, labels = shuffle(image_paths, labels, random_state=seed)

    # 5. Load, clean, and resize images
    def load_and_resize(image_path, size):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(size)
            return np.array(img)
        except Exception as e:
            print(f"Error with image {image_path}: {e}")
            return None

    X = []
    y = []
    for img_path, label in zip(image_paths, labels):
        img = load_and_resize(img_path, IMG_SIZE)
        if img is not None:
            X.append(img)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"Dataset size after cleaning: {X.shape}, {y.shape}")



if __name__ == "__main__":
    main()