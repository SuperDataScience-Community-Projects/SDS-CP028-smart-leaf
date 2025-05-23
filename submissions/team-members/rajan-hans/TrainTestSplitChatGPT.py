#requires  
# #pip install tensorflow scikit-learn
# pip install pylance

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
import random

# Define directories
base_dir = r"C:/MyWork/Tech-Work/SDS/Data/CropDisease"
output_dir = r"C:/MyWork/Tech-Work/SDS/Data/CropDiseaseSplit"

# Get all image paths with labels
image_paths = []
labels = []

for crop in os.listdir(base_dir):
    crop_path = os.path.join(base_dir, crop)
    if os.path.isdir(crop_path):
        for disease in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease)
            if os.path.isdir(disease_path):
                for image_name in os.listdir(disease_path):
                    image_paths.append(os.path.join(disease_path, image_name))
                    labels.append(f"{crop}_{disease}")

# First split (train_temp: 90%, test: 10%)
train_temp_paths, test_paths, train_temp_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.10, random_state=42, stratify=labels
)

# Second split (train: 70%, validation: 20% of original dataset)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_temp_paths, train_temp_labels, test_size=2/9, random_state=42, stratify=train_temp_labels
)

# Utility function to copy files into directories
def copy_images(image_list, label_list, subset_name):
    for img_path, label in zip(image_list, label_list):
        dest_dir = os.path.join(output_dir, subset_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, dest_dir)

# Copy images into respective folders
copy_images(train_paths, train_labels, 'train')
copy_images(val_paths, val_labels, 'validation')
copy_images(test_paths, test_labels, 'test')

print("Image dataset split completed successfully.")
