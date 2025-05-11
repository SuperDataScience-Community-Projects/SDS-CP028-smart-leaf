import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Paths
original_dataset_dir = 'BangladeshiCrops/Crop___Disease'  # Root path where Corn, Potato, etc. folders are
base_dir = 'SmartLeaf_dataset'                            # Where you want to create train/val/test folders
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
# test_ratio = 0.1 (implicitly the remaining)

# Create train, val, and test directories
for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

# Traverse two levels: crop -> class
for crop_folder in os.listdir(original_dataset_dir):
    crop_path = os.path.join(original_dataset_dir, crop_folder)

    if os.path.isdir(crop_path):
        # Now go inside each disease class
        for class_folder in os.listdir(crop_path):
            class_path = os.path.join(crop_path, class_folder)

            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                random.shuffle(images)

                total_images = len(images)
                train_split = int(total_images * train_ratio)
                val_split = int(total_images * (train_ratio + val_ratio))

                train_images = images[:train_split]
                val_images = images[train_split:val_split]
                test_images = images[val_split:]

                # Create corresponding class folders under train/val/test
                train_class_dir = os.path.join(train_dir, class_folder)
                val_class_dir = os.path.join(val_dir, class_folder)
                test_class_dir = os.path.join(test_dir, class_folder)

                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(val_class_dir, exist_ok=True)
                os.makedirs(test_class_dir, exist_ok=True)

                # Copy images
                for img in train_images:
                    shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))

                for img in val_images:
                    shutil.copy2(os.path.join(class_path, img), os.path.join(val_class_dir, img))

                for img in test_images:
                    shutil.copy2(os.path.join(class_path, img), os.path.join(test_class_dir, img))

print("Dataset split into train/val/test successfully!")
