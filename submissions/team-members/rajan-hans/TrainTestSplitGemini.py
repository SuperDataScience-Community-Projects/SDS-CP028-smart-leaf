import os
import shutil
import random

source_root = 'CropDisease' # Your main folder path
destination_root = 'SplitDataset' # Where you want the new split data to go

split_ratios = {'train': 0.7, 'validation': 0.2, 'test': 0.1}

# Create top-level split directories
for split_name in split_ratios.keys():
    os.makedirs(os.path.join(destination_root, split_name), exist_ok=True)

# Walk through the source directory structure
for crop_folder in os.listdir(source_root):
    crop_path = os.path.join(source_root, crop_folder)
    if os.path.isdir(crop_path):
        for disease_folder in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease_folder)
            if os.path.isdir(disease_path):
                # Get all image files in the current disease folder
                images = [f for f in os.listdir(disease_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                random.shuffle(images) # Shuffle to ensure randomness

                # Calculate split counts
                num_images = len(images)
                train_count = int(num_images * split_ratios['train'])
                val_count = int(num_images * split_ratios['validation'])
                # Test count takes the remainder to ensure all images are used
                test_count = num_images - train_count - val_count

                # Divide images based on calculated counts
                train_images = images[:train_count]
                val_images = images[train_count : train_count + val_count]
                test_images = images[train_count + val_count : ]

                splits = {
                    'train': train_images,
                    'validation': val_images,
                    'test': test_images
                }

                # Copy files to their respective destinations
                for split_name, image_list in splits.items():
                    dest_crop_path = os.path.join(destination_root, split_name, crop_folder)
                    dest_disease_path = os.path.join(dest_crop_path, disease_folder)
                    os.makedirs(dest_disease_path, exist_ok=True) # Create necessary sub-subfolders

                    for image_name in image_list:
                        src_path = os.path.join(disease_path, image_name)
                        dst_path = os.path.join(dest_disease_path, image_name)
                        shutil.copy(src_path, dst_path)
                print(f"Split completed for {crop_folder}/{disease_folder}. Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

print(f"Data splitting complete. Check the '{destination_root}' directory.")