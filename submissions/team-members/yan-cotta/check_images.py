import os
from PIL import Image

def check_images(directory):
    corrupt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(os.path.join(root, file))
                    img.verify()  # Checks if the file is valid
                except (IOError, SyntaxError) as e:
                    corrupt_path = os.path.join(root, file)
                    corrupt_files.append(corrupt_path)
                    print(f"Corrupt image: {corrupt_path}")
    return corrupt_files

# Run the check
dataset_dir = "dataset"
corrupt_files = check_images(dataset_dir)

# Remove corrupt files
for file in corrupt_files:
    os.remove(file)
    print(f"Removed: {file}")
