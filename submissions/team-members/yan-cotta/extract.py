import zipfile
import os

zip_path = "new-bangladeshi-crop-disease.zip"
extract_dir = "dataset"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted dataset to {extract_dir}")

