# Welcome to the SuperDataScience Community Project!
Welcome to the Smart Leaf: Deep Learning-Based Multi-Crop Disease Detection repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:

## Project Overview
**SmartLeaf** focuses on building a convolutional neural network (CNN) to classify and detect diseases in four crop species (Corn, Potato, Rice, and Wheat). With **14 distinct classes** and **13,024 images**, this project leverages state-of-the-art deep learning techniques to identify common leaf diseases (and healthy leaves) with high accuracy. Once developed, the model will be deployed via Streamlit for a user-friendly interface that provides real-time disease predictions.

Link to Dataset: https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease

## Project Objectives
During preprocessing, all images should be resized (e.g., to 224×224), normalized (e.g., scale pixel values to [0,1] or standardize mean and variance), and then augmented to improve model generalization. Below is a recommended set of data augmentation transformations to randomly apply during training, along with common utilities from **PyTorch** and **TensorFlow** (no code, just tool names) that you can refer to in each framework’s documentation:

1. **Random Horizontal Flip**  
   - **PyTorch:** `transforms.RandomHorizontalFlip(p=...)`  
   - **TensorFlow:** `tf.image.random_flip_left_right`  
   - **Benefit**: Helps the model learn orientation invariance.

2. **Random Vertical Flip**  
   - **PyTorch:** `transforms.RandomVerticalFlip(p=...)`  
   - **TensorFlow:** `tf.image.random_flip_up_down`  
   - **Benefit**: Increases diversity in leaf orientation.

3. **Random Rotation**  
   - **PyTorch:** `transforms.RandomRotation(degrees=...)`  
   - **TensorFlow:** `tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=...)`  
   - **Benefit**: Handles various angles at which leaves might be captured.

4. **Random Zoom**  
   - **PyTorch:** `transforms.RandomAffine(scale=(..., ...))` or `transforms.RandomResizedCrop(...)`  
   - **TensorFlow:** `tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=...)`  
   - **Benefit**: Helps the model adapt to different leaf sizes and distances.

5. **Random Shift (Translation)**  
   - **PyTorch:** `transforms.RandomAffine(translate=(..., ...))`  
   - **TensorFlow:** `tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=..., height_shift_range=...)`  
   - **Benefit**: Improves robustness to off-center leaf images.

6. **Random Brightness & Contrast Adjustments**  
   - **PyTorch:** `transforms.ColorJitter(brightness=..., contrast=...)`  
   - **TensorFlow:** `tf.image.adjust_brightness` and `tf.image.adjust_contrast` (or `ImageDataGenerator` params)  
   - **Benefit**: Compensates for variability in lighting conditions.

7. **Color Jitter (Optional)**  
   - **PyTorch:** `transforms.ColorJitter(hue=..., saturation=...)`  
   - **TensorFlow:** `tf.image.adjust_hue`, `tf.image.adjust_saturation` (or `ImageDataGenerator` params)  
   - **Benefit**: Accommodates for real-world variations in leaf color due to environment or camera settings.

## Workflow

### **Phase 1: Setup (1 Week)**
- Setup GitHub repo and project folders
- Setup virtual environment and respective libraries

### **Phase 2: EDA (1 Week)**
1. **Data Collection**  
   - Ensure all images are properly labeled into the 14 distinct classes.
2. **Data Cleaning & Exploration**  
   - Verify image shapes, remove corrupted files, and inspect class balance.
3. **Data Preprocessing**  
   - Resize images (e.g., 224×224) and normalize pixel values.  
   - Apply augmentation (rotations, flips, shifts) to improve model robustness.

### **Phase 3: Model Development (2 Weeks)**
1. **Architecture Design**  
   - Select or design a CNN (custom or transfer learning with ResNet, MobileNet, etc.).
2. **Training & Validation**  
   - Train the CNN, tracking accuracy and loss curves.  
   - Tune hyperparameters (learning rate, batch size, etc.).
3. **Performance Evaluation**  
   - Use accuracy, precision, recall, F1-score, and confusion matrix to measure results.  
   - Adjust model or data preprocessing if necessary.

### **Phase 4: Deployment (1 Week)**
1. **Streamlit App**  
   - Create an app that allows image uploads and displays predictions in real time.  
   - Show confidence scores or Grad-CAM explanations for interpretability. (optional)
1. **Documentation & Hosting**  
   - Prepare a clear README.  
   - Deploy the app on Streamlit Community Cloud or another hosting service.

## Timeline

| Phase                                | Task                                      | Duration   |
| ------------------------------------ | ----------------------------------------- | ---------- |
| **Phase 1: Setup**                   | GitHub repo, environment setup            | Week 1     |
| **Phase 2: Dataset & Preprocessing** | Collect, clean, preprocess & augment data | Week 2     |
| **Phase 3: Model Development**       | Design, train & evaluate CNN              | Week 3 & 4 |
| **Phase 4: Deployment**              | Build & deploy Streamlit application      | Week 5     |
