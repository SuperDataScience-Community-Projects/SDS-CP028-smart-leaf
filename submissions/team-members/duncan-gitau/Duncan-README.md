🌾 Bangladeshi Crop Disease Classification using CNN
This project builds a Convolutional Neural Network (CNN) model to classify Bangladeshi crop diseases using image data.
The dataset is sourced from Kaggle, and the model is developed using PyTorch.

📂 Dataset
Source: New Bangladeshi Crop Disease Dataset by Nafisa Homaira Moin

Download: Automated using KaggleHub

Content: Categorized images of crop diseases from Bangladesh.

🧪 Model Architecture
CNN-based classification using pre-trained models from torchvision.models

Data augmentation with PyTorch transforms

Adaptive learning rate scheduling via ReduceLROnPlateau

Evaluation with a confusion matrix and classification report.

🛠️ Tech Stack
Python

PyTorch

Torchvision

KaggleHub

Matplotlib & Seaborn

Scikit-learn

TQDM (progress bars)

Optuna (for hyperparameter tuning)

🖼️ Data Augmentation
Applied to training set:

Random resized crops

Horizontal and vertical flips

Rotations and affine transformations

Perspective distortion

Color jittering

Validation set:

Resizing and normalization only.

🚀 Training Setup
Dataset split: 80% training / 20% validation

DataLoader optimized with num_workers=4

Model supports Apple Silicon (MPS backend) if available.

🔧 Requirements
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
📎 Acknowledgements
Dataset: Nafisa Homaira Moin (Kaggle)

Libraries: PyTorch, TorchVision
