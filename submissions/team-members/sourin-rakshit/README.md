# 🌾 Bangladeshi Crop Disease Classification using CNN

This project focuses on building a convolutional neural network (CNN) model to classify Bangladeshi crop diseases using image data. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease), and PyTorch was used to develop and train the deep learning model.

---

## 📂 Dataset

The dataset is hosted on Kaggle and contains categorized images of crop diseases from Bangladesh.

- **Source**: [New Bangladeshi Crop Disease Dataset](https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease)
- **Download**: Automated via [`kagglehub`](https://github.com/KaggleHub/kagglehub)

---

## 🧪 Model Architecture

- CNN-based classification using pre-trained models (from `torchvision.models`)
- Augmentation using PyTorch `transforms`
- Adaptive learning rate with `ReduceLROnPlateau`
- Evaluation with confusion matrix and classification report

---

## 🧰 Tech Stack

- Python
- PyTorch
- Torchvision
- KaggleHub
- Matplotlib, Seaborn
- Scikit-learn
- TQDM
- Optuna

---

## 🖼️ Data Augmentation

Applied only to the training set:

- Random resized crops
- Horizontal & vertical flips
- Rotations & affine transformations
- Perspective distortion
- Color jittering

Validation set is resized and normalized only.

---

## 🚀 Training Setup

- Dataset is split into 80% training and 20% validation.
- Data loaders use `num_workers=4` for faster loading.
- Model runs on Apple Silicon using Metal backend if available (`mps` device).

---

## 🔧 Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```
---

## 📎 Acknowledgements
Dataset by Nafisa Homaira Moin on Kaggle

PyTorch and TorchVision for model development
