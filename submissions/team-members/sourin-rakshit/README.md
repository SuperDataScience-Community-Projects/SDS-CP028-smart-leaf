# 🌾 SmartLeaf: Bangladeshi Crop Disease Classification using CNN

This project builds a convolutional neural network (CNN) using PyTorch to classify crop diseases in Bangladesh. The model leverages transfer learning with EfficientNetV2-S architecture to identify diseases across four major crops: **Corn, Potato, Rice, and Wheat**.

---

## 📂 Dataset

The dataset contains **13,024 images** of crop leaves categorized into **14 distinct classes** (including healthy and diseased states).

* **Source**: [New Bangladeshi Crop Disease Dataset](https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease)
* **Download Method**: Automated using [`kagglehub`](https://github.com/KaggleHub/kagglehub)

---

## 🧪 Model Architecture

* **EfficientNetV2-S** for transfer learning (pretrained on ImageNet).
* Customized fully-connected layers for 14-class crop disease classification.
* Adaptive learning rate adjustment using `ReduceLROnPlateau` scheduler.
* Model evaluation through confusion matrix, accuracy, precision, recall, and F1-score metrics.

---

## 🖼️ Data Augmentation

Applied exclusively on the training dataset to enhance model generalization:

* Random resized crops
* Horizontal & vertical flips
* Rotations & affine transformations
* Perspective distortion
* Color jittering (brightness, contrast, saturation, hue adjustments)

Validation data undergo resizing and normalization only.

---

## 🚀 Training Setup

* Data split: 80% training / 20% validation.
* Batch loading optimized with `num_workers=4`.
* Compatible with GPU (`cuda`) and Apple Silicon (`mps`) for accelerated training.

---

## 🧰 Tech Stack

* Python
* PyTorch
* Torchvision
* KaggleHub
* Matplotlib, Seaborn
* Scikit-learn
* TQDM
* Optuna (for hyperparameter tuning)

---

## 🔧 Requirements & Installation

Install necessary dependencies:

```bash
pip install -r requirements.txt
```

Ensure dataset download via `kagglehub`:

```bash
kagglehub datasets download nafishamoin/new-bangladeshi-crop-disease
```

---

## 📈 Model Performance

The trained model achieves:

* **Accuracy:** \~95%
* **Precision, Recall, F1-score:** \~92-95% across classes
* Clearly visualized confusion matrix highlighting class-level performance.

---

## 🖥️ Usage

Run training and evaluation by executing the notebook (`Sourin-Smart-Leaf-EfficientNet.py`).

Launch the **Streamlit app** for real-time disease prediction:

```bash
streamlit run app.py
```

---

## 📎 Acknowledgements

* Dataset provided by **Nafisa Homaira Moin** on Kaggle.
* PyTorch and TorchVision libraries for model implementation.
