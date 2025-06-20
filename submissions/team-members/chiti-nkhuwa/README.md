# 🌿 Smart Leaf: Multi-Crop Disease Detection

This project uses deep learning to classify plant leaf diseases across four major crops — Corn, Potato, Rice, and Wheat — covering 14 distinct disease and healthy classes.

---

## 📁 Files Included

- `chiti_nkhuwa_smart_leaf_Model.keras` — trained Keras model  
- `chiti_nkhuwa_smart_leaf_project.ipynb` — Colab notebook with EDA, model building & evaluation  
- `app.py` — Gradio web interface to run the model  
- `requirements.txt` — required libraries for deployment  
- `README.md` — this file

---

## 📒 Notebook Summary

The notebook walks through:

- Data cleaning and preprocessing
- Image resizing and augmentation
- Class imbalance visualization
- Multiple model iterations
- Use of callbacks (early stopping, learning rate scheduling)
- Final training and evaluation (Test accuracy: ~87%)

This notebook serves as both a reference and documentation of the model development process.

## 🚀 Deployment

To run the Smart Leaf Gradio app locally:

### 1. Clone the repository and navigate to the folder

```bash
git clone https://github.com/ChitiNkhuwa/SDS-CP028-smart-leaf.git
cd SDS-CP028-smart-leaf/chiti-nkhuwa
```

### 2. Install required dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Gradio app

```bash
python app.py
```

This will launch a browser-based interface at a local address (e.g. http://127.0.0.1:7860) and a public Gradio share link.

---

## 🖼️ Preview

<img width="1277" alt="Screenshot 2025-05-16 at 3 48 36 PM" src="https://github.com/user-attachments/assets/cd62eecd-5d0c-4b06-a15e-62fa66f91ba8" />

