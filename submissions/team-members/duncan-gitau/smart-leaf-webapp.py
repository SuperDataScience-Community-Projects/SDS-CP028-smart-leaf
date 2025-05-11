import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# Download model from Google Drive only if it doesn't already exist
model_path = "resnet50_crop_disease.pth"
if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/1yT9nnsq9OeQPTpx07ThNdQeILXfx7HyU/view?usp=sharing"
    gdown.download(url, model_path, quiet=False, fuzzy=True)


# Define class labels
class_names = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy','Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight','Potato___Healthy','Potato___Late_Blight',
    'Rice___Brown_Spot','Rice___Healthy','Rice___Leaf_Blast','Rice___Neck_Blast',
    'Wheat___Brown_Rust','Wheat___Healthy','Wheat___Yellow_Rust'
]

NUM_CLASSES = 14

# Load model
@st.cache_resource
def load_model():
    model = resnet50(weights=None)

    # Match your trained model's head
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    ckpt = torch.load("resnet50_crop_disease.pth", map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

st.title("🌿 Crop Disease Detection with ResNet-50")
st.markdown("Upload a leaf image or capture from your webcam to identify crop disease.")

# Image input method
input_type = st.radio("Choose Input Method", ("Upload Image", "Capture with Camera"))

image = None

if input_type == "Upload Image":
    #uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg","JPG"])
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif input_type == "Capture with Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        image = Image.open(camera_input).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

# Inference
if st.button("Predict Disease"):
    if image is not None:
        with st.spinner("Analyzing..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                prediction = class_names[pred.item()]
            st.success(f"✅ Prediction: **{prediction}**")
    else:
        st.warning("Please upload or capture an image first.")
