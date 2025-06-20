import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io

# --------- Load Model and Class Names ----------
@st.cache_resource
def load_model():
    model = torch.load("best_model.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

# Example: replace with actual class names used during training
class_names = [
    "Corn___Healthy", "Corn___Common_Rust", "Potato___Early_Blight", 
    "Potato___Late_Blight", "Rice___Brown_Spot", "Wheat___Leaf_Rust"
]

# --------- Image Preprocessing Function ----------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize to match model's expected input
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean (ImageNet)
                             [0.229, 0.224, 0.225])  # std (ImageNet)
    ])
    return transform(image).unsqueeze(0)  # add batch dimension

# --------- Prediction Function ----------
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        return class_names[predicted_idx.item()]

# --------- Streamlit UI ----------
st.title("🌿 Leaf Disease Classifier (PyTorch Model)")

st.write("Upload a leaf image to detect crop and disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    image_tensor = preprocess_image(image)
    prediction = predict(model, image_tensor)

    st.success(f"✅ Predicted Class: **{prediction}**")
