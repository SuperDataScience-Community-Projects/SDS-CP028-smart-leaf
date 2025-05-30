import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
from pathlib import Path

# --- Configuration ---
NUM_CLASSES = 14
CLASS_NAMES = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight',
    'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Hispa', 'Rice___Leaf_Blast', 'Rice___Neck_Blast',
    'Wheat___Brown_Rust', 'Wheat___Healthy'
]
# Determine the correct model path relative to this app.py file
# app.py is in submissions/team-members/yan-cotta/
# model is in submissions/team-members/yan-cotta/scripts/outputs/best_leaf_disease_model.pth
MODEL_PATH = Path(__file__).parent / "scripts" / "outputs" / "best_leaf_disease_model.pth"

# --- Model Definition ---
class LeafDiseaseResNet(nn.Module):
    """ResNet18 model for leaf disease classification."""
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=False) # Set pretrained=False as we load custom weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # No need to freeze layers here as we are loading a fully trained model state

    def forward(self, x):
        return self.model(x)

# --- Image Transformations ---
def get_transform() -> transforms.Compose:
    """Returns image transformations for validation/inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Model Loading ---
@st.cache_resource # Cache the model loading for efficiency
def load_model_cached(model_path, num_classes, device):
    """Loads the pre-trained model."""
    model = LeafDiseaseResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Streamlit UI ---
def run_app(model, device, transform):
    """Creates the Streamlit user interface."""
    st.title("Leaf Disease Classification 🌿")
    st.write("Upload an image of a plant leaf to classify its disease.")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                predicted_class_name = CLASS_NAMES[predicted_idx]
                confidence = probabilities[predicted_idx].item()

            st.success(f"Prediction: {predicted_class_name}")
            st.info(f"Confidence: {confidence:.2%}")

            # Display probabilities for all classes
            st.subheader("Prediction Probabilities:")
            probs_df = {CLASS_NAMES[i]: f"{probabilities[i].item():.2%}" for i in range(NUM_CLASSES)}
            st.table(probs_df)

        except Exception as e:
            st.error(f"Error processing the image: {e}")
            # Log the error for debugging if needed
            # print(f"Error: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    current_device = torch.device('cpu') # Use CPU for inference
    
    # Check if model file exists
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        st.stop()

    try:
        loaded_model = load_model_cached(MODEL_PATH, NUM_CLASSES, current_device)
        image_transform = get_transform()
        run_app(loaded_model, current_device, image_transform)
    except Exception as e:
        st.error(f"Failed to load the model or run the app: {e}")
        # Log the error for debugging if needed
        # print(f"Startup Error: {e}")
