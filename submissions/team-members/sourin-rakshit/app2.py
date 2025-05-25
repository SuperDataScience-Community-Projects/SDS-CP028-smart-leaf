import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# Set page configuration
st.set_page_config(
    page_title="Smart Leaf Disease Classifier",
    page_icon="🍃",
    layout="wide"
)

# Define the class names based on the user's model
class_names = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
]

# Function to load the model
@st.cache_resource
def load_model():
    # Check if model file exists
    if not os.path.exists('best_model.pth'):
        st.error("Model file 'best_model.pth' not found. Please upload the model file.")
        return None
    
    # Load the model
    model = efficientnet_v2_s(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing function
def preprocess_image(image):
    # Define the same transformation as in the original code for validation/testing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# Function to make predictions
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        return predicted_class.item(), confidence.item(), probabilities.squeeze().tolist()

# Function to generate Grad-CAM visualization
def get_gradcam(model, layer, image_tensor, class_idx):
    activations, gradients = [], []
    
    def forward_hook(m, input, output):
        activations.append(output.detach())
    
    def backward_hook(m, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    handle_forward = layer.register_forward_hook(forward_hook)
    handle_backward = layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    model.zero_grad()
    output = model(image_tensor)
    
    # Target for backprop
    score = output[0, class_idx]
    score.backward()
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Get activation and gradient
    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    
    # Calculate weights and apply to activation maps
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * act).sum(0)
    
    # Apply ReLU and normalize
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam.cpu().numpy()

# Function to display Grad-CAM overlay
def display_gradcam(model, image, image_tensor, predicted_class):
    # Get the last convolutional layer for Grad-CAM
    target_layer = model.features[-1]
    
    # Generate Grad-CAM
    cam = get_gradcam(model, target_layer, image_tensor, predicted_class)
    
    # Convert image to numpy array for display
    img_np = np.array(image.resize((224, 224)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display original image
    ax.imshow(img_np)
    
    # Overlay heatmap
    ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.axis('off')
    
    # Return the figure
    return fig

# Main app
def main():
    st.title("Smart Leaf Disease Classifier")
    st.write("Upload leaf images to identify diseases")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("Please upload the model file 'best_model.pth' to the app directory.")
        return
    
    # File uploader
    uploaded_files = st.file_uploader("Choose leaf image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    
    if uploaded_files:
        # Create columns based on number of uploaded files
        cols = st.columns(min(3, len(uploaded_files)))
        
        for i, uploaded_file in enumerate(uploaded_files):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                # Display image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption=f"Uploaded Image {i+1}", use_container_width=True)
                
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                # Make prediction
                predicted_class, confidence, all_probs = predict(model, image_tensor)
                
                # Display prediction
                st.success(f"Prediction: {class_names[predicted_class]}")
                st.info(f"Confidence: {confidence*100:.2f}%")
                


if __name__ == "__main__":
    main()
