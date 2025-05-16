import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("chiti_nkhuwa_smart_leaf_Model.keras")

# Define class names (in order)
class_names = [
    'Rice___Leaf_Blast', 'Potato___Late_Blight', 'Corn___Common_Rust',
    'Rice___Neck_Blast', 'Corn___Northern_Leaf_Blight', 'Wheat___Brown_Rust',
    'Corn___Healthy', 'Corn___Gray_Leaf_Spot', 'Potato___Healthy',
    'Rice___Healthy', 'Rice___Brown_Spot', 'Wheat___Healthy',
    'Wheat___Yellow_Rust', 'Potato___Early_Blight'
]

# Preprocess input image
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict function
def predict_leaf_disease(image):
    processed = preprocess(image)
    prediction = model.predict(processed)[0]
    top_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)
    return f"{top_class} ({confidence}%)"

# Create Gradio app
interface = gr.Interface(
    fn=predict_leaf_disease,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Smart Leaf Disease Detector",
    description="Upload an image of a crop leaf to detect possible disease."
)

if __name__ == "__main__":
    interface.launch()
