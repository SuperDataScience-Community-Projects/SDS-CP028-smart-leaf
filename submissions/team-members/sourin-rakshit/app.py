# app.py

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_v2_s

# ── 1) CLASS NAMES & SETTINGS ─────────────────────────────────────────────
CLASS_NAMES = [
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
THRESHOLD = 0.8  # only show preds ≥ 80%


# ── 2) TRANSFORMS ───────────────────────────────────────────────────────────
# Inference transform (we'll apply it to each TTA crop)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 3) MODEL LOADING ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)

    # build EfficientNet-V2-S and swap head
    model = efficientnet_v2_s(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, num_classes)
    )

    # load your trained weights
    state = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device

# ── 4) TTA INFERENCE FUNCTION ───────────────────────────────────────────────
def tta_predict(img: Image.Image, model, device):
    """
    Apply simple TTA: original, H‐flip, V‐flip → average softmax probs.
    """
    crops = []
    for fn in (lambda x: x, TF.hflip, TF.vflip):
        aug = fn(img)
        tensor = val_transform(aug)
        crops.append(tensor)
    batch = torch.stack(crops).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)
    return probs.mean(dim=0)  # average over TTA

# ── 5) STREAMLIT UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="🌾 Smart Leaf Classifier", layout="centered")
st.title("🌾 Smart Leaf: Crop Disease Classifier")
st.write("Upload leaf images; only predictions ≥ 80% are shown, otherwise you'll get the top‐1 guess flagged as low confidence.")

files = st.file_uploader(
    "Choose JPG/PNG images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:
    model, device = load_model()

    for file in files:
        img = Image.open(file).convert("RGB")
        st.image(img, caption=file.name, use_column_width=True)

        # predict with TTA
        probs = tta_predict(img, model, device)

        # filter high‐confidence
        high_conf = [
            (CLASS_NAMES[i], float(probs[i]))
            for i in range(len(probs))
            if probs[i] >= THRESHOLD
        ]

        if high_conf:
            st.markdown("**Predictions (≥ 80%):**")
            for label, p in sorted(high_conf, key=lambda x: x[1], reverse=True):
                st.write(f"- {label}: {p*100:.1f}%")
        else:
            # fallback to top‐1 with a warning
            top_p, top_i = probs.max(0)
            label = CLASS_NAMES[top_i]
            st.warning(f"No class reached 80% confidence. "
                       f"Top guess: **{label}** ({top_p*100:.1f}%)")

