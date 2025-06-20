#!/bin/bash
set -e  # Exit on any error

echo "Downloading models..."

# Download CNN model (model.keras)
echo "Downloading Custom CNN model..."
gdown "https://drive.google.com/uc?id=1GJwsPdwcIjQJOSPFgYXrxBuR7eQp9Q0V" -O model.keras

# Download EfficientNetB0 model (efficientnetb0.keras)
echo "Downloading EfficientNetB0 model..."
gdown "https://drive.google.com/uc?id=1efntqU3xrwthRDtHIllGH5XcDO0_9FZG" -O efficientnetb0.keras

echo "All models downloaded successfully!" 