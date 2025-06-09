# Smart Leaf Project Setup

This repository contains a Jupyter notebook for plant disease classification using deep learning and a Streamlit web application for making predictions.

## Setup Instructions

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the models:
```bash
chmod +x download_models.sh
./download_models.sh
```

## Data Structure
The project expects the following directory structure:
```
.
├── data/              # Original dataset
├── split-data/        # Train/validation/test splits
│   ├── train/
│   ├── val/
│   └── test/
├── model.keras        # CNN model trained from scratch (downloaded)
└── efficientnetb0.keras  # Fine-tuned EfficientNetB0 model (downloaded)
```

## Running the Notebook
1. Ensure your virtual environment is activated
2. Start Jupyter notebook:
```bash
jupyter notebook
```
3. Open `david-notebook.ipynb`
4. Run all cells up to (but not including) the Model Development section

## Running the Web Application
1. Ensure your virtual environment is activated and models are downloaded
2. Start the Streamlit app:
```bash
streamlit run app.py
```
3. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Web Application Features
- Upload images of crop leaves for disease detection
- Choose between two models:
  1. CNN from Scratch: A custom convolutional neural network built from scratch
  2. EfficientNetB0 (Fine-tuned): A transfer learning model based on the EfficientNetB0 architecture
- Get instant predictions with confidence scores
- Color-coded results for healthy and diseased predictions
- Informative model descriptions and architecture details

## Models
The project uses two trained models that will be downloaded automatically using the `download_models.sh` script:
1. `model.keras` (273.7 MB): Custom CNN trained from scratch
2. `efficientnetb0.keras` (56 MB): Fine-tuned EfficientNetB0 model

If you encounter any download issues:
- Ensure you have a stable internet connection
- Try downloading directly from [Google Drive](https://drive.google.com/drive/folders/1vagecyU5UA08D5p43h8cQ2l6B77ixeLv?usp=drive_link)
- Place the downloaded files in the project root directory

## Important Notes
- The data splitting code in the notebook is commented out as it only needs to be run once
- The file renaming code is also commented out for the same reason
- Make sure all image files are in their respective class folders under the data directory
- Both model files must be present in the root directory for the web application to work properly 