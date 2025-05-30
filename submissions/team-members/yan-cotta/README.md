# Smart Leaf Disease Classification

This project implements a deep learning solution for classifying crop diseases from leaf images, focusing on crops common in Bangladesh, such as corn, potato, rice, and wheat. The project leverages a convolutional neural network (CNN) to identify 14 disease classes, addressing challenges like class imbalance and image variability through robust preprocessing and optimization techniques.

## Project Structure
```
.
├── dataset/                   # Raw dataset (gitignored)
├── dataset_organized/         # Reorganized dataset
├── split_dataset/            # Train/val/test splits
├── utils/
│   ├── __init__.py           # Makes utils a package
│   └── data_utils.py         # Utility functions for data processing and visualization
├── scripts/
│   ├── __init__.py           # Makes scripts a package
│   ├── extract.py            # Extracts dataset from zip
│   ├── check_images.py       # Validates image integrity
│   ├── analyze_classes.py    # Analyzes class distribution
│   ├── cleanup_dataset.py    # Removes corrupt images
│   ├── data_preprocessing.py # Splits dataset into train/val/test
│   ├── model_evaluation.py   # Trains and evaluates CNN model
│   ├── verify_setup.py       # Verifies environment setup
│   └── generate_metrics_plot.py # Generates aggregated metrics plot
├── requirements.txt          # Project dependencies
├── scripts/outputs/          # Output files (metrics, plots, logs)
│   ├── evaluation_results.json # Baseline evaluation metrics
│   ├── baseline_summary.txt  # Summary of baseline performance
│   ├── confusion_matrix_fold_*.png # Confusion matrices per fold
│   ├── aggregated_class_metrics.png # Class-wise performance plot
│   └── evaluation.log        # Training and evaluation logs
└── README.md                # Project documentation
```

## Features

- **Robust Data Preprocessing**: Extracts, validates, and splits the dataset into train/validation/test sets
- **Class Imbalance Handling**: Analyzes class distribution and applies class weights and oversampling (in progress)
- **Comprehensive Evaluation**: Performs 5-fold cross-validation with precision, recall, F1-score, and ROC-AUC metrics
- **Data Augmentation**: Applies standardized image transforms to enhance model robustness
- **Modular Code**: Reusable utilities for data processing, visualization, and model training
- **Detailed Outputs**: Generates confusion matrices, class-wise metrics plots, and JSON reports

## Progress (as of May 10, 2025)

### Step 1: Environment Setup
- Configured a reproducible environment with requirements.txt and verify_setup.py
- Ensured all dependencies (PyTorch, scikit-learn, imbalanced-learn, etc.) are installed
- Set random seeds (RANDOM_SEED = 42) for reproducibility

### Step 2: Baseline Model Evaluation
- Trained and evaluated a custom CNN (LeafDiseaseClassifier) using 5-fold cross-validation (3 epochs per fold)
- Achieved an average F1-score of 0.739 and validation loss of 0.487
- Identified strong classes (F1 > 0.9): Corn___Common_Rust, Corn___Healthy, Potato___Early_Blight, Potato___Late_Blight, Rice___Neck_Blast
- Noted problematic classes (F1 < 0.7): Rice___Leaf_Blast (0.2975), Rice___Healthy (0.4978), Rice___Brown_Spot (0.5184), Corn___Gray_Leaf_Spot (0.5237), Corn___Northern_Leaf_Blight (0.6895)
- Generated five confusion matrices and an aggregated metrics plot
- Documented findings in scripts/outputs/baseline_summary.txt
- Fixed a plotting error in plot_aggregated_metrics to generate aggregated_class_metrics.png

### Step 3: Addressing Class Imbalance
- Implemented class weights using CrossEntropyLoss instead of oversampling
- Achieved significant improvements:
  - Average F1 Score increased to 0.804 (from baseline 0.7620)
  - Average Validation Loss decreased to 0.409 (from baseline 0.4870)
- Generated comprehensive fold-wise metrics:
  - Fold 1: F1 = 0.792, Val Loss = 0.4591
  - Fold 2: F1 = 0.854, Val Loss = 0.3772
  - Fold 3: F1 = 0.828, Val Loss = 0.3326
  - Fold 4: F1 = 0.700, Val Loss = 0.5522
  - Fold 5: F1 = 0.845, Val Loss = 0.3236
- Identified areas for improvement in rice disease classification

### Step 4: Model Optimization
- Increased training epochs to 10 per fold for better convergence
- Fixed hyperparameters:
  - Learning Rate: 0.001
  - Batch Size: 32
  - Dropout Rate: 0.3
- Enhanced model training stability with consistent decrease in train loss
- Applied class weights effectively (e.g., 6.1203 for Potato___Healthy)

### Current Status and Next Steps
- **Strengths**: 
  - Excellent performance on Corn___Healthy and Rice___Neck_Blast
  - Improved overall metrics across most classes
  - Stable training with consistent loss reduction
- **Challenges**: 
  - Rice___Leaf_Blast and Rice___Healthy still show suboptimal performance
  - Some feature confusion between rice disease classes
- **Next Steps**:
  - Enhance model robustness for rice classes
  - Explore additional data augmentation techniques
  - Fine-tune hyperparameters for problematic classes

## Performance Metrics

### Latest Evaluation Results
- **Model Performance**:
  - Average F1 Score: 0.804
  - Average Validation Loss: 0.409
  - Consistent improvement across folds
- **Training Process**:
  - 10 epochs per fold
  - Train loss progression example (Fold 5): 0.8613 → 0.3315
  - Effective class weight application for balance

### Visualization Highlights
- Bar charts show improved class-wise performance
- Confusion matrices reveal specific areas for improvement
- Detailed metrics available in scripts/outputs/evaluation_results.json

## Utilities (utils/data_utils.py)

- **Data Transforms**: Applies resizing, normalization, and basic augmentation (flips) for training and validation
- **Image Validation**: Verifies image integrity to exclude corrupt files
- **Class Distribution**: Visualizes class distribution (class_distribution.png) to identify imbalances
- **Class Weights**: Computes weights for loss function to mitigate class imbalance

## Installation

### Clone the Repository:
```bash
git clone https://github.com/YanCotta/SDS-CP028-smart-leaf.git
cd SDS-CP028-smart-leaf/submissions/team-members/yan-cotta
```

### Create and Activate a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

#### Requirements.txt includes:
```
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0
imbalanced-learn==0.11.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
Pillow==10.0.0
splitfolders==0.5.1
```

### Verify Setup:
```bash
python scripts/verify_setup.py
```
This checks dependencies, directory structure, and script integrity.

## Dataset

### Download:
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease)
2. Place new-bangladeshi-crop-disease.zip in the project root

### Prepare:
- The dataset is gitignored to manage repository size
- Extract and preprocess using provided scripts (see Usage)

## Usage

Run scripts in sequence to process the dataset and evaluate the model:

### Extract Dataset:
```bash
python scripts/extract.py
```
Extracts new-bangladeshi-crop-disease.zip to dataset/

### Validate Images:
```bash
python scripts/check_images.py
```
Identifies and removes corrupt images, logging results to image_validation_log.txt

### Clean Dataset:
```bash
python scripts/cleanup_dataset.py
```
Removes invalid images based on check_images.py output

### Analyze Class Distribution:
```bash
python scripts/analyze_classes.py
```
Generates class_distribution.png showing class imbalances

### Preprocess Dataset:
```bash
python scripts/data_preprocessing.py
```
Reorganizes dataset/ into dataset_organized/ and splits into split_dataset/

### Evaluate Model:
```bash
python scripts/model_evaluation.py
```
Trains and evaluates the CNN with 5-fold cross-validation

### Generate Metrics Plot:
```bash
python scripts/generate_metrics_plot.py
```
Creates aggregated_class_metrics.png from evaluation_results.json

## Running the Streamlit Application

To visualize predictions on your own leaf images, you can use the Streamlit application.

1.  **Ensure Dependencies are Installed**:
    If you haven't already, install all necessary packages:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure your virtual environment is activated.)

2.  **Navigate to the Correct Directory**:
    The Streamlit app `app.py` is located in the `submissions/team-members/yan-cotta/` directory. The command to run it should typically be executed from the project root directory (`SDS-CP028-smart-leaf`) or from within the `submissions/team-members/yan-cotta/` directory if you adjust paths accordingly. Assuming you are in the `submissions/team-members/yan-cotta` directory:
    ```bash
    # If you are in SDS-CP028-smart-leaf/ (project root):
    # cd submissions/team-members/yan-cotta 
    # Or, if already in submissions/team-members/yan-cotta/, you can proceed.
    ```

3.  **Ensure the Model is Available**:
    The Streamlit application requires the trained model file `best_leaf_disease_model.pth`. This file is generated by the model evaluation script. If it's not present in `submissions/team-members/yan-cotta/scripts/outputs/`, run the evaluation first:
    ```bash
    python scripts/model_evaluation.py
    ```

4.  **Run the Streamlit App**:
    Once the dependencies are installed and the model file is available, start the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    This will typically open the application in your default web browser.

## Output Files

### Data Analysis:
- class_distribution.png: Visualizes class distribution
- image_validation_log.txt: Logs corrupt image checks

### Model Evaluation:
- evaluation_results.json: Baseline metrics
- evaluation_results_smote.json: Oversampling metrics
- confusion_matrix_fold_*.png: Confusion matrices for each fold
- aggregated_class_metrics.png: Class-wise performance metrics
- baseline_summary.txt: Summary of baseline evaluation
- evaluation.log: Detailed training/evaluation logs

## Dependencies

- torch & torchvision: Deep learning framework
- scikit-learn: Metrics and cross-validation
- imbalanced-learn: Oversampling techniques
- numpy & pandas: Data manipulation
- matplotlib & seaborn: Visualization
- Pillow: Image processing
- splitfolders: Dataset splitting

See requirements.txt for version details.

## Notes

- Class Imbalance: Addressed using class weights and WeightedRandomSampler
- Performance: Baseline F1-score of 0.739 with issues in minority classes
- Reproducibility: Random seeds are set
- Hardware: CPU used for baseline; GPU recommended
- Dataset: Must be downloaded manually

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License.

