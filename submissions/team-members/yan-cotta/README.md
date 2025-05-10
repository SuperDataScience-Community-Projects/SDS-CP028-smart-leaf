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
- Implemented oversampling with `WeightedRandomSampler` to boost minority class performance.
- Initial results showed improved F1-scores for some classes (e.g., `Rice___Healthy`: +0.2299) but a higher validation loss (2.7789 vs. 0.4870) and worse average F1 (0.6365 vs. 0.7620).
- Adjusted `SAMPLING_FACTOR` to 0.2 and increased `NUM_EPOCHS` to 5 to stabilize training.
- Ready for Step 4 (data augmentation) and Step 5 (hyperparameter tuning) to further refine the model.

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

