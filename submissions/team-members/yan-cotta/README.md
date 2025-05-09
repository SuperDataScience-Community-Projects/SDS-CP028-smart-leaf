# Week 2 Submission: Data Preprocessing by Yan

## Tasks Completed

- **Dataset Retrieval**: Extracted the Kaggle dataset using `extract.py`.
- **Image Verification**: Checked for corrupt images with `check_images.py` (no issues found).
- **Class Imbalance Analysis**: Analyzed class distribution and visualized it in `class_distribution.png` using `analyze_classes.py`.
- **Data Preprocessing**: Split the dataset into training (70%), validation (15%), and test (15%) sets with `data_preprocessing.py`.

## Files

- `extract.py`: Extracts the dataset zip file.
- `check_images.py`: Verifies image integrity.
- `analyze_classes.py`: Analyzes and plots class distribution.
- `class_distribution.png`: Visualization of class distribution.
- `data_preprocessing.py`: Splits dataset into train/validation/test sets.

## Notes

- Class Folders: Confirmed in `split_dataset/train`, `split_dataset/val`, and `split_dataset/test` with no "Invalid" folder issues.
- Check `class_distribution.png` for specific class imbalances (e.g., some classes may have zero samples).

## How to Reproduce This Work

To replicate the preprocessing steps, follow these instructions. Since the dataset is too large to include, you'll need to download it separately.

### Download the Dataset

1. Obtain the Kaggle dataset (`new-bangladeshi-crop-disease.zip`) from Kaggle Dataset.
2. Place the zip file in the project root directory (e.g., `~/Documents/Git/SDS-CP028-smart-leaf`).

### Process the Dataset

1. **Extract the Dataset**:
    ```bash
    python extract.py
    ```
    This extracts the contents of `new-bangladeshi-crop-disease.zip` into the `dataset/` folder.

2. **Verify Images**:
    ```bash
    python check_images.py
    ```
    This scans the `dataset/` folder for corrupt images and removes them.

3. **Analyze Class Distribution**:
    ```bash
    python analyze_classes.py
    ```
    This generates `class_distribution.png`, visualizing the number of images per class.

4. **Split the Dataset**:
    - Remove non-class folders if present (e.g., `rm -r dataset/Invalid` or `rm dataset/Info.txt`).
    - Run from the project root:
      ```bash
      python submissions/team-members/yan-cotta/data_preprocessing.py
      ```
    This splits the dataset into `split_dataset/train` (70%), `split_dataset/val` (15%), and `split_dataset/test` (15%).

5. **Evaluate Model Performance**:
    ```bash
    python model_evaluation.py
    ```
    This script performs:
    - 5-fold cross-validation
    - Computation of precision, recall, F1-score, and AUC-ROC metrics
    - Class imbalance analysis
    - Generation of confusion matrices and performance visualizations
    
    Outputs:
    - `confusion_matrix_fold_*.png`: Confusion matrix for each fold
    - `class_metrics.png`: Bar plot of precision, recall, and F1-score per class
    - `class_performance_metrics.csv`: Detailed metrics for each class

### Prerequisites

Install the required packages:
```bash
pip install -r requirements.txt
```

The requirements include:
- scikit-learn: For model evaluation metrics and cross-validation
- torch & torchvision: For the deep learning model
- numpy & pandas: For data handling and analysis
- matplotlib & seaborn: For visualization
- imbalanced-learn: For handling class imbalance
