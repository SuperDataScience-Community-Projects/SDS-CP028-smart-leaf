# Smart Leaf Disease Classification

This project implements a deep learning solution for classifying crop diseases from leaf images, with a focus on crops common in Bangladesh.

## Project Structure

```plaintext
.
├── utils/
│   └── data_utils.py      # Utility functions for data processing and visualization
├── scripts/
│   ├── extract.py         # Dataset extraction
│   ├── check_images.py    # Image validation
│   ├── analyze_classes.py # Class distribution analysis
│   ├── data_preprocessing.py  # Dataset splitting and preprocessing
│   └── model_evaluation.py   # Model evaluation and metrics
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features

- Robust data preprocessing pipeline
- Comprehensive data validation and cleaning
- Class imbalance analysis and handling
- Modular code structure with reusable utilities
- Standardized data transforms for consistency
- Detailed model evaluation metrics

## Utilities (`utils/data_utils.py`)

- **Data Transforms**: Standardized image transformations for training and validation
- **Image Validation**: Functions to verify image integrity
- **Class Distribution**: Tools for analyzing and visualizing class distribution
- **Class Weights**: Computation of class weights for handling imbalance

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/smart-leaf.git
   cd smart-leaf
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease)
2. Place `new-bangladeshi-crop-disease.zip` in the project root

## Usage

1. **Extract Dataset**:

   ```bash
   python extract.py
   ```

2. **Validate Images**:

   ```bash
   python check_images.py
   ```

3. **Analyze Class Distribution**:

   ```bash
   python analyze_classes.py
   ```

4. **Preprocess Dataset**:

   ```bash
   python data_preprocessing.py
   ```

5. **Evaluate Model**:

   ```bash
   python model_evaluation.py
   ```

## Output Files

- `class_distribution.png`: Visualization of class distribution
- `confusion_matrix_fold_*.png`: Confusion matrices for each CV fold
- `class_metrics.png`: Performance metrics by class
- `class_performance_metrics.csv`: Detailed performance metrics

## Dependencies

- PyTorch & torchvision
- scikit-learn
- numpy & pandas
- matplotlib & seaborn
- Pillow
- imbalanced-learn

## Notes

- The dataset is gitignored to manage repository size
- Class imbalances are handled through weighted sampling
- All images are validated before processing
- Data augmentation is applied consistently across training
