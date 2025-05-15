"""
Utility functions for data processing and visualization in the Smart Leaf project.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import pandas as pd

def get_transforms(img_size: Tuple[int, int] = (224, 224)) -> Dict[str, transforms.Compose]:
    """Get data transforms for training and validation/testing.
    
    Args:
        img_size: Tuple of (height, width) for resizing images
    
    Returns:
        Dictionary containing train and val transforms
    """
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),  # Increased rotation angle
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added hue adjustment
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)  # Added random erasing
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transforms, 'val': val_transforms}

def verify_image_corruption(image_path: str) -> bool:
    """Verify if an image file is corrupted.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if image is valid, False if corrupted
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def plot_class_distribution(class_counts: Dict[str, int], output_path: str = 'class_distribution.png'):
    """Plot and save class distribution visualization.
    
    Args:
        class_counts: Dictionary of class names and their counts
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 8))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compute_class_weights(dataset) -> torch.Tensor:
    """Compute class weights for imbalanced dataset.
    
    Args:
        dataset: PyTorch dataset with targets attribute
        
    Returns:
        Tensor of class weights
    """
    targets = torch.tensor([label for _, label in dataset])
    class_counts = torch.bincount(targets)
    total_samples = len(dataset)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts.float())
    return weights

def compute_fold_metrics(predictions: np.ndarray, true_labels: np.ndarray, class_names: List[str]) -> Dict:
    """Compute performance metrics for a single fold.
    
    Args:
        predictions: Array of predicted labels
        true_labels: Array of true labels
        class_names: List of class names
        
    Returns:
        Dictionary containing precision, recall, f1, and ROC-AUC scores
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )
    
    # Compute ROC-AUC for each class (one-vs-rest)
    roc_auc = []
    for class_idx in range(len(class_names)):
        try:
            roc_auc.append(roc_auc_score(
                (true_labels == class_idx).astype(int),
                (predictions == class_idx).astype(int)
            ))
        except:
            roc_auc.append(np.nan)
    
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'roc_auc': roc_auc
    }

def save_metrics(metrics: List[Dict], output_path: Path) -> None:
    """Save evaluation metrics to JSON file.
    
    Args:
        metrics: List of metric dictionaries from each fold
        output_path: Path to save the metrics file
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_aggregated_metrics(metrics: List[Dict], output_dir: Path) -> None:
    """Plot aggregated metrics across all folds.
    
    Args:
        metrics: List of metric dictionaries from each fold
        output_dir: Directory to save the plots
    """
    # Convert metrics to DataFrame for easier plotting
    fold_data = pd.DataFrame(metrics)
    
    # Plot metrics over folds
    plt.figure(figsize=(10, 6))
    plt.plot(fold_data['fold'], fold_data['val_loss'], marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Across Folds')
    plt.savefig(output_dir / 'loss_across_folds.png')
    plt.close()
    
    # Plot average metrics by class
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in metrics], axis=0),
        'recall': np.mean([m['recall'] for m in metrics], axis=0),
        'f1': np.mean([m['f1'] for m in metrics], axis=0),
        'roc_auc': np.mean([m['roc_auc'] for m in metrics], axis=0)
    }
    
    metrics_df = pd.DataFrame(avg_metrics)
    metrics_df.plot(kind='bar', figsize=(15, 6))
    plt.title('Average Performance Metrics by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_class_metrics.png')
    plt.close()
