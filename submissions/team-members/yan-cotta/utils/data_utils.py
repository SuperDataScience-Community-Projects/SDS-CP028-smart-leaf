import os
import shutil
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def get_transforms(img_size: Tuple[int, int] = (224, 224)) -> Dict[str, transforms.Compose]:
    """Get data transforms for training and validation/testing.
    
    Args:
        img_size: Tuple of (height, width) for resizing images
    
    Returns:
        Dictionary containing train and val transforms
    """
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    total_samples = len(targets)
    return total_samples / (len(class_counts) * class_counts.float())
