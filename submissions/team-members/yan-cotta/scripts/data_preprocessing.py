"""
Data preprocessing script for the Smart Leaf Disease Classification project.
Splits the dataset into train, validation, and test sets, and sets up data loaders.
"""

import os
import logging
from pathlib import Path
import splitfolders
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.data_utils import get_transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = min(os.cpu_count() or 1, 8)
RANDOM_SEED = 42
SPLIT_RATIO = (0.7, 0.15, 0.15)  # train, val, test

def setup_paths() -> tuple[Path, Path]:
    """Set up input and output paths for dataset processing."""
    project_dir = Path(__file__).parent.parent  # Move up to project root from scripts/
    input_path = project_dir / "dataset_organized"
    output_path = project_dir / "split_dataset"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
        
    output_path.mkdir(exist_ok=True)
    return input_path, output_path

def create_datasets(split_path: Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test datasets and dataloaders.
    
    Args:
        split_path: Path to the root directory containing train/val/test splits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    transforms = get_transforms(img_size=IMG_SIZE)
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        split_path / "train",
        transform=transforms['train']
    )
    val_dataset = datasets.ImageFolder(
        split_path / "val",
        transform=transforms['val']
    )
    test_dataset = datasets.ImageFolder(
        split_path / "test",
        transform=transforms['val']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    logging.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def main():
    """Main execution function."""
    try:
        # Setup paths
        input_path, output_path = setup_paths()
        
        # Split dataset
        logging.info(f"Splitting dataset from {input_path} into {output_path}")
        logging.info(f"Split ratio: {SPLIT_RATIO} (train, val, test)")
        
        splitfolders.ratio(
            str(input_path),
            output=str(output_path),
            seed=RANDOM_SEED,
            ratio=SPLIT_RATIO
        )
        
        # Create datasets and dataloaders
        train_loader, val_loader, test_loader = create_datasets(output_path)
        logging.info("Dataset preprocessing completed successfully")
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()