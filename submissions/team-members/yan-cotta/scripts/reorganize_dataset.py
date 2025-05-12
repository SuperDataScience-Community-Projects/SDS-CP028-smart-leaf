"""
Dataset reorganization script for the Smart Leaf Disease Classification project.
Flattens the hierarchical dataset structure and validates images during transfer.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict
from utils.data_utils import verify_image_corruption

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories() -> tuple[Path, Path]:
    """Set up source and target directories for dataset reorganization.
    
    Returns:
        Tuple of (source_dir, target_dir) paths
        
    Raises:
        FileNotFoundError: If source directory doesn't exist
    """
    source_dir = Path("dataset/BangladeshiCrops/BangladeshiCrops/Crop___Disease").resolve()
    target_dir = Path("dataset_organized").resolve()
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Source directory: {source_dir}")
    logging.info(f"Target directory: {target_dir}")
    
    return source_dir, target_dir
    
def copy_and_validate_images(source_path: Path, target_path: Path) -> Dict[str, int]:
    """Copy images from source to target directory with validation.
    
    Args:
        source_path: Path to source directory containing images
        target_path: Path to target directory for copied images
        
    Returns:
        Dictionary with counts of copied and skipped images
    """
    stats = {'copied': 0, 'skipped': 0, 'invalid': 0}
    target_path.mkdir(parents=True, exist_ok=True)
    
    for img_file in source_path.iterdir():
        if img_file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
            if verify_image_corruption(str(img_file)):
                target_file = target_path / img_file.name
                # Skip if file already exists and is valid
                if target_file.exists() and verify_image_corruption(str(target_file)):
                    stats['skipped'] += 1
                    continue
                    
                shutil.copy2(img_file, target_file)
                stats['copied'] += 1
            else:
                stats['invalid'] += 1
                logging.warning(f"Skipping corrupt image: {img_file}")
    
    return stats

def reorganize_dataset() -> Optional[Path]:
    """Reorganize the dataset by flattening the directory structure.
    
    Returns:
        Path to reorganized dataset directory or None if error occurs
    """
    try:
        source_dir, target_dir = setup_directories()
        total_stats = {'copied': 0, 'skipped': 0, 'invalid': 0}
        
        # Process each crop type directory
        for crop_dir in source_dir.iterdir():
            if crop_dir.is_dir():
                # Process each disease class directory
                for disease_dir in crop_dir.iterdir():
                    if disease_dir.is_dir():
                        target_class_dir = target_dir / disease_dir.name
                        logging.info(f"Processing class: {disease_dir.name}")
                        
                        stats = copy_and_validate_images(disease_dir, target_class_dir)
                        for key in total_stats:
                            total_stats[key] += stats[key]
                            
                        logging.info(
                            f"Class {disease_dir.name} stats: "
                            f"copied={stats['copied']}, "
                            f"skipped={stats['skipped']}, "
                            f"invalid={stats['invalid']}"
                        )
        
        logging.info(
            f"\nDataset reorganization completed!\n"
            f"Total images copied: {total_stats['copied']}\n"
            f"Total images skipped: {total_stats['skipped']}\n"
            f"Total invalid images: {total_stats['invalid']}"
        )
        
        return target_dir
        
    except Exception as e:
        logging.error(f"Error during dataset reorganization: {str(e)}")
        return None

if __name__ == "__main__":
    reorganize_dataset()
