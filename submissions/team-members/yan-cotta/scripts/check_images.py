"""
Image validation script for the Smart Leaf Disease Classification project.
Checks for and optionally removes corrupt image files in the dataset.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from utils.data_utils import verify_image_corruption

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_corrupt_images(directory: Path) -> List[Path]:
    """
    Find corrupt image files in a directory and its subdirectories.
    
    Args:
        directory: Path to the directory to check
        
    Returns:
        List of paths to corrupt image files
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
        
    corrupt_files = []
    total_files = 0
    
    for img_path in directory.rglob('*'):
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
            total_files += 1
            if not verify_image_corruption(str(img_path)):
                corrupt_files.append(img_path)
                logging.warning(f"Corrupt image found: {img_path}")
                
            if total_files % 100 == 0:  # Progress update every 100 files
                logging.info(f"Processed {total_files} files...")
                
    return corrupt_files

def remove_corrupt_files(files: List[Path]) -> None:
    """Remove corrupt files and log the action."""
    for file in files:
        try:
            file.unlink()
            logging.info(f"Removed corrupt file: {file}")
        except Exception as e:
            logging.error(f"Failed to remove {file}: {str(e)}")

def main():
    """Main execution function."""
    try:
        dataset_dir = Path("dataset")
        
        logging.info(f"Starting image validation in {dataset_dir}")
        corrupt_files = find_corrupt_images(dataset_dir)
        
        if corrupt_files:
            logging.warning(f"Found {len(corrupt_files)} corrupt files")
            user_input = input("Do you want to remove corrupt files? [y/N]: ")
            
            if user_input.lower() == 'y':
                remove_corrupt_files(corrupt_files)
                logging.info("Corrupt files removed successfully")
            else:
                logging.info("No files were removed")
        else:
            logging.info("No corrupt files found")
            
    except Exception as e:
        logging.error(f"Error during image validation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
