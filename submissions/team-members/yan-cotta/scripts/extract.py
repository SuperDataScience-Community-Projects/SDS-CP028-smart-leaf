"""
Dataset extraction script for the Smart Leaf Disease Classification project.
Extracts the dataset zip file into the appropriate directory structure.
"""

import zipfile
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_dataset(zip_path: Path, extract_dir: Path) -> None:
    """
    Extract the dataset zip file to the specified directory.
    
    Args:
        zip_path: Path to the dataset zip file
        extract_dir: Path to extract the dataset to
        
    Raises:
        FileNotFoundError: If zip file doesn't exist
        zipfile.BadZipFile: If zip file is corrupted
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip file not found: {zip_path}")
        
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Extracting {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    logging.info("Dataset extraction completed successfully")

def main():
    """Main execution function."""
    try:
        zip_path = Path("new-bangladeshi-crop-disease.zip")
        extract_dir = Path("dataset")
        extract_dataset(zip_path, extract_dir)
        
    except Exception as e:
        logging.error(f"Error during dataset extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main()

