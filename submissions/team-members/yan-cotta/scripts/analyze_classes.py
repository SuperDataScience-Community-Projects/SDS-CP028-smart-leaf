"""
Analyze and visualize class distribution in the dataset.
This script counts images per class and generates a distribution plot.
"""

import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from utils.data_utils import plot_class_distribution, verify_image_corruption

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_images_per_class(directory: str) -> dict:
    """
    Count the number of valid images in each class directory.
    
    Args:
        directory: Path to the dataset root directory
        
    Returns:
        Dictionary mapping class names to image counts
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    class_counts = {}
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
        
    for crop_folder in directory.iterdir():
        if crop_folder.is_dir():
            for disease_class in crop_folder.iterdir():
                if disease_class.is_dir():
                    valid_images = [
                        f for f in disease_class.iterdir()
                        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'} 
                        and verify_image_corruption(str(f))
                    ]
                    class_counts[disease_class.name] = len(valid_images)
                    logging.info(f"Found {len(valid_images)} valid images in {disease_class.name}")
                    
    return class_counts

def main():
    """Main execution function."""
    dataset_dir = Path("dataset/BangladeshiCrops/BangladeshiCrops/Crop___Disease")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Expected counts from original dataset
    expected_counts = {
        "Corn___Common_Rust": 1192, "Corn___Gray_Leaf_Spot": 513,
        "Corn___Healthy": 1162, "Corn___Northern_Leaf_Blight": 985,
        "Potato___Early_Blight": 1000, "Potato___Healthy": 152,
        "Potato___Late_Blight": 1000, "Rice___Brown_Spot": 613,
        "Rice___Healthy": 1488, "Rice___Leaf_Blast": 977,
        "Rice___Neck_Blast": 1000, "Wheat___Brown_Rust": 902,
        "Wheat___Healthy": 1116, "Wheat___Yellow_Rust": 924
    }
    
    try:
        logging.info(f"Starting analysis of dataset in {dataset_dir}")
        class_counts = count_images_per_class(dataset_dir)
        
        # Log results and compare with expected counts
        logging.info("\nClass distribution analysis:")
        for class_name, count in class_counts.items():
            expected = expected_counts.get(class_name, 0)
            diff = count - expected
            status = "✓" if diff == 0 else "!"
            logging.info(f"{status} {class_name}: {count} images (Expected: {expected}, Diff: {diff})")
            
        # Generate visualization
        plot_path = output_dir / "class_distribution.png"
        plot_class_distribution(class_counts, str(plot_path))
        logging.info(f"\nVisualization saved to {plot_path}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise
        
if __name__ == "__main__":
    main()
