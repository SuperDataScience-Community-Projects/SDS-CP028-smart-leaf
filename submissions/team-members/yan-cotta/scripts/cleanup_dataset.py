"""
Dataset cleanup script for the Smart Leaf Disease Classification project.
Removes empty directories and fixes dataset structure.
"""

import os
import logging
from pathlib import Path
from typing import Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cleanup_split_dataset(base_dir: Path) -> Dict[str, int]:
    """
    Remove empty directories and fix the dataset structure recursively.
    
    Args:
        base_dir: Path to the base dataset directory
        
    Returns:
        Dictionary with counts of removed directories and errors
        
    Raises:
        FileNotFoundError: If base directory doesn't exist
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")
    
    stats = {'removed': 0, 'errors': 0}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            logging.warning(f"Split directory not found: {split_dir}")
            continue
            
        logging.info(f"Cleaning up {split} directory...")
        
        # Walk bottom-up through directory tree
        for path in reversed(list(split_dir.rglob('*'))):
            if path.is_dir():
                try:
                    # Check if directory is empty (no files or subdirectories)
                    if not any(path.iterdir()):
                        path.rmdir()
                        stats['removed'] += 1
                        logging.info(f"Removed empty directory: {path}")
                except OSError as e:
                    stats['errors'] += 1
                    logging.error(f"Error removing directory {path}: {e}")
                    
    return stats

def main():
    """Main execution function."""
    try:
        base_dir = Path("split_dataset")
        logging.info(f"Starting dataset cleanup in {base_dir}")
        
        stats = cleanup_split_dataset(base_dir)
        
        logging.info(
            f"\nCleanup completed!\n"
            f"Directories removed: {stats['removed']}\n"
            f"Errors encountered: {stats['errors']}"
        )
        
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
        raise

if __name__ == "__main__":
    main()
