"""
Verification script to test all dependencies and file paths
for the Smart Leaf Disease Classification project.
"""

import os
import sys
import logging
from pathlib import Path
import importlib
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_dependencies() -> Dict[str, bool]:
    """Check if all required packages are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'PIL',
        'imblearn',
        'splitfolders'
    ]
    
    results = {}
    for package in required_packages:
        try:
            importlib.import_module(package)
            results[package] = True
            logging.info(f"✓ {package} is installed")
        except ImportError:
            results[package] = False
            logging.error(f"✗ {package} is missing")
    return results

def verify_directory_structure() -> Dict[str, bool]:
    """Verify all required directories exist."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    required_paths = {
        'dataset': project_dir / 'dataset',
        'dataset_organized': project_dir / 'dataset_organized',
        'split_dataset': project_dir / 'split_dataset',
        'utils': project_dir / 'utils',
        'scripts': project_dir / 'scripts'
    }
    
    results = {}
    for name, path in required_paths.items():
        exists = path.exists()
        results[name] = exists
        if exists:
            logging.info(f"✓ {name} directory found at {path}")
        else:
            logging.warning(f"✗ {name} directory missing at {path}")
    return results

def verify_script_imports() -> Dict[str, bool]:
    """Verify all scripts can be imported without errors."""
    script_dir = Path(__file__).parent
    scripts = [
        'analyze_classes.py',
        'check_images.py',
        'cleanup_dataset.py',
        'data_preprocessing.py',
        'extract.py',
        'model_evaluation.py',
        'reorganize_dataset.py'
    ]
    
    results = {}
    sys.path.insert(0, str(script_dir.parent))
    
    for script in scripts:
        script_path = script_dir / script
        if not script_path.exists():
            logging.error(f"✗ Script not found: {script}")
            results[script] = False
            continue
            
        try:
            # Try to compile the script to check for syntax errors
            with open(script_path, 'r') as f:
                compile(f.read(), script_path, 'exec')
            logging.info(f"✓ {script} syntax is valid")
            results[script] = True
        except Exception as e:
            logging.error(f"✗ Error in {script}: {str(e)}")
            results[script] = False
    
    return results

def verify_utils_module() -> Dict[str, bool]:
    """Verify utils module and its functions."""
    try:
        from utils import data_utils
        functions = [
            'get_transforms',
            'verify_image_corruption',
            'plot_class_distribution',
            'compute_class_weights'
        ]
        
        results = {}
        for func in functions:
            has_func = hasattr(data_utils, func)
            results[func] = has_func
            if has_func:
                logging.info(f"✓ utils.data_utils.{func} is available")
            else:
                logging.error(f"✗ utils.data_utils.{func} is missing")
        return results
    except Exception as e:
        logging.error(f"✗ Error importing utils module: {str(e)}")
        return {'utils_module': False}

def main():
    """Run all verifications."""
    try:
        logging.info("Starting setup verification...")
        
        # Check dependencies
        deps_ok = all(check_dependencies().values())
        logging.info("\nDependency check complete")
        
        # Verify directory structure
        dirs_ok = all(verify_directory_structure().values())
        logging.info("\nDirectory structure check complete")
        
        # Verify scripts
        scripts_ok = all(verify_script_imports().values())
        logging.info("\nScript verification complete")
        
        # Verify utils module
        utils_ok = all(verify_utils_module().values())
        logging.info("\nUtils module verification complete")
        
        # Overall status
        if all([deps_ok, dirs_ok, scripts_ok, utils_ok]):
            logging.info("\n✅ All verifications passed successfully!")
        else:
            logging.warning("\n⚠️ Some verifications failed. Please check the logs above.")
            
    except Exception as e:
        logging.error(f"Error during verification: {str(e)}")
        raise

if __name__ == "__main__":
    main()
