"""
Model evaluation script for the Smart Leaf Disease Classification project.
Performs k-fold cross-validation and generates detailed performance metrics.
"""

import os
import sys
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.models import resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import get_transforms, compute_class_weights

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / "outputs" / "evaluation.log"),
        logging.StreamHandler()
    ]
)

# Constants
RANDOM_SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_FOLDS = 5
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_datasets(data_dir: Path) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Load datasets with different transforms for training and validation."""
    transforms = get_transforms(img_size=IMG_SIZE)
    train_dataset = datasets.ImageFolder(data_dir, transform=transforms['train'])
    val_dataset = datasets.ImageFolder(data_dir, transform=transforms['val'])
    logging.info(f"Loaded datasets from {data_dir} with {len(train_dataset)} samples")
    return train_dataset, val_dataset

def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_fold(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluate model on a single fold."""
    model.eval()
    predictions = []
    true_labels = []
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            try:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            except RuntimeError as e:
                logging.error(f"Error during evaluation: {str(e)}")
                continue
    avg_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    return avg_loss, predictions, true_labels

class LeafDiseaseResNet(nn.Module):
    """ResNet18 model for leaf disease classification."""
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        for name, param in self.model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

def plot_confusion_matrix(cm, classes, fold, save_dir):
    """Plot confusion matrix for a fold."""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(save_dir / f'confusion_matrix_fold_{fold}.png')
    plt.close()

class EarlyStopping:
    """Early stopping to prevent overfitting and save best model."""
    def __init__(self, patience: int = 3, verbose: bool = True, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def compute_fold_metrics(predictions, true_labels, class_names):
    """Compute metrics for a single fold."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    roc_auc = []
    for class_idx in range(len(class_names)):
        true_binary = (true_labels == class_idx).astype(int)
        pred_binary = (predictions == class_idx).astype(int)
        try:
            roc_auc.append(roc_auc_score(true_binary, pred_binary))
        except:
            roc_auc.append(np.nan)
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'roc_auc': roc_auc,
        'class_names': class_names
    }

def save_metrics(fold_metrics, file_path):
    """Save metrics to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(fold_metrics, f, indent=4)

def plot_aggregated_metrics(fold_metrics, output_dir):
    """Plot aggregated metrics across folds."""
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        if key not in ['fold', 'class_names']:
            avg_metrics[key] = np.mean([fm[key] for fm in fold_metrics], axis=0)
    
    metrics_df = pd.DataFrame(avg_metrics)
    metrics_df['Class'] = fold_metrics[0]['class_names']
    
    plt.figure(figsize=(15, 8))
    metrics_df.set_index('Class')[['precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar')
    plt.title('Average Performance Metrics by Class Across Folds')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_class_metrics.png')
    plt.close()

def main():
    """Run model evaluation with k-fold cross validation."""
    try:
        device = torch.device('cpu')
        logging.info(f"Using device: {device}")
        
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "dataset_organized"
        train_dataset, val_dataset = load_datasets(data_dir)
        n_classes = len(train_dataset.classes)
        
        class_weights = compute_class_weights(train_dataset).to(device)
        logging.info(f"Class weights: {class_weights}")
        
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        labels = torch.tensor([label for _, label in train_dataset])
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            logging.info(f"\nProcessing fold {fold + 1}/{NUM_FOLDS}")
            
            # Create data loaders without SMOTE
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=SubsetRandomSampler(train_idx),
                num_workers=NUM_WORKERS
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                sampler=SubsetRandomSampler(val_idx),
                num_workers=NUM_WORKERS
            )
            
            # Initialize ResNet18 model
            model = LeafDiseaseResNet(num_classes=n_classes).to(device)
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            
            early_stopping = EarlyStopping(patience=3, verbose=True)
            best_model_state = None
            best_val_loss = float('inf')
            
            for epoch in range(NUM_EPOCHS):
                train_loss = train_model(model, train_loader, criterion, optimizer, device)
                val_loss, val_predictions, val_labels = evaluate_fold(model, val_loader, criterion, device)
                logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Manually log learning rate changes
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    logging.info(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                
                if early_stopping(val_loss):
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            val_loss, predictions, true_labels = evaluate_fold(model, val_loader, criterion, device)
            metrics = compute_fold_metrics(predictions, true_labels, train_dataset.classes)
            metrics['fold'] = fold + 1
            metrics['val_loss'] = val_loss
            metrics['class_names'] = train_dataset.classes
            fold_metrics.append(metrics)
            
            cm = confusion_matrix(true_labels, predictions)
            plot_confusion_matrix(cm, train_dataset.classes, fold + 1, script_dir / "outputs")
        
        save_metrics(fold_metrics, script_dir / "outputs" / "evaluation_results.json")
        plot_aggregated_metrics(fold_metrics, script_dir / "outputs")
        logging.info("\nEvaluation completed successfully!")
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    main()