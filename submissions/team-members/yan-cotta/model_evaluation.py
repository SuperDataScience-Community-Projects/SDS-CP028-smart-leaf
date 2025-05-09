import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data(data_dir):
    """Load the dataset and prepare for k-fold cross-validation"""
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset

def compute_class_weights(dataset):
    """Compute class weights to handle class imbalance"""
    targets = torch.tensor([label for _, label in dataset])
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    return class_weights

def evaluate_fold(model, dataloader, criterion, device):
    """Evaluate model on a single fold"""
    model.eval()
    predictions = []
    true_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels), running_loss / len(dataloader)

def plot_confusion_matrix(cm, classes, fold):
    """Plot confusion matrix for a fold"""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(f'confusion_matrix_fold_{fold}.png')
    plt.close()

def plot_class_metrics(metrics_df):
    """Plot precision, recall, and F1-score for each class"""
    plt.figure(figsize=(15, 8))
    metrics_df.plot(kind='bar')
    plt.title('Performance Metrics by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_metrics.png')
    plt.close()

def identify_problematic_classes(metrics_df, threshold=0.7):
    """Identify classes with poor performance"""
    poor_performance = metrics_df[metrics_df['F1-score'] < threshold]
    return poor_performance

def main():
    # Initialize variables for cross-validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_data('path_to_your_dataset')  # Update with your dataset path
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # Prepare arrays to store metrics
    fold_metrics = []
    class_names = dataset.classes
    n_classes = len(class_names)
    
    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(dataset)
    
    labels = [label for _, label in dataset]
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
        
        # Initialize model and training components
        model = YourModel()  # Replace with your model architecture
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Evaluate the model
        predictions, true_labels, val_loss = evaluate_fold(model, val_loader, criterion, device)
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)
        
        # Compute ROC-AUC score for each class (one-vs-rest)
        roc_auc = []
        for class_idx in range(n_classes):
            true_binary = (true_labels == class_idx).astype(int)
            pred_binary = (predictions == class_idx).astype(int)
            try:
                roc_auc.append(roc_auc_score(true_binary, pred_binary))
            except:
                roc_auc.append(np.nan)
        
        # Store metrics for this fold
        fold_metrics.append({
            'fold': fold + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
        
        # Plot confusion matrix for this fold
        cm = confusion_matrix(true_labels, predictions)
        plot_confusion_matrix(cm, class_names, fold + 1)
    
    # Aggregate metrics across folds
    avg_precision = np.mean([m['precision'] for m in fold_metrics], axis=0)
    avg_recall = np.mean([m['recall'] for m in fold_metrics], axis=0)
    avg_f1 = np.mean([m['f1'] for m in fold_metrics], axis=0)
    avg_roc_auc = np.mean([m['roc_auc'] for m in fold_metrics], axis=0)
    
    # Create DataFrame with class-wise metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1-score': avg_f1,
        'ROC-AUC': avg_roc_auc
    })
    
    # Plot class-wise metrics
    plot_class_metrics(metrics_df.set_index('Class'))
    
    # Identify problematic classes
    poor_performance = identify_problematic_classes(metrics_df)
    if not poor_performance.empty:
        print("\nClasses requiring attention (F1-score < 0.7):")
        print(poor_performance.to_string())
    
    # Save metrics to CSV
    metrics_df.to_csv('class_performance_metrics.csv', index=False)
    
    # Print overall metrics
    print("\nOverall Performance Metrics:")
    print(f"Average Precision: {avg_precision.mean():.3f}")
    print(f"Average Recall: {avg_recall.mean():.3f}")
    print(f"Average F1-score: {avg_f1.mean():.3f}")
    print(f"Average ROC-AUC: {avg_roc_auc.mean():.3f}")

if __name__ == '__main__':
    main()
