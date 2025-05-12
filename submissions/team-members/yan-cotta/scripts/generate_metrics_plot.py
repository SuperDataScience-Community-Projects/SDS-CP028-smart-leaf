import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

with open('scripts/outputs/evaluation_results.json') as f:
    metrics = json.load(f)
plot_aggregated_metrics(metrics, Path('scripts/outputs'))