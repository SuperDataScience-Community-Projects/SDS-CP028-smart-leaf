# scripts/compare_metrics.py
import json
import numpy as np
from pathlib import Path

def compare_metrics(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        m1, m2 = json.load(f1), json.load(f2)
    avg_f1_1 = np.mean([np.mean(m['f1']) for m in m1])
    avg_f1_2 = np.mean([np.mean(m['f1']) for m in m2])
    print(f"Baseline F1: {avg_f1_1:.4f}, Oversampling F1: {avg_f1_2:.4f}")
    print("\nClass-wise F1-scores:")
    for i, cls in enumerate(m1[0]['class_names']):
        f1_1 = np.mean([m['f1'][i] for m in m1])
        f1_2 = np.mean([m['f1'][i] for m in m2])
        print(f"{cls}: Baseline = {f1_1:.4f}, Oversampling = {f1_2:.4f}, Change = {f1_2 - f1_1:.4f}")
    print(f"\nBaseline Validation Loss: {np.mean([m['val_loss'] for m in m1]):.4f}")
    print(f"Oversampling Validation Loss: {np.mean([m['val_loss'] for m in m2]):.4f}")

compare_metrics(
    'scripts/outputs/evaluation_results_smote.json',
    'scripts/outputs/evaluation_results_tuned.json'
)