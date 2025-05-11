import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Download & point to the right directory
# (ensure you have configured Kaggle credentials in Colab)
#path = kagglehub.dataset_download("nafishamoin/new-bangladeshi-crop-disease")
#dataset_path = "/Users/sourinrakshit/.cache/kagglehub/datasets/nafishamoin/new-bangladeshi-crop-disease/versions/2/BangladeshiCrops/BangladeshiCrops/Crop___Disease"
#print("Dataset ready at:", dataset_path)


download_path = kagglehub.dataset_download("nafishamoin/new-bangladeshi-crop-disease")
dataset_path = os.path.join(
    download_path,
    "BangladeshiCrops",
    "BangladeshiCrops",
    "Crop___Disease"
)

# 2. Custom Dataset
class CropDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        cls = []
        for main_class in os.listdir(self.root_dir):
            main_path = os.path.join(self.root_dir, main_class)
            if not os.path.isdir(main_path):
                continue
            for subclass in os.listdir(main_path):
                sub_path = os.path.join(main_path, subclass)
                if not os.path.isdir(sub_path):
                    continue
                name = f"{main_class}_{subclass.split('_')[-2]}_{subclass.split('_')[-1]}"
                cls.append(name)
        cls = sorted(set(cls))
        return cls, {c: i for i, c in enumerate(cls)}

    def _make_dataset(self):
        samples = []
        for main_class in os.listdir(self.root_dir):
            main_path = os.path.join(self.root_dir, main_class)
            if not os.path.isdir(main_path):
                continue
            for subclass in os.listdir(main_path):
                sub_path = os.path.join(main_path, subclass)
                if not os.path.isdir(sub_path):
                    continue
                label = self.class_to_idx[f"{main_class}_{subclass.split('_')[-2]}_{subclass.split('_')[-1]}"]
                for fn in os.listdir(sub_path):
                    file_path = os.path.join(sub_path, fn)
                    if os.path.isfile(file_path):
                        samples.append((file_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn, label = self.samples[idx]
        img = Image.open(fn).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 3. Transforms
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1), shear=10),
    transforms.RandomPerspective(0.2, 0.3),
    transforms.ColorJitter(0.2,0.2,0.3,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# 4. Load full dataset & compute class weights
full_ds = CropDiseaseDataset(dataset_path, transform=None)
labels = [lbl for _, lbl in full_ds.samples]
num_classes = len(full_ds.classes)
cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
class_weights = torch.tensor(cw, dtype=torch.float)

print(f"Found {len(full_ds)} images across {num_classes} classes.")

# 5. Stratified train/val/test split
sss1 = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))

train_val_labels = [labels[i] for i in train_val_idx]
sss2 = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
train_sub_idx, val_sub_idx = next(sss2.split(np.zeros(len(train_val_labels)), train_val_labels))

train_idx = [train_val_idx[i] for i in train_sub_idx]
val_idx   = [train_val_idx[i] for i in val_sub_idx]

train_ds = Subset(full_ds, train_idx)
val_ds   = Subset(full_ds, val_idx)
test_ds  = Subset(full_ds, test_idx)

train_ds.dataset.transform = train_tf
val_ds.dataset.transform   = val_tf
test_ds.dataset.transform  = val_tf

# 6. DataLoaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dl_kwargs = {"batch_size": 32, "num_workers": 4, "pin_memory": True} if device.type=="cuda" else {"batch_size": 32}
train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

# 7. Model definition
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*28*28,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = BasicCNN(num_classes).to(device)

# 8. Loss, optimizer, early stopping setup
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
best_acc, patience, counter = 0.0, 5, 0

# 9. Training loop
for epoch in range(1, 11):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/10 [Train]"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/10 [Val]"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = accuracy_score(targets, preds) * 100
    print(f"Epoch {epoch}: Train loss {total_loss/len(train_loader):.4f}, Val acc {acc:.2f}%")

    if acc > best_acc:
        best_acc, counter = acc, 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"Training complete! Best val acc: {best_acc:.2f}%")

# 10. Final evaluation on test set
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing"):
        x = x.to(device)
        preds = model(x).argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute metrics
test_acc = accuracy_score(y_true, y_pred) * 100
macro_prec = precision_score(y_true, y_pred, average='macro')
weighted_prec = precision_score(y_true, y_pred, average='weighted')
macro_rec = recall_score(y_true, y_pred, average='macro')
weighted_rec = recall_score(y_true, y_pred, average='weighted')
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=full_ds.classes)

# Print metrics
print(f"\nTest Accuracy: {test_acc:.2f}%")
print(f"Macro Precision: {macro_prec:.4f}")
print(f"Weighted Precision: {weighted_prec:.4f}")
print(f"Macro Recall: {macro_rec:.4f}")
print(f"Weighted Recall: {weighted_rec:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print("\nClassification Report:\n", report)

# Confusion Matrix Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=full_ds.classes, yticklabels=full_ds.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()