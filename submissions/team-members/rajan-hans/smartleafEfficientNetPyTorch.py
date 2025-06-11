import os
import numpy as np
import kagglehub
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import TrivialAugmentWide
import seaborn as sns
import matplotlib.pyplot as plt

class CropDiseaseDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        classes = []
        for main in os.listdir(self.root_dir):
            main_path = os.path.join(self.root_dir, main)
            if not os.path.isdir(main_path):
                continue
            for sub in os.listdir(main_path):
                sub_path = os.path.join(main_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                name = f"{main}_{sub.split('_')[-2]}_{sub.split('_')[-1]}"
                classes.append(name)
        classes = sorted(set(classes))
        return classes, {c: i for i, c in enumerate(classes)}

    def _make_dataset(self):
        samples = []
        for main in os.listdir(self.root_dir):
            main_path = os.path.join(self.root_dir, main)
            if not os.path.isdir(main_path):
                continue
            for sub in os.listdir(main_path):
                sub_path = os.path.join(main_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                label = self.class_to_idx[f"{main}_{sub.split('_')[-2]}_{sub.split('_')[-1]}"]
                for fn in os.listdir(sub_path):
                    fp = os.path.join(sub_path, fn)
                    if os.path.isfile(fp):
                        samples.append((fp, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, lbl = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        return img, lbl

class TransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    download_path = kagglehub.dataset_download("nafishamoin/new-bangladeshi-crop-disease")
    dataset_path = os.path.join(download_path, "BangladeshiCrops", "BangladeshiCrops", "Crop___Disease")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    full_ds = CropDiseaseDataset(dataset_path)
    labels = [lbl for _, lbl in full_ds.samples]
    num_classes = len(full_ds.classes)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_val_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

    subset_labels = [labels[i] for i in train_val_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_sub_idx, val_sub_idx = next(sss2.split(np.zeros(len(subset_labels)), subset_labels))

    train_idx = [train_val_idx[i] for i in train_sub_idx]
    val_idx   = [train_val_idx[i] for i in val_sub_idx]

    train_ds = TransformWrapper(Subset(full_ds, train_idx), train_transform)
    val_ds   = TransformWrapper(Subset(full_ds, val_idx),   val_transform)
    test_ds  = TransformWrapper(Subset(full_ds, test_idx),  val_transform)

    data_kwargs = {'batch_size': 64, 'num_workers': 4, 'pin_memory': use_amp}
    train_loader = DataLoader(train_ds, shuffle=True,  **data_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **data_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **data_kwargs)

    train_labels = [labels[i] for i in train_idx]
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(num_classes), y=train_labels),
        dtype=torch.float
    ).to(device)

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    epochs, patience = 30, 5
    best_f1, wait = 0.0, 0
    history = {key: [] for key in [
        'train_loss','val_loss','train_acc','val_acc',
        'train_prec','val_prec','train_rec','val_rec','train_f1','val_f1']}

    for epoch in range(1, epochs+1):
        model.train()
        running = {'loss':0,'preds':[],'targs':[]}
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running['loss'] += loss.item()*x.size(0)
            running['preds'].extend(out.argmax(1).cpu().numpy())
            running['targs'].extend(y.cpu().numpy())

        tloss = running['loss']/len(train_ds)
        tacc  = accuracy_score(running['targs'], running['preds'])
        tprec = precision_score(running['targs'], running['preds'], average='macro')
        trec  = recall_score(running['targs'], running['preds'], average='macro')
        tf1   = f1_score(running['targs'], running['preds'], average='macro')

        model.eval()
        running = {'loss':0,'preds':[],'targs':[]}
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                running['loss'] += loss.item()*x.size(0)
                running['preds'].extend(out.argmax(1).cpu().numpy())
                running['targs'].extend(y.cpu().numpy())

        vloss = running['loss']/len(val_ds)
        vacc  = accuracy_score(running['targs'], running['preds'])
        vprec = precision_score(running['targs'], running['preds'], average='macro')
        vrec  = recall_score(running['targs'], running['preds'], average='macro')
        vf1   = f1_score(running['targs'], running['preds'], average='macro')

        scheduler.step(vf1)

        for k,v in zip(history.keys(), [tloss,vloss,tacc,vacc,tprec,vprec,trec,vrec,tf1,vf1]):
            history[k].append(v)

        print(f"Epoch {epoch}: Train Loss={tloss:.4f}, Acc={tacc:.4f}, F1={tf1:.4f} | Val Loss={vloss:.4f}, Acc={vacc:.4f}, F1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Final Test"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_targets.extend(y.cpu().numpy())

    print("Test Accuracy:   ", accuracy_score(test_targets, test_preds))
    print("Test Recall (m): ", recall_score(test_targets, test_preds, average='macro'))
    print("Test F1 (m):     ", f1_score(test_targets, test_preds, average='macro'))
    print("\nClassification Report:\n",
          classification_report(test_targets, test_preds, target_names=full_ds.classes))

    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=full_ds.classes, yticklabels=full_ds.classes)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Test Confusion Matrix")
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
