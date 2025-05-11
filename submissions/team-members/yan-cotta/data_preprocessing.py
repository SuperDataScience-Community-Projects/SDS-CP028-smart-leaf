import os
import splitfolders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms (example transforms, adjust as needed)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Split the dataset into train (70%), validation (15%), and test (15%)
splitfolders.ratio(
    "/home/yan/Documents/Git/SDS-CP028-smart-leaf/temp_dataset",
    output="split_dataset",
    seed=42,
    ratio=(0.7, 0.15, 0.15)  # Updated to 3-way split
)

# Load the datasets
train_dataset = datasets.ImageFolder("split_dataset/train", transform=train_transforms)
val_dataset = datasets.ImageFolder("split_dataset/val", transform=val_transforms)
test_dataset = datasets.ImageFolder("split_dataset/test", transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Print dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")