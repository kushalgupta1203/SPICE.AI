import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)  

# Dataset Class
class SolarPanelDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])  
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.data.iloc[idx, 1:].astype(int).values, dtype=torch.float32)
        return image, labels

# Data Augmentation & Normalization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths for Training and Validation Data
train_csv = "D:/Projects/SPICE.AI/dataset/splitted/train/_classes.csv"
train_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/train"
val_csv = "D:/Projects/SPICE.AI/dataset/splitted/val/_classes.csv"
val_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/val"

# Load Datasets
train_dataset = SolarPanelDataset(train_csv, train_img_dir, train_transform)
valid_dataset = SolarPanelDataset(val_csv, val_img_dir, valid_transform)

# Reduce GPU Overload
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=True, pin_memory=True, persistent_workers=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=0, shuffle=False, pin_memory=True, persistent_workers=False)

# Define Model with Dropout
def get_mobilenet_v3():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),  
        nn.Linear(512, 9)  
    )
    return model

# Training Function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_mobilenet_v3().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    scaler = torch.amp.GradScaler()
    epochs = 25  
    best_val_loss = np.inf
    early_stop_count = 0
    patience = 5  

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        avg_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Learning Rate Adjustment
        scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "solar_panel_mobilenetv3_best.pth")  
            early_stop_count = 0  
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    print("Training Complete. Best Model Saved.")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_model()
