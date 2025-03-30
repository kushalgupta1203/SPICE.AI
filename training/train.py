import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Custom Dataset
class SolarPanelDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename = self.data.iloc[idx, 0]
        image_path = os.path.join(self.img_dir, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        label_values = self.data.iloc[idx, 1:].astype(float).values
        labels = torch.tensor(label_values, dtype=torch.float32)

        return image, labels

# Data Augmentation using Albumentations
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.75, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

valid_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Paths
train_csv = "D:/Projects/SPICE.AI/dataset/splitted/train/_classes.csv"
train_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/train"
val_csv = "D:/Projects/SPICE.AI/dataset/splitted/val/_classes.csv"
val_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/val"

# Load Datasets
train_dataset = SolarPanelDataset(train_csv, train_img_dir, train_transform)
valid_dataset = SolarPanelDataset(val_csv, val_img_dir, valid_transform)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)

# Define MobileNetV3-Large Model
def get_mobilenet():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(0.3),  # Added dropout to prevent overfitting
        nn.Linear(num_ftrs, 8)  # 8 output classes
    )
    return model

# Training Function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_mobilenet().to(device)

    # Compute Class Weights
    class_counts = train_dataset.data.iloc[:, 1:].sum(axis=0).values
    pos_weight = torch.tensor(1.0 / (class_counts / (class_counts.sum() + 1e-6)), dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 50
    save_path = "D:/Projects/SPICE.AI/models"
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Accuracy Calculation
            predicted_train = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.numel()

            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100

        # Validation Phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images_val, labels_val in valid_loader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                outputs_val = model(images_val)
                loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item()
                
                predicted_val = (torch.sigmoid(outputs_val) > 0.5).float()
                correct_val += (predicted_val == labels_val).sum().item()
                total_val += labels_val.numel()

        avg_val_loss = val_loss / len(valid_loader)
        val_accuracy = correct_val / total_val * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        model_save_path = os.path.join(save_path, f"spice_ai_mobilenet_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model Saved: {model_save_path}")

    print("Training Complete")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_model()