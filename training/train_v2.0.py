import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
from torchvision import transforms
from PIL import Image

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

        # Read the image as a NumPy array
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert NumPy array to PIL Image for transforms
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Extract labels from the CSV
        label_values = self.data.iloc[idx, 1:].astype(float).values
        labels = torch.tensor(label_values, dtype=torch.float32)

        return image, labels




# Data Augmentation using PyTorch Transforms
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

# Paths
train_csv = "D:/Projects/SPICE.AI/dataset/splitted/train/_classes.csv"
train_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/train"
val_csv = "D:/Projects/SPICE.AI/dataset/splitted/val/_classes.csv"
val_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/val"

# Load Datasets
try:
    train_dataset = SolarPanelDataset(train_csv, train_img_dir, train_transform)
    valid_dataset = SolarPanelDataset(val_csv, val_img_dir, valid_transform)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)

# Define MobileNetV3-Large Model
def get_mobilenet():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 8)  # 8 output classes
    return model

# Training Function
def train_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = get_mobilenet().to(device)

        # Loss Function and Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

        epochs = 50
        save_path = "D:/Projects/SPICE.AI/models"
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train, total_train = 0, 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for i, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Accuracy Calculation (Fixed for Multi-Label Classification)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_train += (predicted == labels).sum().item()
                total_train += labels.numel()

                progress_bar.set_postfix(loss=running_loss / (i + 1))

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train * 100

            # Validation Phase
            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    predicted_val = (torch.sigmoid(outputs) > 0.5).float()
                    correct_val += (predicted_val == labels).sum().item()
                    total_val += labels.numel()

            avg_val_loss = val_loss / len(valid_loader)
            val_accuracy = correct_val / total_val * 100

            # Status Update after Epoch
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)

            # Save Model Every Epoch
            model_save_path = os.path.join(save_path, f"spice_ai_mobilenet_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model Saved: {model_save_path}")

        print("Training Complete")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_model()
