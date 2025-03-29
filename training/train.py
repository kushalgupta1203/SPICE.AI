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
        # Get image filename and construct full path
        image_filename = self.data.iloc[idx, 0]
        image_path = os.path.join(self.img_dir, image_filename)

        # Check if the file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Open and convert the image to RGB format
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Extract labels from the CSV file (convert to tensor)
        labels = torch.tensor(self.data.iloc[idx, 1:].astype(int).values, dtype=torch.float32)
        return image, labels


# Data Augmentation & Normalization for Training and Validation
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

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=0, shuffle=False, pin_memory=True)

# Define Model Function with Dropout for Regularization
def get_mobilenet_v3():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Increased dropout rate for regularization
        nn.Linear(512, 8)  
    )
    return model

# Training Function with Model Saving for Each Epoch
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_mobilenet_v3().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 25  
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
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted_train = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.numel()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100

        # Validation Phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images_val, labels_val in valid_loader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs_val = model(images_val)
                    loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item()
                predicted_val = (torch.sigmoid(outputs_val) > 0.5).float()
                correct_val += (predicted_val == labels_val).sum().item()
                total_val += labels_val.numel()

        avg_val_loss = val_loss / len(valid_loader)
        val_accuracy = correct_val / total_val * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Save Model for Each Epoch with Epoch Number in Filename
        model_save_path = os.path.join(save_path, f"spice_ai_mobilenetv3_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch+1}.")

    print("Training Complete.")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_model()
