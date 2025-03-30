import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

# Dataset Class
class SolarPanelDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # First column is filename
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(self.data.iloc[idx, 1:].astype(int).values, dtype=torch.float32)
        return image, labels

# Define Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
train_csv = "D:/Projects/SPICE.AI/dataset/splitted/train/_classes.csv"
train_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/train"
val_csv = "D:/Projects/SPICE.AI/dataset/splitted/val/_classes.csv"
val_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/val"
test_csv = "D:/Projects/SPICE.AI/dataset/splitted/test/_classes.csv"
test_img_dir = "D:/Projects/SPICE.AI/dataset/splitted/test"

model_path = "D:/Projects/SPICE.AI/models/spice_ai_mobilenetv3_v2.0.pth"

# Load Dataset
train_dataset = SolarPanelDataset(train_csv, train_img_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

val_dataset = SolarPanelDataset(val_csv, val_img_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

test_dataset = SolarPanelDataset(test_csv, test_img_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Define MobileNetV3 Model
def get_mobilenetv3():
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),  # Corrected dimensions
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1280, 8)     # Adjusted to match checkpoint
    )
    return model


# Evaluation Function
def evaluate_model(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()
    
    accuracy = correct / total * 100
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss

# Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_mobilenetv3().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    print("\nEvaluating Model...")

    train_acc, train_loss = evaluate_model(train_loader, model, criterion, device)
    val_acc, val_loss = evaluate_model(val_loader, model, criterion, device)
    test_acc, test_loss = evaluate_model(test_loader, model, criterion, device)

    print(f"\nTrain Accuracy: {train_acc:.2f}%, Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()