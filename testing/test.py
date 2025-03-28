import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import SolarPanelClassifier

# Paths
test_dir = "D:/Projects/SPICE.AI/dataset/processed/test"
MODEL_PATH = "models/model.pth"

# Hyperparameters
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6  

# Transformations (Only Normalization)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model
model = SolarPanelClassifier(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Evaluation
criterion = nn.CrossEntropyLoss()
test_loss, correct, total = 0, 0, 0

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = 100 * correct / total
test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
