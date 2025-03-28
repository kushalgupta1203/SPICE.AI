import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import SolarPanelClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Paths
train_dir = "D:/Projects/SPICE.AI/dataset/augmented/train"
val_dir = "D:/Projects/SPICE.AI/dataset/processed/val"

# Hyperparameters
BATCH_SIZE = 32
LR = 0.0005  # Reduced learning rate for better stability
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6  # Multi-label classification (Dust, Snow, Bird Drop, etc.)
WEIGHT_DECAY = 1e-4  # Prevents overfitting

# Enhanced Data Augmentation for Training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  # Increased randomness
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # New addition
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Only Normalization for Validation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model
model = SolarPanelClassifier(num_classes=NUM_CLASSES).to(DEVICE)

# Modify Fully Connected Layers to Add Dropout
for module in model.modules():
    if isinstance(module, nn.Linear):
        module.dropout = nn.Dropout(p=0.4)  # Regularization

# ✅ **Change Loss Function for Multi-Label Classification**
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Training Loop
best_val_loss = float('inf')
early_stop_counter = 0
EARLY_STOPPING_PATIENCE = 3  # Reduce patience for faster stopping

os.makedirs("models", exist_ok=True)  # Create models directory

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        # ✅ **Change Label Processing**
        labels = nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()  # Convert to multi-label format

        loss = criterion(outputs, labels)  # BCE Loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # ✅ **Modify Accuracy Calculation (Multi-Label)**
        predicted = (torch.sigmoid(outputs) > 0.5).float()  # Apply threshold
        correct += (predicted == labels).sum().item()  # Count correct labels
        total += labels.numel()  # Total labels (not just batch size)

    train_acc = 100 * correct / total
    print(f"Training Loss: {train_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}%")

    # Validation Step
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            labels = nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    val_acc = 100 * correct / total
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

    # Learning Rate Scheduling
    scheduler.step(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/model.pth")  # Save in models folder
        print("Model saved in models folder as 'model.pth'")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print("Training Complete")
