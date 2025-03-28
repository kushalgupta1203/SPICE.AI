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

# Enable CuDNN optimizations
torch.backends.cudnn.benchmark = True

# Paths
train_dir = "D:/Projects/SPICE.AI/dataset/augmented/train"
val_dir = "D:/Projects/SPICE.AI/dataset/processed/val"

# Hyperparameters
BATCH_SIZE = 128
LR = 0.0005
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3
ACCUMULATION_STEPS = 2  # Helps with small batch sizes
DROPOUT_RATE = 0.5  # Dropout for regularization

# Data Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_regularization(model):
    """Applies dropout layers to prevent overfitting."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = DROPOUT_RATE

def train():
    # Load Datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Load Model
    model = SolarPanelClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    apply_regularization(model)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Training Loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels = nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()  # Multi-label handling

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS  # Gradient accumulation

            # Backpropagation
            scaler.scale(loss).backward() if scaler else loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or i == len(train_loader) - 1:
                scaler.step(optimizer) if scaler else optimizer.step()
                scaler.update() if scaler else None
                optimizer.zero_grad()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Multi-label threshold
            correct += (predicted == labels).sum().item()
            total += labels.numel()

        train_acc = 100 * correct / total
        print(f"Training Loss: {train_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                labels = nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()

                with torch.cuda.amp.autocast():
                    outputs = model(images)
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

        # Early Stopping & Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/model.pth")
            print("Model saved in models folder as 'model.pth'")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training Complete")


if __name__ == "__main__":
    train()
