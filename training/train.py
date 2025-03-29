import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Paths
train_dir = r"D:\Projects\SPICE.AI\dataset\splitted\train"
csv_path = os.path.join(train_dir, "_classes.csv")

val_dir = r"D:\Projects\SPICE.AI\dataset\splitted\val"
val_csv_path = os.path.join(val_dir, "_classes.csv")

save_path = r"D:\Projects\SPICE.AI\models\model.pth"

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CSV
df_train = pd.read_csv(csv_path)
df_val = pd.read_csv(val_csv_path)
class_names = df_train.columns[1:].tolist()
num_classes = len(class_names)

# Image Transformations (No Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class SolarPanelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[1:].values.astype(float), dtype=torch.float32)
        return image, labels

# DataLoaders (Batch Size 128)
batch_size = 128  # Updated batch size
train_dataset = SolarPanelDataset(csv_path, train_dir, transform=transform)
val_dataset = SolarPanelDataset(val_csv_path, val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Model (ResNet50)
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes),
    nn.Sigmoid()
)
model = model.to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Mixed Precision (Faster Training)
scaler = torch.cuda.amp.GradScaler()

# Early Stopping
best_val_loss = float("inf")
patience = 5
counter = 0

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision training
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # Validation Step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")
        counter = 0
    else:
        counter += 1

    scheduler.step(avg_val_loss)

    # Early Stopping
    if counter >= patience:
        print("Early stopping triggered!")
        break

print(f"Training Complete. Best model saved at {save_path}.")
