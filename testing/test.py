import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys

# Ensure correct import from models directory
sys.path.append(os.path.abspath("D:/Projects/SPICE.AI/"))
from models.model import SolarPanelClassifier

# Paths
TEST_DIR = "D:/Projects/SPICE.AI/dataset/processed/test"
MODEL_PATH = "D:/Projects/SPICE.AI/models/model.pth"

# Hyperparameters
BATCH_SIZE = 1  # Process one image at a time for individual inspection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6  

CLASS_NAMES = [
    "Clean",
    "Bird Drop",
    "Dusty",
    "Snow-Covered",
    "Electrical Damage",
    "Physical Damage"
]

CLEANING_SUGGESTIONS = {
    "Bird Drop": "Use mild detergent and water to remove bird droppings.",
    "Dusty": "Regularly clean with a soft brush or water spray.",
    "Snow-Covered": "Use a soft broom or heater-based removal.",
    "Electrical Damage": "Inspect wiring and consult a technician.",
    "Physical Damage": "Replace damaged panels or consult a professional."
}

# Ensure model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Transformations (Only Normalization)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model
model = SolarPanelClassifier(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Set model to evaluation mode

# Function to calculate inspection score
def get_inspection_score(pred_class, confidences):
    """
    Generate inspection score based on classification and confidence.
    - Clean → High score (85-100)
    - Issues → Lower score based on severity
    """
    if pred_class == "Clean":
        return 90 + (confidences.max().item() * 10)  # High score for clean panels
    
    # Reduce score based on confidence in problematic class
    return max(30, 80 - (confidences.max().item() * 50))  # Score between 30-80

# Evaluation (per image)
for idx, (image, _) in enumerate(test_loader):
    image = image.to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        confidences = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        predicted_class_idx = torch.argmax(confidences).item()
        predicted_class = CLASS_NAMES[predicted_class_idx]

        # Calculate inspection score
        score = get_inspection_score(predicted_class, confidences)

        # Get cleaning suggestions if needed
        cleaning_tip = CLEANING_SUGGESTIONS.get(predicted_class, "No cleaning required.")

        print(f"\n--- Image {idx+1} ---")
        print(f"Predicted Class: {predicted_class}")
        print(f"Inspection Score: {score:.2f}/100")
        print(f"Suggestion: {cleaning_tip}")
