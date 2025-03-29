import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import numpy as np
from datetime import datetime

# Load Model Function
def load_model(model_path, device):
    try:
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, 8)
        )
        # Load state dict with map_location for compatibility
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except RuntimeError as e:
        st.error(f"Error loading model: {e}")
        raise

# Image Preprocessing Function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Open Image Function
def open_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")

# Predict Function
def predict(image, model, device):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image).squeeze().cpu().numpy()
    scores = [round(100 * (1 / (1 + np.exp(-x))), 2) for x in outputs]
    return {label: score for label, score in zip(CLASS_CONFIG.keys(), scores)}

# Compute Inspection Score with Weighted Ranges
def compute_inspection_score(predictions):
    total = 0
    for label, score in predictions.items():
        config = CLASS_CONFIG[label]
        total += get_weighted_value(score, config["ranges"])
    
    max_possible = sum(max(w for _, _, w in cfg["ranges"]) * 100 
                      for cfg in CLASS_CONFIG.values())
    normalized = (total / abs(max_possible)) * 100 if max_possible != 0 else 0
    return round(max(0, min(100, normalized)), 2)

# Get Weighted Value Based on Ranges
def get_weighted_value(score, ranges):
    for min_val, max_val, weight in ranges:
        if min_val <= score < max_val:
            return score * weight
    return score

# Cleaning Suggestions Based on Scores
def cleaning_suggestions(scores):
    suggestions = []
    if scores[2] > 20: suggestions.append("Repair physical damage immediately")
    if scores[3] > 20: suggestions.append("Consult electrical engineer")
    if scores[4] > 20: suggestions.append("Remove snow accumulation")
    if scores[5] > 20: suggestions.append("Clear water obstructions")
    if scores[6] > 20: suggestions.append("Clean foreign particles")
    if scores[7] > 20: suggestions.append("Install bird deterrents")
    return suggestions or ["No critical issues detected"]

# Scoring Configuration for Each Class
CLASS_CONFIG = {
    "Panel Detected": {
        "ranges": [(0, 20, 0.5), (20, 40, 0.8), (40, 100, 1.2)],
        "description": "Detection confidence level"
    },
    "Clean Panel": {
        "ranges": [(0, 20, 0.1), (20, 40, 0.3), (40, 70, 0.6), (70, 100, 1.0)],
        "description": "Cleanliness assessment"
    },
    "Physical Damage": {
        "ranges": [(0, 10, -0.2), (10, 30, -0.5), (30, 100, -1.5)],
        "description": "Structural integrity evaluation"
    },
    "Electrical Damage": {
        "ranges": [(0, 10, -0.2), (10, 30, -0.5), (30, 100, -1.5)],
        "description": "Electrical safety evaluation"
    },
    "Snow Covered": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 100, -1.0)],
        "description": "Snow coverage impact"
    },
    "Water Obstruction": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 100, -1.0)],
        "description": "Water obstruction impact"
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 100, -1.5)],
        "description": "Impact of foreign particles"
    },
    "Bird Interference": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 100, -1.5)],
        "description": "Impact of bird interference"
    }
}

# Main Function for Streamlit App
def main():
    st.set_page_config(page_title="SPICE.AI: Solar Panel Inspection", layout="wide")
    
    # Title and Current Date/Time Display
    st.title("SPICE.AI: Solar Panel Inspection & Classification Engine")
    st.caption(f"Current date: {datetime.now().strftime('%A %B %d %Y %I:%M %p %Z')}")
    
    # Tabs for Different Sections
    tabs = st.tabs(["📖 How to Use", "🏆 Total Score", "📊 Label Analysis", "🔍 Outcome"])
    
    with tabs[0]:
        st.header("User Guide")
        st.markdown("""
            Upload an image of a solar panel to analyze its condition.
            The system evaluates cleanliness and detects potential issues such as physical damage or obstructions.
            Scores are weighted based on severity.
            """)
    
    # Model Loading and File Upload Section
    model_path = "D:/Projects/SPICE.AI/models/old/spice_ai_mobilenetv3_epoch_2.pth"
    
