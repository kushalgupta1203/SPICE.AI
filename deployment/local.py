import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import torch.nn as nn
import torchvision.models as models
import numpy as np
import requests
import pandas as pd

# Load Model Function
def load_model(model_path, device, num_classes=8):
    try:
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[0].in_features

        # Adjust classifier to match checkpoint dimensions
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1280),  # Match checkpoint dimensions
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)  # Match final output dimensions
        )

        # Load state dict with strict=False to allow partial loading
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

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
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error opening image: {e}")
        return None


# Function to display a smaller version of the image
def display_compressed_image(image, max_width=400):
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        image = image.resize((max_width, new_height))
    st.image(image, use_container_width=False)

# Predict Function
def predict(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor).squeeze().cpu().numpy()
    scores = [round(100 * (1 / (1 + np.exp(-x))), 1) for x in outputs]  # Changed to 1 decimal
    return {label: score for label, score in zip(CLASS_CONFIG.keys(), scores)}

# Function to print label analysis to the terminal
def print_label_analysis(predictions):
    print("--- Panel Analysis ---")
    for label, score in predictions.items():
        print(f"{label}: {score:.1f}%")
    print("----------------------")

def display_total_score(total_score):
    st.markdown(f"## Total Score: {total_score:.1f}/100")  # Force 1 decimal

    # Add score classification with more granular thresholds
    if total_score >= 90:
        st.success("EXCELLENT CONDITION")
    elif total_score >= 80:
        st.success("VERY GOOD CONDITION")
    elif total_score >= 70:
        st.success("GOOD CONDITION")
    elif total_score >= 60:
        st.warning("ABOVE AVERAGE CONDITION")
    elif total_score >= 50:
        st.warning("FAIR CONDITION")
    elif total_score >= 40:
        st.warning("POOR CONDITION")
    elif total_score >= 30:
        st.error("VERY POOR CONDITION")
    else:
        st.error("CRITICAL CONDITION")

# Cleaning Suggestions Based on Scores
def cleaning_suggestions(predictions):
    suggestions = []

    # Clean panel check
    clean_panel = predictions.get("Clean Panel", 0)
    if clean_panel > 90 and predictions.get("Physical Damage", 0) < 10 and predictions.get("Electrical Damage", 0) < 10:
        return ["🟢 No cleaning required. Panel is in excellent condition."]
    
    if clean_panel < 70:
        suggestions.append("🟠 Cleaning required. Dirt accumulation may impact efficiency.")

    # Physical Damage
    damage = predictions.get("Physical Damage", 0)
    if damage > 90:
        suggestions.append("🔴 Critical physical damage! Immediate repair required.")
    elif damage > 80:
        suggestions.append("🔴 Critical physical damage! Immediate repair required.")
    elif damage > 70:
        suggestions.append("🟠 High physical damage. Repair strongly recommended.")
    elif damage > 60:
        suggestions.append("🟠 High physical damage. Repair strongly recommended.")
    elif damage > 50:
        suggestions.append("🟠 High physical damage. Repair strongly recommended.")
    elif damage > 40:
        suggestions.append("🟠 High physical damage. Repair strongly recommended.")
    elif damage > 30:
        suggestions.append("🟡 Moderate physical damage. Schedule maintenance soon.")
    elif damage > 20:
        suggestions.append("🟡 Noticeable physical damage. Preventive action advised.")
    elif damage > 10:
        suggestions.append("🟡 Negligible physical damage detected. Preventive steps recommended.")
    elif damage > 5:
        suggestions.append("🟡 Negligible physical damage detected. Preventive steps recommended.")

    # Electrical Damage
    electrical = predictions.get("Electrical Damage", 0)
    if electrical > 90:
        suggestions.append("🔴 Critical electrical damage! Immediate expert consultation required.")
    elif electrical > 80:
        suggestions.append("🔴 Severe electrical issue. Urgent inspection required.")
    elif electrical > 70:
        suggestions.append("🔴 High electrical damage. Troubleshooting required soon.")
    elif electrical > 60:
        suggestions.append("🔴 High electrical damage. Troubleshooting required soon.")
    elif electrical > 50:
        suggestions.append("🔴 High electrical damage. Troubleshooting required soon.")
    elif electrical > 40:
        suggestions.append("🔴 High electrical damage. Troubleshooting required soon.")
    elif electrical > 30:
        suggestions.append("🟠 Low electrical concern detected. Monitor for worsening symptoms.")
    elif electrical > 20:
        suggestions.append("🟠 Very minor electrical issue. No immediate action required but stay cautious.")
    elif electrical > 10:
        suggestions.append("🟠 Negligible electrical concern detected. Preventive steps recommended.")


    # Snow Coverage
    snow = predictions.get("Snow Covered", 0)
    if snow > 90:
        suggestions.append("🔴 Panel fully covered with snow! Immediate removal needed.")
    elif snow > 80:
        suggestions.append("🔴 Panel fully covered with snow! Immediate removal needed.")
    elif snow > 70:
        suggestions.append("🔴 Panel fully covered with snow! Immediate removal needed.")
    elif snow > 60:
        suggestions.append("🔴 Panel fully covered with snow! Immediate removal needed.")
    elif snow > 50:
        suggestions.append("🔴 Panel fully covered with snow! Immediate removal needed.")
    elif snow > 40:
        suggestions.append("🔴 Panel fully covered with snow! Immediate removal needed.")
    elif snow > 30:
        suggestions.append("🟠 Low snow presence. No immediate action but prevent buildup.")
    elif snow > 20:
        suggestions.append("🟠 Low snow presence. No immediate action but prevent buildup.")
    elif snow > 10:
        suggestions.append("🟠 Low snow presence. No immediate action but prevent buildup.")

    # Water Obstruction
    water = predictions.get("Water Obstruction", 0)
    if water > 90:
        suggestions.append("🔴 Heavy water accumulation. Cleaning recommended urgently.")
    elif water > 80:
        suggestions.append("🔴 Heavy water accumulation. Cleaning recommended urgently.")
    elif water > 70:
        suggestions.append("🔴 Heavy water accumulation. Cleaning recommended urgently.")
    elif water > 60:
        suggestions.append("🔴 Heavy water accumulation. Cleaning recommended urgently.")
    elif water > 50:
        suggestions.append("🟠 Moderate water presence. Consider clearing soon.")
    elif water > 40:
        suggestions.append("🟠 Moderate water presence. Consider clearing soon.")
    elif water > 30:
        suggestions.append("🟠 Moderate water presence. Consider clearing soon.")
    elif water > 20:
        suggestions.append("🟠 Moderate water presence. Consider clearing soon.")
    elif water > 10:
        suggestions.append("🟠 Moderate water presence. Consider clearing soon.")

    # Foreign Particle Contamination (Dust, Dirt, etc.)
    contamination = predictions.get("Foreign Particle Contamination", 0)
    if contamination > 90:
        suggestions.append("🔴 Heavy dust accumulation. Clean the panel soon.")
    elif contamination > 80:
        suggestions.append("🔴 Heavy dust accumulation. Clean the panel soon.")
    elif contamination > 70:
        suggestions.append("🔴 Heavy dust accumulation. Clean the panel soon.")
    elif contamination > 60:
        suggestions.append("🔴 Heavy dust accumulation. Clean the panel soon.")
    elif contamination > 50:
        suggestions.append("🟠 Moderate dust accumulation detected. Cleaning recommended.")
    elif contamination > 40:
        suggestions.append("🟠 Moderate dust accumulation detected. Cleaning recommended.")
    elif contamination > 30:
        suggestions.append("🟠 Moderate dust accumulation detected. Cleaning recommended.")
    elif contamination > 20:
        suggestions.append("🟠 Moderate dust accumulation detected. Cleaning recommended.")
    elif contamination > 10:
        suggestions.append("🟢 Light contamination present. Preventive cleaning advised.")
    elif contamination > 5:
        suggestions.append("🟢 Light contamination present. Preventive cleaning advised.")

    # Bird Droppings/Interference
    birds = predictions.get("Bird Interference", 0)
    if birds > 90:
        suggestions.append("🔴 Severe bird interference! Install deterrents immediately.")
    elif birds > 80:
        suggestions.append("🔴 Severe bird interference! Install deterrents immediately.")
    elif birds > 70:
        suggestions.append("🟠 Moderate bird presence. Deterrents may be needed.")
    elif birds > 60:
        suggestions.append("🟠 Moderate bird presence. Deterrents may be needed.")
    elif birds > 50:
        suggestions.append("🟠 Moderate bird presence. Deterrents may be needed.")
    elif birds > 40:
        suggestions.append("🟠 Moderate bird presence. Deterrents may be needed.")
    elif birds > 30:
        suggestions.append("🟠 Moderate bird presence. Deterrents may be needed.")
    elif birds > 20:
        suggestions.append("🟡 Light bird activity. Monitor and take action if needed.")
    elif birds > 10:
        suggestions.append("🟡 Light bird activity. Monitor and take action if needed.")

    return suggestions or ["🟢 No major issues detected. Panel in reasonable condition."]



# Scoring Configuration for Each Class
CLASS_CONFIG = {
    "Panel Detected": {
        "ranges": [(0, 5, 0), (5, 10, 0),
                   (10, 15, 0), (15, 20, 0), 
                   (20, 25, 0.05), (25, 30, 0.05), 
                   (30, 35, 0.1), (35, 40, 0.1),
                   (40, 45, 0.2), (45, 50, 0.2),
                   (50, 55, 0.6), (55, 60, 0.6),
                   (60, 65, 0.6), (65, 70, 0.6),
                   (70, 75, 0.7), (75, 80, 0.7),
                   (80, 85, 0.8), (85, 90, 0.85),
                   (90, 95, 0.9), (95, 101, 0.9)]
    },
    "Clean Panel": {
        "ranges": [(0, 5, 0.7), (5, 10, 0.7),
                   (10, 15, 0.7), (15, 20, 0.7),
                   (20, 25, 0.5), (25, 30, 0.5),
                   (30, 35, 0.5), (35, 40, 0.5),
                   (40, 45, 0.2), (45, 50, 0.2),
                   (50, 55, 0.2), (55, 60, 0.2),
                   (60, 65, 0.5), (65, 70, 0.1),
                   (70, 75, 0.5), (75, 80, 0.1),
                   (80, 85, 0.05), (85, 90, 0.05),
                   (90, 95, 0.05), (95, 101, 0.05)]
    },
    "Physical Damage": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Electrical Damage": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Snow Covered": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Water Obstruction": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Bird Interference": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    }
}

SCORE_RATIOS = {
    "Panel Detected": {
        "ranges": [(0, 5, 0), (5, 10, 0),
                   (10, 15, 0), (15, 20, 0), 
                   (20, 25, 0.05), (25, 30, 0.05), 
                   (30, 35, 0.1), (35, 40, 0.1),
                   (40, 45, 0.2), (45, 50, 0.2),
                   (50, 55, 0.6), (55, 60, 0.6),
                   (60, 65, 0.6), (65, 70, 0.6),
                   (70, 75, 0.7), (75, 80, 0.7),
                   (80, 85, 0.8), (85, 90, 0.85),
                   (90, 95, 0.9), (95, 101, 0.9)]
    },
    "Clean Panel": {
        "ranges": [(0, 5, 0.7), (5, 10, 0.7),
                   (10, 15, 0.7), (15, 20, 0.7),
                   (20, 25, 0.5), (25, 30, 0.5),
                   (30, 35, 0.5), (35, 40, 0.5),
                   (40, 45, 0.2), (45, 50, 0.2),
                   (50, 55, 0.2), (55, 60, 0.2),
                   (60, 65, 0.5), (65, 70, 0.1),
                   (70, 75, 0.5), (75, 80, 0.1),
                   (80, 85, 0.05), (85, 90, 0.05),
                   (90, 95, 0.05), (95, 101, 0.05)]
    },
    "Physical Damage": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Electrical Damage": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Snow Covered": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Water Obstruction": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    },
    "Bird Interference": {
        "ranges": [(0, 2, -1), (2, 5, -0.8), (5, 10, -0.8),
                   (10, 15, -0.7), (15, 20, -0.7),
                   (20, 25, -0.5), (25, 30, -0.5),
                   (30, 35, -0.2), (35, 40, -0.2),
                   (40, 45, -0.3), (45, 50, -0.3),
                   (50, 55, -0.4), (55, 60, -0.4),
                   (60, 65, -0.45), (65, 70, -0.45),
                   (70, 75, -0.45), (75, 80, -0.45),
                   (80, 85, -0.55), (85, 90, -0.55),
                   (90, 95, -0.55), (95, 101, -0.55)]
    }
}



def main():
    st.set_page_config(page_title="SPICE.AI", layout="wide")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a CSS media query approach for responsive images
    st.markdown("""
        <style>
        .desktop-logo {
            display: block;
        }
        .mobile-logo {
            display: none;
        }

        /* Media query for mobile devices */
        @media (max-width: 768px) {
            .desktop-logo {
                display: none;
            }
            .mobile-logo {
                display: block;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Display both logos with appropriate CSS classes
    try:
        st.markdown("""
            <div class="desktop-logo">
                <img src="https://github.com/kushalgupta1203/SPICE.AI/blob/main/deployment/logo_comp.png?raw=true" style="width: 100%;">
            </div>
            <div class="mobile-logo">
                <img src="https://github.com/kushalgupta1203/SPICE.AI/blob/main/deployment/logo_phone.png?raw=true" style="width: 100%;">
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error with logo display: {e}")
        st.title("SPICE.AI: Solar Panel Inspection & Classification Engine")
        st.warning("Logo not found. Using default title.")

    # Add a visual separator
    st.markdown("---")


    # Store model, image, tensor, and predictions in session state
    if 'panel_detection_model' not in st.session_state:
        try:
            panel_model_path = r"D:\Projects\SPICE.AI\models\spice_ai_mobilenetv3_v2.0.pth"
            # Load the panel detection model (v2.0) with 8 classes, as it predicts "Panel Detected" along with others
            st.session_state.panel_detection_model = load_model(panel_model_path, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading panel detection model: {e}")
            st.session_state.panel_detection_model = None

    if 'inspection_model_v11' not in st.session_state:
        try:
            inspection_model_path = r"D:\Projects\SPICE.AI\models\spice_ai_mobilenetv3_v1.1.pth"
            # Load the inspection model (v1.1) also with 8 classes
            st.session_state.inspection_model_v11 = load_model(inspection_model_path, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading inspection model v1.1: {e}")
            st.session_state.inspection_model_v11 = None

    if 'inspection_model_v20' not in st.session_state:
        try:
            inspection_model_path = r"D:\Projects\SPICE.AI\models\spice_ai_mobilenetv3_v2.0.pth"
            # Load the inspection model (v2.0)
            st.session_state.inspection_model_v20 = load_model(inspection_model_path, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading inspection model v2.0: {e}")
            st.session_state.inspection_model_v20 = None

    uploaded_file = st.file_uploader("Upload solar panel image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Open the image and store it in the session state
        image = open_image(uploaded_file)
        if image is not None:
            st.session_state.image = image
            st.write("Uploaded Image:")
            display_compressed_image(image)  # Display smaller version

            # Preprocess the image and store the tensor in the session state
            st.session_state.image_tensor = preprocess_image(image)

        # Predict using Panel Detection Model v2.0 immediately after upload
        if st.session_state.panel_detection_model is not None:
            panel_predictions = predict(st.session_state.image_tensor, st.session_state.panel_detection_model, device)
            panel_detected_score = panel_predictions.get("Panel Detected", 0)

            if panel_detected_score < 50:
                st.error("No panel detected. Re-upload an image with a clear view of the solar panel.")
                for key in list(st.session_state.keys()):
                    if key not in ['panel_detection_model', 'image']:
                        del st.session_state[key]
                st.stop()
            else:
                st.success("Image uploaded.")
                st.session_state.panel_predictions = panel_predictions
                st.session_state.image_uploaded = True
        else:
            st.error("Inspection model failed to load.")

    # Inspection Analysis Section
    if 'image_tensor' in st.session_state and \
    st.session_state.inspection_model_v11 is not None and \
    st.session_state.inspection_model_v20 is not None and \
    'panel_predictions' in st.session_state:

        # Add a visual separator
        st.markdown("---")

        predictions_v11 = predict(st.session_state.image_tensor, st.session_state.inspection_model_v11, device)
        predictions_v20 = predict(st.session_state.image_tensor, st.session_state.inspection_model_v20, device)

        final_predictions = {}
        for label in CLASS_CONFIG.keys():
            if label == "Clean Panel":
                final_predictions[label] = max(predictions_v11[label], predictions_v20[label])
            elif label in ["Physical Damage", "Electrical Damage", "Snow Covered", "Water Obstruction", "Foreign Particle Contamination", "Bird Interference"]:
                final_predictions[label] = min(predictions_v11[label], predictions_v20[label])
            elif label == "Panel Detected":
                final_predictions[label] = predictions_v20[label]
            else:
                st.error(f"Unexpected label: {label}")
                continue

        st.session_state.inspection_predictions = final_predictions
        st.header("Inspection Analysis")
        print_label_analysis(final_predictions)

        df = pd.DataFrame.from_dict(final_predictions, orient='index', columns=['Score'])
        df['Score'] = df['Score'].apply(lambda x: f"{x:.2f}%")

        st.markdown("""
            <style>
                div[data-testid="stDataFrame"] td {
                    text-align: center !important;
                }
            </style>
        """, unsafe_allow_html=True)

        st.dataframe(df)

        total_score = 0.0
        for label, value in final_predictions.items():
            if label in SCORE_RATIOS:
                for (low, high, ratio) in SCORE_RATIOS[label]["ranges"]:
                    if low <= value < high:
                        total_score += value * ratio
                        break

        display_total_score(total_score)

        # Cleaning Suggestions Section
        st.header("Suggestions")
        predictions = st.session_state.inspection_predictions
        suggestions = cleaning_suggestions(predictions)

        severity_order = {"🔴": 1, "🟠": 2, "🟡": 3, "🟢": 4}
        sorted_suggestions = sorted(suggestions, key=lambda s: severity_order.get(s[:1], 5))

        for suggestion in sorted_suggestions:
            st.markdown(f"- {suggestion}")
    else:
        st.info("Upload an image to see the inspection report of your solar panel.")



if __name__ == "__main__":
    main()
