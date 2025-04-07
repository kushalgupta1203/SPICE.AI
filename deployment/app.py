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
def load_model(model_url, device, num_classes=8):
     try:
         # Download the model file from URL
         response = requests.get(model_url)
         response.raise_for_status()  # Raise an exception for bad responses
 
         # Load model from the downloaded content
         model_bytes = BytesIO(response.content)
 
         # Create model architecture
         model = models.mobilenet_v3_large(weights=None)
         num_ftrs = model.classifier[0].in_features
 
         # Adjust classifier to match checkpoint dimensions
         model.classifier = nn.Sequential(
             nn.Linear(num_ftrs, 1280),  # Match checkpoint dimensions
             nn.Hardswish(),
             nn.Dropout(0.2),
             nn.Linear(1280, num_classes)  # Match final output dimensions
         )
 
         # Load state dict from bytes with strict=False to allow partial loading
         state_dict = torch.load(model_bytes, map_location=device)
         model.load_state_dict(state_dict, strict=False)
 
         model.to(device)
         model.eval()
         return model
     except Exception as e:
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
        st.success("GOOD CONDITION")
    elif total_score >= 70:
        st.warning("AVERAGE CONDITION")
    elif total_score >= 60:
        st.warning("POOR CONDITION")
    elif total_score >= 50:
        st.error("POOR CONDITION")
    elif total_score >= 40:
        st.error("CRITICAL CONDITION")
    elif total_score >= 30:
        st.error("CRITICAL CONDITION")
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
        suggestions.append("🔴 Heavy foreign particle accumulation. Clean the panel soon.")
    elif contamination > 80:
        suggestions.append("🔴 Heavy foreign particle accumulation. Clean the panel soon.")
    elif contamination > 70:
        suggestions.append("🔴 Heavy foreign particle accumulation. Clean the panel soon.")
    elif contamination > 60:
        suggestions.append("🔴 Heavy foreign particle accumulation. Clean the panel soon.")
    elif contamination > 50:
        suggestions.append("🟠 Moderate foreign particle accumulation detected. Cleaning recommended.")
    elif contamination > 40:
        suggestions.append("🟠 Moderate foreign particle accumulation detected. Cleaning recommended.")
    elif contamination > 30:
        suggestions.append("🟠 Moderate foreign particle accumulation detected. Cleaning recommended.")
    elif contamination > 20:
        suggestions.append("🟠 Moderate foreign particle accumulation detected. Cleaning recommended.")
    elif contamination > 10:
        suggestions.append("🟢 Light foreign particle contamination present. Preventive cleaning advised.")
    elif contamination > 5:
        suggestions.append("🟢 Light foreign particle contamination present. Preventive cleaning advised.")

    # Bird Droppings/Interference
    birds = predictions.get("Bird Interference", 0)
    if birds > 90:
        suggestions.append("🔴 Severe bird interference! Install deterrents immediately.")
    elif birds > 80:
        suggestions.append("🔴 Severe bird interference! Install deterrents immediately.")
    elif birds > 70:
        suggestions.append("🟠 Moderate bird interference. Deterrents may be needed.")
    elif birds > 60:
        suggestions.append("🟠 Moderate bird interference. Deterrents may be needed.")
    elif birds > 50:
        suggestions.append("🟠 Moderate bird interference. Deterrents may be needed.")
    elif birds > 40:
        suggestions.append("🟠 Moderate bird interference. Deterrents may be needed.")
    elif birds > 30:
        suggestions.append("🟠 Moderate bird interference. Deterrents may be needed.")
    elif birds > 20:
        suggestions.append("🟡 Light bird activity. Monitor and take action if needed.")
    elif birds > 10:
        suggestions.append("🟡 Light bird activity. Monitor and take action if needed.")

    return suggestions or ["🟢 No major issues detected. Panel in reasonable condition."]



# Scoring Configuration for Each Class
CLASS_CONFIG = {
    "Panel Detected": {
        "ranges": [(0, 2, 0.017), (2, 4, 0.023), (4, 6, 0.038), (6, 8, 0.041), (8, 10, 0.056),
                   (10, 12, 0.062), (12, 14, 0.071), (14, 16, 0.084), (16, 18, 0.092), (18, 20, 0.103),
                   (20, 22, 0.312), (22, 24, 0.324), (24, 26, 0.339), (26, 28, 0.347), (28, 30, 0.358),
                   (30, 32, 0.417), (32, 34, 0.429), (34, 36, 0.438), (36, 38, 0.446), (38, 40, 0.459),
                   (40, 42, 0.516), (42, 44, 0.527), (44, 46, 0.538), (46, 48, 0.549), (48, 50, 0.557),
                   (50, 52, 0.613), (52, 54, 0.628), (54, 56, 0.635), (56, 58, 0.647), (58, 60, 0.659),
                   (60, 62, 0.712), (62, 64, 0.724), (64, 66, 0.736), (66, 68, 0.748), (68, 70, 0.759),
                   (70, 72, 0.814), (72, 74, 0.827), (74, 76, 0.836), (76, 78, 0.842), (78, 80, 0.859),
                   (80, 82, 0.913), (82, 84, 0.925), (84, 86, 0.937), (86, 88, 0.948), (88, 90, 0.95),
                   (90, 92, 0.95), (92, 94, 0.95), (94, 96, 0.95), (96, 98, 0.95), (98, 100, 0.95), (100, 101, 0.95)]
    },
    "Clean Panel": {
        "ranges": [(0, 2, -0.917), (2, 4, -0.925), (4, 6, -0.936), (6, 8, -0.948), (8, 10, -0.95),
                   (10, 12, -0.714), (12, 14, -0.726), (14, 16, -0.735), (16, 18, -0.747), (18, 20, -0.758),
                   (20, 22, -0.713), (22, 24, -0.725), (24, 26, -0.737), (26, 28, -0.748), (28, 30, -0.759),
                   (30, 32, -0.514), (32, 34, -0.527), (34, 36, -0.535), (36, 38, -0.542), (38, 40, -0.558),
                   (40, 42, -0.217), (42, 44, -0.228), (44, 46, -0.239), (46, 48, -0.246), (48, 50, -0.253),
                   (50, 52, -0.113), (52, 54, -0.127), (54, 56, -0.138), (56, 58, -0.142), (58, 60, -0.159),
                   (60, 62, -0.116), (62, 64, -0.128), (64, 66, -0.137), (66, 68, -0.145), (68, 70, -0.156),
                   (70, 72, -0.057), (72, 74, -0.063), (74, 76, -0.071), (76, 78, -0.082), (78, 80, -0.094),
                   (80, 82, 0.014), (82, 84, 0.023), (84, 86, 0.031), (86, 88, 0.042), (88, 90, 0.052),
                   (90, 92, 0.017), (92, 94, 0.026), (94, 96, 0.038), (96, 98, 0.047), (98, 100, 0.059), (100, 101, 0.068)]
    },
    "Physical Damage": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.817), (4, 6, -0.823), (6, 8, -0.834), (8, 10, -0.845),
                   (10, 12, -0.716), (12, 14, -0.724), (14, 16, -0.735), (16, 18, -0.746), (18, 20, -0.753),
                   (20, 22, -0.517), (22, 24, -0.525), (24, 26, -0.536), (26, 28, -0.548), (28, 30, -0.559),
                   (30, 32, -0.214), (32, 34, -0.226), (34, 36, -0.237), (36, 38, -0.243), (38, 40, -0.252),
                   (40, 42, -0.317), (42, 44, -0.326), (44, 46, -0.332), (46, 48, -0.345), (48, 50, -0.358),
                   (50, 52, -0.413), (52, 54, -0.421), (54, 56, -0.432), (56, 58, -0.443), (58, 60, -0.456),
                   (60, 62, -0.462), (62, 64, -0.473), (64, 66, -0.484), (66, 68, -0.492), (68, 70, -0.459),
                   (70, 72, -0.463), (72, 74, -0.471), (74, 76, -0.482), (76, 78, -0.493), (78, 80, -0.457),
                   (80, 82, -0.562), (82, 84, -0.573), (84, 86, -0.584), (86, 88, -0.591), (88, 90, -0.559),
                   (90, 92, -0.561), (92, 94, -0.572), (94, 96, -0.583), (96, 98, -0.556), (98, 100, -0.547), (100, 101, -0.553)]
    },
    "Electrical Damage": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.812), (4, 6, -0.823), (6, 8, -0.831), (8, 10, -0.842),
                   (10, 12, -0.713), (12, 14, -0.724), (14, 16, -0.736), (16, 18, -0.747), (18, 20, -0.758),
                   (20, 22, -0.512), (22, 24, -0.523), (24, 26, -0.534), (26, 28, -0.546), (28, 30, -0.557),
                   (30, 32, -0.517), (32, 34, -0.528), (34, 36, -0.539), (36, 38, -0.541), (38, 40, -0.552),
                   (40, 42, -0.513), (42, 44, -0.524), (44, 46, -0.535), (46, 48, -0.546), (48, 50, -0.557),
                   (50, 52, -0.514), (52, 54, -0.526), (54, 56, -0.537), (56, 58, -0.548), (58, 60, -0.559),
                   (60, 62, -0.515), (62, 64, -0.527), (64, 66, -0.538), (66, 68, -0.549), (68, 70, -0.551),
                   (70, 72, -0.615), (72, 74, -0.627), (74, 76, -0.638), (76, 78, -0.649), (78, 80, -0.651),
                   (80, 82, -0.713), (82, 84, -0.725), (84, 86, -0.736), (86, 88, -0.747), (88, 90, -0.758),
                   (90, 92, -0.763), (92, 94, -0.772), (94, 96, -0.758), (96, 98, -0.749), (98, 100, -0.751), (100, 101, -0.752)]
    },
    "Snow Covered": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.813), (4, 6, -0.824), (6, 8, -0.836), (8, 10, -0.847),
                   (10, 12, -0.719), (12, 14, -0.721), (14, 16, -0.732), (16, 18, -0.743), (18, 20, -0.754),
                   (20, 22, -0.514), (22, 24, -0.525), (24, 26, -0.536), (26, 28, -0.547), (28, 30, -0.558),
                   (30, 32, -0.517), (32, 34, -0.529), (34, 36, -0.531), (36, 38, -0.542), (38, 40, -0.554),
                   (40, 42, -0.516), (42, 44, -0.527), (44, 46, -0.538), (46, 48, -0.549), (48, 50, -0.551),
                   (50, 52, -0.518), (52, 54, -0.529), (54, 56, -0.531), (56, 58, -0.542), (58, 60, -0.553),
                   (60, 62, -0.519), (62, 64, -0.521), (64, 66, -0.532), (66, 68, -0.544), (68, 70, -0.555),
                   (70, 72, -0.619), (72, 74, -0.621), (74, 76, -0.632), (76, 78, -0.643), (78, 80, -0.654),
                   (80, 82, -0.719), (82, 84, -0.728), (84, 86, -0.739), (86, 88, -0.741), (88, 90, -0.752),
                   (90, 92, -0.762), (92, 94, -0.773), (94, 96, -0.784), (96, 98, -0.756), (98, 100, -0.749), (100, 101, -0.758)]
    },
    "Water Obstruction": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.817), (4, 6, -0.828), (6, 8, -0.839), (8, 10, -0.841),
                   (10, 12, -0.716), (12, 14, -0.727), (14, 16, -0.738), (16, 18, -0.749), (18, 20, -0.751),
                   (20, 22, -0.515), (22, 24, -0.526), (24, 26, -0.537), (26, 28, -0.548), (28, 30, -0.559),
                   (30, 32, -0.516), (32, 34, -0.527), (34, 36, -0.538), (36, 38, -0.549), (38, 40, -0.551),
                   (40, 42, -0.517), (42, 44, -0.528), (44, 46, -0.539), (46, 48, -0.541), (48, 50, -0.552),
                   (50, 52, -0.518), (52, 54, -0.529), (54, 56, -0.531), (56, 58, -0.542), (58, 60, -0.553),
                   (60, 62, -0.519), (62, 64, -0.521), (64, 66, -0.532), (66, 68, -0.543), (68, 70, -0.554),
                   (70, 72, -0.617), (72, 74, -0.628), (74, 76, -0.639), (76, 78, -0.641), (78, 80, -0.652),
                   (80, 82, -0.718), (82, 84, -0.729), (84, 86, -0.731), (86, 88, -0.742), (88, 90, -0.753),
                   (90, 92, -0.762), (92, 94, -0.773), (94, 96, -0.764), (96, 98, -0.755), (98, 100, -0.746), (100, 101, -0.754)]
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 2, 0.013), (2, 4, -0.517), (4, 6, -0.528), (6, 8, -0.539), (8, 10, -0.541),
                   (10, 12, -0.315), (12, 14, -0.326), (14, 16, -0.337), (16, 18, -0.348), (18, 20, -0.359),
                   (20, 22, -0.316), (22, 24, -0.327), (24, 26, -0.518), (26, 28, -0.529), (28, 30, -0.531),
                   (30, 32, -0.317), (32, 34, -0.328), (34, 36, -0.339), (36, 38, -0.341), (38, 40, -0.352),
                   (40, 42, -0.318), (42, 44, -0.329), (44, 46, -0.417), (46, 48, -0.428), (48, 50, -0.439),
                   (50, 52, -0.319), (52, 54, -0.321), (54, 56, -0.419), (56, 58, -0.421), (58, 60, -0.432),
                   (60, 62, -0.311), (62, 64, -0.322), (64, 66, -0.333), (66, 68, -0.344), (68, 70, -0.355),
                   (70, 72, -0.312), (72, 74, -0.323), (74, 76, -0.334), (76, 78, -0.345), (78, 80, -0.356),
                   (80, 82, -0.313), (82, 84, -0.324), (84, 86, -0.335), (86, 88, -0.346), (88, 90, -0.357),
                   (90, 92, -0.314), (92, 94, -0.325), (94, 96, -0.336), (96, 98, -0.347), (98, 100, -0.358), (100, 101, -0.369)]
    },
    "Bird Interference": {
        "ranges": [(0, 2, 0.017), (2, 4, -0.516), (4, 6, -0.527), (6, 8, -0.538), (8, 10, -0.549),
                   (10, 12, -0.319), (12, 14, -0.321), (14, 16, -0.332), (16, 18, -0.343), (18, 20, -0.354),
                   (20, 22, -0.313), (22, 24, -0.324), (24, 26, -0.516), (26, 28, -0.527), (28, 30, -0.538),
                   (30, 32, -0.314), (32, 34, -0.325), (34, 36, -0.336), (36, 38, -0.347), (38, 40, -0.358),
                   (40, 42, -0.315), (42, 44, -0.326), (44, 46, -0.337), (46, 48, -0.348), (48, 50, -0.359),
                   (50, 52, -0.316), (52, 54, -0.327), (54, 56, -0.338), (56, 58, -0.349), (58, 60, -0.351),
                   (60, 62, -0.317), (62, 64, -0.328), (64, 66, -0.339), (66, 68, -0.341), (68, 70, -0.352),
                   (70, 72, -0.318), (72, 74, -0.329), (74, 76, -0.331), (76, 78, -0.342), (78, 80, -0.353),
                   (80, 82, -0.319), (82, 84, -0.321), (84, 86, -0.332), (86, 88, -0.343), (88, 90, -0.354),
                   (90, 92, -0.311), (92, 94, -0.322), (94, 96, -0.333), (96, 98, -0.344), (98, 100, -0.355), (100, 101, -0.366)]
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
        st.title("SPICE.AI: Solar Panel Inspection & Classification Engine")

    # Store model, image, tensor, and predictions in session state 
    if 'panel_detection_model' not in st.session_state:
        try:
            panel_model_url = "https://raw.githubusercontent.com/kushalgupta1203/SPICE.AI/main/deployment/spice_ai_mobilenetv3_v2.0.pth"
            # Load the panel detection model (v2.0) with 8 classes
            st.session_state.panel_detection_model = load_model(panel_model_url, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading panel detection model: {e}")
            st.session_state.panel_detection_model = None

    if 'inspection_model_v11' not in st.session_state:
        try:
            inspection_model_url_v11 = "https://raw.githubusercontent.com/kushalgupta1203/SPICE.AI/main/deployment/spice_ai_mobilenetv3_v1.1.pth"
            # Load the inspection model (v1.1)
            st.session_state.inspection_model_v11 = load_model(inspection_model_url_v11, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading inspection model v1.1: {e}")
            st.session_state.inspection_model_v11 = None

    if 'inspection_model_v20' not in st.session_state:
        try:
            inspection_model_url_v20 = "https://raw.githubusercontent.com/kushalgupta1203/SPICE.AI/main/deployment/spice_ai_mobilenetv3_v2.0.pth"
            # Load the inspection model (v2.0)
            st.session_state.inspection_model_v20 = load_model(inspection_model_url_v20, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading inspection model v2.0: {e}")
            st.session_state.inspection_model_v20 = None

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Open the image and store it in the session state
        image = open_image(uploaded_file)
        if image is not None:
            st.session_state.image = image
            st.header("Uploaded Image:")
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
                final_predictions[label] = (0.3 * predictions_v11[label]) + (0.7 * predictions_v20[label])
            elif label == "Physical Damage":
                final_predictions[label] = (0.5 * predictions_v11[label]) + (0.5 * predictions_v20[label])
            elif label == "Electrical Damage":
                final_predictions[label] = (0.5 * predictions_v11[label]) + (0.5 * predictions_v20[label])
            elif label == "Snow Covered":
                final_predictions[label] = (0.5 * predictions_v11[label]) + (0.5 * predictions_v20[label])
            elif label == "Water Obstruction":
                final_predictions[label] = (0.2 * predictions_v11[label]) + (0.8 * predictions_v20[label])
            elif label == "Foreign Particle Contamination":
                final_predictions[label] = (0.5 * predictions_v11[label]) + (0.5 * predictions_v20[label])
            elif label == "Bird Interference":
                final_predictions[label] = (0.3 * predictions_v11[label]) + (0.7 * predictions_v20[label])
            elif label == "Panel Detected":
                final_predictions[label] = predictions_v20[label]
            else:
                st.error(f"Unexpected label: {label}")

        st.session_state.inspection_predictions = final_predictions
        st.header("Inspection Analysis:")
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
            if label in CLASS_CONFIG:
                for (low, high, ratio) in CLASS_CONFIG[label]["ranges"]:
                    if low <= value < high:
                        total_score += value * ratio
                        break

        # Ensure the score stays within 0-100 range
        if total_score < 0:
            total_score = 0
        elif total_score > 100:
            total_score = 100

        display_total_score(total_score)

        # Cleaning Suggestions Section
        st.header("Suggestions:")
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
