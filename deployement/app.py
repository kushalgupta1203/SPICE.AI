import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Set Page Config
st.set_page_config(
    page_title="SPICE.AI - Solar Panel Inspection",
    page_icon="🌞",
    layout="wide"
)

# Sidebar Branding
st.sidebar.image("logo.png", use_column_width=True)  # Add your SPICE.AI logo
st.sidebar.title("SPICE.AI")
st.sidebar.write("Solar Panel Inspection & Classification Engine")
st.sidebar.markdown("---")
st.sidebar.write("🔍 **How It Works**")
st.sidebar.write("1️⃣ Upload an image of a solar panel")
st.sidebar.write("2️⃣ The AI will classify its condition")
st.sidebar.write("3️⃣ You get an **inspection score (0-100)**")
st.sidebar.write("4️⃣ Cleaning & maintenance suggestions provided")

# Ensure correct import from models directory
sys.path.append(os.path.abspath("D:/Projects/SPICE.AI/"))
from models.model import SolarPanelClassifier

# Load Model
MODEL_PATH = "D:/Projects/SPICE.AI/models/model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6

# Class Names
CLASS_NAMES = [
    "Clean",
    "Bird Drop",
    "Dusty",
    "Snow-Covered",
    "Electrical Damage",
    "Physical Damage"
]

# Cleaning Suggestions
CLEANING_SUGGESTIONS = {
    "Bird Drop": "🦜 Use mild detergent and water to remove bird droppings.",
    "Dusty": "💨 Regularly clean with a soft brush or water spray.",
    "Snow-Covered": "❄ Use a soft broom or heater-based removal.",
    "Electrical Damage": "⚡ Inspect wiring and consult a technician.",
    "Physical Damage": "🔧 Replace damaged panels or consult a professional."
}

# Load Model
@st.cache_resource
def load_model():
    model = SolarPanelClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to calculate inspection score
def get_inspection_score(pred_class, confidence):
    if pred_class == "Clean":
        return 90 + (confidence * 10)  # High score for clean panels
    return max(30, 80 - (confidence * 50))  # Score between 30-80

# Main UI
st.markdown("<h1 style='text-align: center; color: #FFD700;'>🌞 SPICE.AI - Solar Panel Inspection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #AAAAAA;'>AI-powered solar panel classification & maintenance guidance</h4>", unsafe_allow_html=True)
st.write("---")

# File Upload
uploaded_file = st.file_uploader("📤 Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Make Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        confidences = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(confidences).item()
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = confidences[0, predicted_idx].item()

    # Generate Inspection Score
    score = get_inspection_score(predicted_class, confidence)

    # Get Cleaning Suggestion
    suggestion = CLEANING_SUGGESTIONS.get(predicted_class, "✅ No cleaning required!")

    # Display Results
    st.markdown("### 🔍 Inspection Results:")
    st.markdown(f"<h3 style='color:#FFD700;'>Classification: {predicted_class}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4>Confidence: {confidence:.2%}</h4>", unsafe_allow_html=True)

    # Dynamic Score Indicator
    if score > 85:
        color = "green"
        emoji = "✅"
    elif score > 50:
        color = "orange"
        emoji = "⚠"
    else:
        color = "red"
        emoji = "❌"

    st.markdown(f"<h2 style='color:{color};'>{emoji} Inspection Score: {score:.2f}/100</h2>", unsafe_allow_html=True)
    st.info(suggestion)

    # Alert for Maintenance
    if predicted_class == "Clean":
        st.success("✅ Your solar panel is in good condition!")
    else:
        st.warning("⚠ Your solar panel requires maintenance!")

st.write("---")
st.markdown("<h4 style='text-align: center;'>Developed by <b>Kushal</b> 🚀 | Powered by AI</h4>", unsafe_allow_html=True)
