import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Set Page Config
st.set_page_config(
    page_title="SPICE.AI - Solar Panel Inspection",
    page_icon="🌞",
    layout="wide"
)

# Define Class Names
CLASS_NAMES = [
    "Clean", "Bird Drop", "Dusty", "Snow-Covered", "Electrical Damage", "Physical Damage"
]

# Define the Solar Panel Classifier Model
class SolarPanelClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SolarPanelClassifier, self).__init__()
        self.base = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# Load Model Function
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarPanelClassifier(num_classes=6).to(device)
    MODEL_PATH = "D:/Projects/SPICE.AI/models/model.pth"

    state_dict = torch.load(MODEL_PATH, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Handle DataParallel

    model.load_state_dict(state_dict)
    model.eval()
    return model, device

# Load Model
model, DEVICE = load_model()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Main Interface
st.markdown("<h1 style='text-align: center;'>🌞 SPICE.AI - Solar Panel Inspector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>AI-Powered Solar Panel Classification & Inspection</h4>", unsafe_allow_html=True)
st.write("---")

# Upload File
uploaded_file = st.file_uploader("📤 Upload a solar panel image for inspection:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        # Preprocess the image for prediction
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Make Prediction
        with torch.no_grad():
            output = model(tensor)
            predicted_idx = torch.argmax(output).item()

        # Get Class Name
        predicted_class = CLASS_NAMES[predicted_idx]

        # Display Prediction
        st.markdown(f"<h3 style='color: green;'>🔍 Predicted Condition: {predicted_class}</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Footer Information
st.write("---")
st.markdown("<h4 style='text-align: center;'>Developed by <b>Kushal</b> 🚀 | Powered by AI</h4>", unsafe_allow_html=True)
