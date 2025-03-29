import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Load Model Function
def load_model(model_path, device):
    model = models.mobilenet_v3_large(weights=None)  # Initialize without pretrained weights
    num_ftrs = model.classifier[0].in_features  # Get the input features of the classifier
    
    # Define classifier to match saved model's structure
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(512, 9)  # 9 output classes
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)
    model.eval()
    return model

# Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Function to open images (JPEG, PNG, WebP)
def open_image(uploaded_file):
    # Use PIL for supported image formats (JPEG, PNG, WebP)
    image = Image.open(uploaded_file).convert("RGB")
    return image

# Predict Function
def predict(image, model, device, class_labels):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    predictions = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
    return {class_labels[i]: '✅' if predictions[i] else '❌' for i in range(len(class_labels))}

# Streamlit UI
def main():
    st.title("Solar Panel Inspection")
    st.write("Upload an image of a solar panel to check its characteristics.")
    
    # Load Model
    model_path = "D:/Projects/SPICE.AI/models/solar_panel_mobilenetv3_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    class_labels = ["Panel Detected", "Clean Panel", "Physical Damage", "Electrical Damage", 
                    "Snow Covered", "Water Obstruction", "Dust Contamination", 
                    "Bird Interference", "Leaf Obstruction"]
    
    # File Upload
    uploaded_file = st.file_uploader("Upload a solar panel image", type=["jpg", "png", "jpeg", "webp"])
    if uploaded_file is not None:
        image = open_image(uploaded_file)  # Open the image correctly
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess and Predict
        processed_image = preprocess_image(image)
        predictions = predict(processed_image, model, device, class_labels)
        
        # Display Results
        st.subheader("Inspection Results:")
        for label, status in predictions.items():
            st.write(f"**{label}:** {status}")

if __name__ == "__main__":
    main()
