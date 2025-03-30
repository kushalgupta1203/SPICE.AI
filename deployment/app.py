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

def display_total_score(inspection_score):
    st.markdown(f"## Total Inspection Score: {inspection_score:.1f}/100")  # Force 1 decimal

    # Add score classification with more granular thresholds
    if inspection_score >= 90:
        st.success("EXCELLENT CONDITION")
    elif inspection_score >= 80:
        st.success("VERY GOOD CONDITION")
    elif inspection_score >= 70:
        st.success("GOOD CONDITION")
    elif inspection_score >= 60:
        st.warning("ABOVE AVERAGE CONDITION")
    elif inspection_score >= 50:
        st.warning("FAIR CONDITION")
    elif inspection_score >= 40:
        st.warning("POOR CONDITION")
    elif inspection_score >= 30:
        st.error("VERY POOR CONDITION")
    else:
        st.error("CRITICAL CONDITION")

# Cleaning Suggestions Based on Scores
def cleaning_suggestions(predictions):
    suggestions = []
    # Check for evenly cleaned panel
    if predictions["Clean Panel"] > 70 and predictions["Physical Damage"] < 10:
        return ["No cleaning required. Panel appears to be in excellent condition."]

    if predictions["Physical Damage"] > 30:
        suggestions.append("Repair physical damage immediately")
    if predictions["Electrical Damage"] > 40:
        suggestions.append("Consult electrical engineer")
    if predictions["Snow Covered"] > 50:
        suggestions.append("Remove snow accumulation")
    if predictions["Water Obstruction"] > 40:
        suggestions.append("Clear water obstructions")
    if predictions["Foreign Particle Contamination"] > 40:
        suggestions.append("Clean foreign particles")
    if predictions["Bird Interference"] > 40:
        suggestions.append("Install bird deterrents")
    return suggestions or ["No critical issues detected"]

# Enhanced Get Weighted Value Based on Ranges with Smoother Transitions
def get_weighted_value(score, ranges):
    # Direct match within a range
    for min_val, max_val, weight in ranges:
        if min_val <= score < max_val:
            # Add smooth transition near range boundaries (within 5 points)
            if score - min_val < 5 and min_val > 0:
                # Transitioning from previous range
                prev_weight = 0
                for prev_min, prev_max, prev_w in ranges:
                    if prev_max == min_val:
                        prev_weight = prev_w
                        break

                # Calculate blend factor (0-1) for smooth transition
                blend = (score - min_val) / 5
                # Interpolate between previous weight and current weight
                interpolated_weight = prev_weight * (1 - blend) + weight * blend
                return interpolated_weight

            elif max_val - score < 5 and max_val < 101:
                # Transitioning to next range
                next_weight = 0
                for next_min, next_max, next_w in ranges:
                    if next_min == max_val:
                        next_weight = next_w
                        break

                # Calculate blend factor (0-1) for smooth transition
                blend = (max_val - score) / 5
                # Interpolate between current weight and next weight
                interpolated_weight = weight * blend + next_weight * (1 - blend)
                return interpolated_weight

            # Standard case - in the middle of a range
            return weight
    
    # Default fallback - should rarely occur if ranges are properly defined
    return 0

# Compute Inspection Score with Enhanced Weighting and Variation
def compute_inspection_score(predictions):
    total = 0
    weighted_scores = {}

    # Calculate base weighted score with non-linear scaling
    for label, score in predictions.items():
        if label == "Panel Detected":
            continue  # Skip Panel Detected label

        config = CLASS_CONFIG[label]
        weighted_value = get_weighted_value(score, config["ranges"])

        # Apply importance multiplier based on label
        if label == "Physical Damage" or label == "Electrical Damage":
            importance_factor = 1.3
        elif label == "Clean Panel":
            importance_factor = 1.2
        else:
            importance_factor = 1.0

        weighted_value *= importance_factor
        weighted_scores[label] = weighted_value * score
        total += weighted_value * score

    # Calculate max possible score more accurately
    max_possible = sum(100 * max(w for _, _, w in cfg["ranges"]) *
                      (1.3 if label in ["Physical Damage", "Electrical Damage"] else
                       1.2 if label == "Clean Panel" else 1.0)
                      for label, cfg in CLASS_CONFIG.items() if label != "Panel Detected")

    # Normalize with better scaling
    normalized = (total / abs(max_possible)) * 100 if max_possible != 0 else 0

    # Non-linear transformation (S-curve)
    x = normalized / 100.0
    transformed = 0.5 * (np.tanh(3 * (x - 0.5)) + 1)  # S-curve
    final_score = transformed * 100
    final_score = np.clip(final_score, 0, 100)

    return final_score

# Scoring Configuration for Each Class
CLASS_CONFIG = {
    "Panel Detected": {
        "ranges": [(0, 10, 0), (10, 20, 0), (20, 30, 0.05), (30, 40, 0.1), (40, 50, 0.2), (50, 60, 0.6), (60, 70, 0.6), (70, 80, 0.7), (80, 90, 0.7), (90, 101, 0.7)]
    },
    "Clean Panel": {
        "ranges": [(0, 10, 0.05), (10, 20, 0.05), (20, 30, 0.1), (30, 40, 0.15), (40, 50, 0.2), (50, 60, 0.2), (60, 70, 0.25), (70, 80, 0.25), (80, 90, 0.25), (90, 101, 0.3)]
    },
    "Physical Damage": {
        "ranges": [(0, 10, -0.05), (10, 20, -0.05), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.45), (70, 80, -0.45), (80, 90, -0.55), (90, 101, -0.55)]
    },
    "Electrical Damage": {
        "ranges": [(0, 10, -0.05), (10, 20, -0.05), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.45), (70, 80, -0.45), (80, 90, -0.55), (90, 101, -0.55)]
    },
    "Snow Covered": {
        "ranges": [(0, 10, -0.05), (10, 20, -0.05), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.45), (70, 80, -0.45), (80, 90, -0.55), (90, 101, -0.55)]
    },
    "Water Obstruction": {
        "ranges": [(0, 10, 0.05), (10, 20, -0.05), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.45), (70, 80, -0.45), (80, 90, -0.55), (90, 101, -0.55)]
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 10, -0.05), (10, 20, -0.05), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.45), (70, 80, -0.45), (80, 90, -0.55), (90, 101, -0.55)]
    },
    "Bird Interference": {
        "ranges": [(0, 10, -0.05), (10, 20, -0.05), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.45), (70, 80, -0.45), (80, 90, -0.55), (90, 101, -0.55)]
    }
}

# Function to print label analysis to the terminal
def print_label_analysis(predictions):
    print("--- Label Analysis ---")
    for label, score in predictions.items():
        print(f"{label}: {score:.1f}%")
    print("----------------------")

# Main Function for Streamlit App
def main():
    st.set_page_config(page_title="SPICE.AI: Solar Panel Inspection", layout="wide")

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

    # Tabs for Different Sections
    tabs = st.tabs(["How to Use", "Upload Image", "Total Score", "Label Analysis", "Outcome"])

    with tabs[0]:
        st.header("User Guide")

        st.markdown("""
            Follow these steps to analyze your solar panel image:
            - **Upload an Image**: Go to the 'Upload Image' tab and upload a clear image of your solar panel.
            - **Preview**: After uploading the image successfully you will see a preview of it.
            - **Analysis**: The system will evaluate cleanliness and detect potential issues such as physical damage or obstructions.
            - **Scores**: The system provides an overall inspection score along with detailed label analysis.
            """)

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
            with tabs[1]:
                st.header("Upload Image")

        uploaded_file = st.file_uploader("Upload solar panel image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            # Open the image and store it in the session state
            image = open_image(uploaded_file)
            if image is not None:
                st.session_state.image = image
                # Display a compressed version of the image
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
                    # Clear other session states to prevent further execution
                    for key in list(st.session_state.keys()):
                        if key not in ['panel_detection_model', 'image']:
                            del st.session_state[key]
                    st.stop()  # Stop execution here

                else:
                    st.success("Image uploaded. Check the 'Total Score' and 'Label Analysis' tabs.")
                    st.session_state.panel_predictions = panel_predictions # store in session state

            else:
                st.error("Panel Detection model failed to load.")

    #Moved here
    with tabs[3]:
        st.header("Label Analysis")
        if 'image_tensor' in st.session_state and \
           st.session_state.inspection_model_v11 is not None and \
           st.session_state.inspection_model_v20 is not None:

            st.subheader("Label Predictions - Model v1.1")
            predictions_v11 = predict(st.session_state.image_tensor, st.session_state.inspection_model_v11, device)
            print_label_analysis(predictions_v11)
            for label, score in predictions_v11.items():
                st.text(f"{label}: {score:.1f}%")

            st.subheader("Label Predictions - Model v2.0")
            predictions_v20 = predict(st.session_state.image_tensor, st.session_state.inspection_model_v20, device)
            print_label_analysis(predictions_v20)
            for label, score in predictions_v20.items():
                st.text(f"{label}: {score:.1f}%")

            st.session_state.predictions_v11 = predictions_v11
            st.session_state.predictions_v20 = predictions_v20
        else:
            st.warning("Upload an image and ensure models are loaded to see label analysis.")

    with tabs[2]:
        st.header("Total Score")
        if 'predictions_v11' in st.session_state and 'predictions_v20' in st.session_state:
            st.subheader("Inspection Score - Model v1.1")
            inspection_score_v11 = compute_inspection_score(st.session_state.predictions_v11)
            display_total_score(inspection_score_v11)

            st.subheader("Inspection Score - Model v2.0")
            inspection_score_v20 = compute_inspection_score(st.session_state.predictions_v20)
            display_total_score(inspection_score_v20)
        else:
            st.warning("Label analysis needs to be done first.")

    with tabs[4]:
        st.header("Outcome")
        if 'predictions_v11' in st.session_state and 'predictions_v20' in st.session_state:
            st.subheader("Cleaning Suggestions - Model v1.1")
            suggestions_v11 = cleaning_suggestions(st.session_state.predictions_v11)
            for suggestion in suggestions_v11:
                st.markdown(f"- {suggestion}")

            st.subheader("Cleaning Suggestions - Model v2.0")
            suggestions_v20 = cleaning_suggestions(st.session_state.predictions_v20)
            for suggestion in suggestions_v20:
                st.markdown(f"- {suggestion}")
        else:
            st.warning("Generate total score first to see outcome.")

if __name__ == "__main__":
    main()
