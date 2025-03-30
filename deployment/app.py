import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import torch.nn as nn
import torchvision.models as models
import numpy as np
import requests

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

def display_total_score(inspection_score):
    st.markdown(f"## Total Inspection Score: {inspection_score:.1f}/100")  # Force 1 decimal

    # Add score classification with more granular thresholds
    if inspection_score >= 85:
        st.success("EXCELLENT CONDITION")
    elif inspection_score >= 70:
        st.success("GOOD CONDITION")
    elif inspection_score >= 55:
        st.warning("FAIR CONDITION")
    elif inspection_score >= 35:
        st.warning("POOR CONDITION")
    else:
        st.error("CRITICAL CONDITION")

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
                return score * interpolated_weight
            
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
                return score * interpolated_weight
            
            # Standard case - in the middle of a range
            return score * weight
    
    # Default fallback - should rarely occur if ranges are properly defined
    return 0

# Compute Inspection Score with Enhanced Weighting and Variation
def compute_inspection_score(predictions):
    total = 0
    weighted_scores = {}
    
    # Enhanced randomization with contextual factors
    base_randomization = np.random.uniform(-3, 3)
    
    # Get top 3 issues for contextual adjustment
    sorted_issues = sorted([(label, score) for label, score in predictions.items() 
                           if label != "Clean Panel" and label != "Panel Detected"], 
                          key=lambda x: x[1], reverse=True)[:3]
    
    # Create issue-specific adjustments
    issue_adjustment = 0
    for label, score in sorted_issues:
        if score > 50:  # Significant issue detected
            # Different issues have different impacts on final score
            if label in ["Physical Damage", "Electrical Damage"]:
                issue_adjustment -= (score / 100) * np.random.uniform(3, 7)
            elif label in ["Snow Covered", "Water Obstruction"]:
                issue_adjustment -= (score / 100) * np.random.uniform(2, 5)
            else:
                issue_adjustment -= (score / 100) * np.random.uniform(1, 4)
    
    # Cleanliness bonus - if clean but has minor issues
    if predictions["Clean Panel"] > 70 and all(score < 30 for label, score in sorted_issues):
        issue_adjustment += np.random.uniform(1, 5)
    
    # Calculate base weighted score with non-linear scaling
    for label, score in predictions.items():
        config = CLASS_CONFIG[label]
        # Add small per-label variance to prevent identical scores for similar inputs
        adjusted_score = score * (1 + np.random.uniform(-0.05, 0.05))
        weighted_value = get_weighted_value(adjusted_score, config["ranges"])
        
        # Apply importance multiplier based on label
        if label == "Physical Damage" or label == "Electrical Damage":
            importance_factor = 1.3
        elif label == "Clean Panel":
            importance_factor = 1.2
        else:
            importance_factor = 1.0
            
        weighted_value *= importance_factor
        weighted_scores[label] = weighted_value
        total += weighted_value

    # Calculate max possible score more accurately
    max_possible = sum(max(w for _, _, w in cfg["ranges"]) * 100 * 
                      (1.3 if label in ["Physical Damage", "Electrical Damage"] else 
                       1.2 if label == "Clean Panel" else 1.0)
                      for label, cfg in CLASS_CONFIG.items())
    
    # Normalize with better scaling
    normalized = (total / abs(max_possible)) * 100 if max_possible != 0 else 0
    
    # Apply non-linear transformation for more diverse scoring
    # Using a combination of sigmoid and polynomial functions for better distribution
    x = normalized / 100.0  # Scale to 0-1 range
    
    # Apply S-curve with adjustable steepness to create more distinct score categories
    if x < 0.3:
        # Steeper penalty for low scores
        transformed = x * 0.7
    elif x < 0.6:
        # Middle range with moderate scaling
        transformed = 0.21 + (x - 0.3) * 1.0
    else:
        # Upper range with diminishing returns
        transformed = 0.51 + (x - 0.6) * 1.2
    
    # Scale back to 0-100
    final_score = transformed * 100
    
    # Apply contextual adjustments and randomization
    final_score += base_randomization + issue_adjustment
    
    # Ensure boundaries
    final_score = np.clip(final_score, 0, 100)
    
    # Apply slight clustering around common scores to create score "bands"
    # This makes results feel more consistent for similar panels
    band_centers = [15, 35, 55, 75, 90]
    band_pull = 0
    for center in band_centers:
        distance = abs(final_score - center)
        if distance < 10:
            # Pull slightly toward band centers
            pull_strength = (10 - distance) * 0.15
            band_pull = pull_strength * np.sign(center - final_score)
    
    final_score += band_pull
    
    # Final boundary check and rounding
    final_score = np.clip(final_score, 0, 100)
    final_score = round(final_score, 1)
    
    return final_score, weighted_scores

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

# Scoring Configuration for Each Class
CLASS_CONFIG = {
    "Panel Detected": {
        "ranges": [(0, 10, 0), (10, 20, 0), (20, 30, 0.1), (30, 40, 0.2), (40, 50, 0.3), (50, 60, 0.4), (60, 70, 0.5), (70, 80, 0.6), (80, 90, 0.7), (90, 101, 0.8)]
    },
    "Clean Panel": {
        "ranges": [(0, 10, 0.05), (10, 20, 0.1), (20, 30, 0.2), (30, 40, 0.3), (40, 50, 0.4), (50, 60, 0.5), (60, 70, 0.6), (70, 80, 0.7), (80, 90, 0.8), (90, 101, 1.0)]
    },
    "Physical Damage": {
        "ranges": [(0, 10, 0.1), (10, 20, -0.1), (20, 30, -0.2), (30, 40, -0.3), (40, 50, -0.4), (50, 60, -0.5), (60, 70, -0.7), (70, 80, -0.8), (80, 90, -1.0), (90, 101, -1.2)]
    },
    "Electrical Damage": {
        "ranges": [(0, 10, 0.1), (10, 20, 0), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.5), (70, 80, -0.6), (80, 90, -0.7), (90, 101, -0.8)]
    },
    "Snow Covered": {
        "ranges": [(0, 10, 0.1), (10, 20, 0), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.5), (70, 80, -0.6), (80, 90, -0.7), (90, 101, -0.8)]
    },
    "Water Obstruction": {
        "ranges": [(0, 10, 0.1), (10, 20, 0), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.5), (70, 80, -0.6), (80, 90, -0.7), (90, 101, -0.8)]
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 10, 0.1), (10, 20, 0), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.5), (70, 80, -0.6), (80, 90, -0.7), (90, 101, -0.8)]
    },
    "Bird Interference": {
        "ranges": [(0, 10, 0.1), (10, 20, 0), (20, 30, -0.1), (30, 40, -0.2), (40, 50, -0.3), (50, 60, -0.4), (60, 70, -0.5), (70, 80, -0.6), (80, 90, -0.7), (90, 101, -0.8)]
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

    # Title and Current Date/Time Display
    try:
        # Use requests to fetch the image from the URL
        response = requests.get("https://github.com/kushalgupta1203/SPICE.AI/blob/main/deployment/logo.png?raw=true", stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Open the image using PIL
        logo = Image.open(BytesIO(response.content))
        st.image(logo, use_container_width=True)  # Updated to use_container_width
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching logo: {e}")
        st.title("SPICE.AI: Solar Panel Inspection & Classification Engine")
        st.warning("Logo not found.  Using default title.")
    except Exception as e:
        st.error(f"Error opening logo: {e}")
        st.title("SPICE.AI: Solar Panel Inspection & Classification Engine")
        st.warning("Logo not found.  Using default title.")

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
            panel_model_path = r"D:\Projects\SPICE.AI\models\spice_ai_mobilenetv3_v2.0.pth"
            # Load the panel detection model (v2.0) with 8 classes, as it predicts "Panel Detected" along with others
            st.session_state.panel_detection_model = load_model(panel_model_path, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading panel detection model: {e}")
            st.session_state.panel_detection_model = None

    if 'inspection_model' not in st.session_state:
        try:
            inspection_model_path = r"D:\Projects\SPICE.AI\models\spice_ai_mobilenetv3_v1.1.pth"
            # Load the inspection model (v1.1) also with 8 classes
            st.session_state.inspection_model = load_model(inspection_model_path, device, num_classes=8)
        except Exception as e:
            st.error(f"Error loading inspection model: {e}")
            st.session_state.inspection_model = None

    with tabs[1]:
        st.header("Upload Image")

        uploaded_file = st.file_uploader("Upload solar panel image", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_file is not None:
            # Preview uploaded image in compressed size
            image = open_image(uploaded_file)
            if image is not None:
                display_compressed_image(image)

                # Process and predict when an image is uploaded
                st.success("Image uploaded successfully!  Check the score.")

                try:
                    if st.session_state.panel_detection_model is not None and st.session_state.inspection_model is not None:
                        tensor = preprocess_image(image)
                        # Store in session state
                        st.session_state.image = image
                        st.session_state.tensor = tensor

                        # Use the panel detection model to get the initial predictions
                        panel_detection_predictions = predict(tensor, st.session_state.panel_detection_model, device)

                        # Use the inspection model to get the alternative predictions
                        inspection_predictions = predict(tensor, st.session_state.inspection_model, device)

                        # Model Selection Logic
                        final_predictions = {}

                        # 1. Panel Detection (Model v2.0)
                        final_predictions["Panel Detected"] = panel_detection_predictions["Panel Detected"]

                        # 2. Clean Panel (Higher Value between Model v1.1 and v2.0)
                        final_predictions["Clean Panel"] = max(inspection_predictions["Clean Panel"], panel_detection_predictions["Clean Panel"])

                        # 3. Physical Damage, Electrical Damage, Snow Covered, Water Obstruction, Foreign Particle Contamination, Bird Interference (Lower Value between Model v1.1 and v2.0)
                        damage_categories = ["Physical Damage", "Electrical Damage", "Snow Covered", "Water Obstruction", "Foreign Particle Contamination", "Bird Interference"]
                        for category in damage_categories:
                            final_predictions[category] = min(inspection_predictions[category], panel_detection_predictions[category])

                        # Store final predictions in session state
                        st.session_state.predictions = final_predictions

                        # Store the individual model predictions for debugging/comparison purposes
                        st.session_state.panel_detection_predictions = panel_detection_predictions
                        st.session_state.inspection_predictions = inspection_predictions

                        # Enable other tabs after processing the image
                        st.session_state.analysis_done = True

                    else:
                        st.warning("One or more models failed to load. Please check the logs.")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.session_state.analysis_done = False

    # Check if analysis is done before showing other tabs
    if 'analysis_done' in st.session_state and st.session_state.analysis_done:

        with tabs[2]:
            st.header("Total Score")
            if 'predictions' in st.session_state:
                # Calculate the inspection score
                inspection_score, weighted_scores = compute_inspection_score(st.session_state.predictions)
                display_total_score(inspection_score)
            else:
                st.write("No image processed yet. Please upload an image in the 'Upload Image' tab.")

        with tabs[3]:
            st.header("Label Analysis")
            if 'predictions' in st.session_state:
                # Print label analysis to the terminal (for debugging)
                print_label_analysis(st.session_state.predictions)

                # Show the predictions and weighted scores
                st.subheader("Predictions:")
                for label, score in st.session_state.predictions.items():
                    st.write(f"{label}: {score:.1f}%")

                if 'inspection_score' in locals():  # Access the inspection_score within the same scope
                    st.subheader("Weighted Scores:")
                # Use the weighted_scores obtained during score calculation
                    for label, weighted_value in weighted_scores.items():
                        st.write(f"{label}: {weighted_value:.2f}")  # Display weighted scores
            else:
                st.write("No image processed yet. Please upload an image in the 'Upload Image' tab.")

        with tabs[4]:
            st.header("Outcome")
            if 'predictions' in st.session_state:
                # Show cleaning suggestions
                st.subheader("Cleaning Suggestions:")
                suggestions = cleaning_suggestions(st.session_state.predictions)
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
            else:
                st.write("No image processed yet. Please upload an image in the 'Upload Image' tab.")
    else:
        # Optionally display a message when the analysis hasn't been done yet
        if 'analysis_done' in st.session_state and not st.session_state.analysis_done:
            st.info("Please upload an image in the 'Upload Image' tab to see the analysis.")

# Run the main function
if __name__ == "__main__":
    main()
