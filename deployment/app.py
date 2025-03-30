import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Load Model Function
def load_model(model_path, device):
    try:
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[0].in_features
        
        # Adjust classifier to match checkpoint dimensions
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1280),  # Match checkpoint dimensions
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, 8)  # Match final output dimensions
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
    return Image.open(uploaded_file).convert("RGB")

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

# Compute Inspection Score with Weighted Ranges
def compute_inspection_score(predictions):
    total = 0
    weighted_scores = {}
    
    # Add a small randomization factor (±3%) for similar cases
    randomization = np.random.uniform(-3, 3)
    
    for label, score in predictions.items():
        config = CLASS_CONFIG[label]
        weighted_value = get_weighted_value(score, config["ranges"])
        weighted_scores[label] = weighted_value
        total += weighted_value
    
    max_possible = sum(max(w for _, _, w in cfg["ranges"]) * 100 
                      for cfg in CLASS_CONFIG.values())
    normalized = (total / abs(max_possible)) * 100 if max_possible != 0 else 0
    
    # Apply adjustments for score variance and easier interpretation
    normalized = normalized * 1.2  # Increase sensitivity
    
    # Add randomization factor for similar inputs
    normalized += randomization
    
    normalized = np.clip(normalized, 0, 100)  # Ensure score stays within 0-100
    
    # Apply a more aggressive sigmoid function for better score distribution
    final_score = 100 * (1 / (1 + np.exp(-normalized / 15 + 3)))  # Tunable parameters
    
    # Ensure exactly 1 decimal place
    final_score = round(final_score, 1)
    
    return final_score, weighted_scores

def format_score(score):
    return f"{score:.1f}"

# Get Weighted Value Based on Ranges
def get_weighted_value(score, ranges):
    for min_val, max_val, weight in ranges:
        if min_val <= score < max_val:
            return score * weight
    return 0  # Default to 0 if no range matches

# Cleaning Suggestions Based on Scores
def cleaning_suggestions(predictions):
    suggestions = []
    
    # Check for evenly cleaned panel
    if predictions["Clean Panel"] > 70 and predictions["Physical Damage"] < 10:
        return ["No cleaning required. Panel appears to be in excellent condition."]
    
    if predictions["Physical Damage"] > 20: 
        suggestions.append("Repair physical damage immediately")
    if predictions["Electrical Damage"] > 20: 
        suggestions.append("Consult electrical engineer")
    if predictions["Snow Covered"] > 20: 
        suggestions.append("Remove snow accumulation")
    if predictions["Water Obstruction"] > 20: 
        suggestions.append("Clear water obstructions")
    if predictions["Foreign Particle Contamination"] > 20: 
        suggestions.append("Clean foreign particles")
    if predictions["Bird Interference"] > 20: 
        suggestions.append("Install bird deterrents")
    return suggestions or ["No critical issues detected"]

# Enhanced efficiency impact calculation
def calculate_efficiency_impact(inspection_score, predictions):
    # Base impact from overall score
    base_impact = max(0, 100 - inspection_score) * 0.3
    
    # Additional impact from specific issues
    additional_impact = 0
    
    # Physical and electrical damage have the highest impact
    if predictions["Physical Damage"] > 30:
        additional_impact += predictions["Physical Damage"] * 0.25
    elif predictions["Physical Damage"] > 10:
        additional_impact += predictions["Physical Damage"] * 0.1
        
    if predictions["Electrical Damage"] > 30:
        additional_impact += predictions["Electrical Damage"] * 0.3
    elif predictions["Electrical Damage"] > 10:
        additional_impact += predictions["Electrical Damage"] * 0.15
    
    # Obstruction impacts
    if predictions["Snow Covered"] > 40:
        additional_impact += predictions["Snow Covered"] * 0.2
    elif predictions["Snow Covered"] > 20:
        additional_impact += predictions["Snow Covered"] * 0.1
    
    if predictions["Water Obstruction"] > 40:
        additional_impact += predictions["Water Obstruction"] * 0.15
    elif predictions["Water Obstruction"] > 20:
        additional_impact += predictions["Water Obstruction"] * 0.08
    
    # Contamination impacts
    if predictions["Foreign Particle Contamination"] > 40:
        additional_impact += predictions["Foreign Particle Contamination"] * 0.1
    elif predictions["Foreign Particle Contamination"] > 20:
        additional_impact += predictions["Foreign Particle Contamination"] * 0.05
    
    if predictions["Bird Interference"] > 40:
        additional_impact += predictions["Bird Interference"] * 0.1
    elif predictions["Bird Interference"] > 20:
        additional_impact += predictions["Bird Interference"] * 0.05
    
    # Calculate total impact (capped at 100%)
    total_impact = min(100, base_impact + additional_impact)
    
    # Calculate actual efficiency
    actual_efficiency = max(0, 100 - total_impact)
    
    return {
        "total_impact": round(total_impact, 2),
        "actual_efficiency": round(actual_efficiency, 2),
        "base_impact": round(base_impact, 2),
        "additional_impact": round(additional_impact, 2),
        "breakdown": {
            "physical_damage": round(predictions["Physical Damage"] * 0.25 if predictions["Physical Damage"] > 30 else 
                               predictions["Physical Damage"] * 0.1 if predictions["Physical Damage"] > 10 else 0, 2),
            "electrical_damage": round(predictions["Electrical Damage"] * 0.3 if predictions["Electrical Damage"] > 30 else 
                                predictions["Electrical Damage"] * 0.15 if predictions["Electrical Damage"] > 10 else 0, 2),
            "snow_covered": round(predictions["Snow Covered"] * 0.2 if predictions["Snow Covered"] > 40 else 
                             predictions["Snow Covered"] * 0.1 if predictions["Snow Covered"] > 20 else 0, 2),
            "water_obstruction": round(predictions["Water Obstruction"] * 0.15 if predictions["Water Obstruction"] > 40 else 
                                 predictions["Water Obstruction"] * 0.08 if predictions["Water Obstruction"] > 20 else 0, 2),
            "contamination": round(predictions["Foreign Particle Contamination"] * 0.1 if predictions["Foreign Particle Contamination"] > 40 else 
                              predictions["Foreign Particle Contamination"] * 0.05 if predictions["Foreign Particle Contamination"] > 20 else 0, 2),
            "bird_interference": round(predictions["Bird Interference"] * 0.1 if predictions["Bird Interference"] > 40 else 
                                 predictions["Bird Interference"] * 0.05 if predictions["Bird Interference"] > 20 else 0, 2)
        }
    }

# Scoring Configuration for Each Class
CLASS_CONFIG = {
    "Panel Detected": {
        "ranges": [(0, 20, 0.5), (20, 40, 0.8), (40, 101, 1.2)],
        "description": "Detection confidence level"
    },
    "Clean Panel": {
        "ranges": [(0, 20, 0.1), (20, 40, 0.3), (40, 70, 0.6), (70, 101, 1.0)],
        "description": "Cleanliness assessment"
    },
    "Physical Damage": {
        "ranges": [(0, 10, -0.2), (10, 30, -0.5), (30, 101, -1.5)],
        "description": "Structural integrity evaluation"
    },
    "Electrical Damage": {
        "ranges": [(0, 10, -0.2), (10, 30, -0.5), (30, 101, -1.5)],
        "description": "Electrical safety evaluation"
    },
    "Snow Covered": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 101, -1.0)],
        "description": "Snow coverage impact"
    },
    "Water Obstruction": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 101, -1.0)],
        "description": "Water obstruction impact"
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 101, -1.5)],
        "description": "Impact of foreign particles"
    },
    "Bird Interference": {
        "ranges": [(0, 20, -0.1), (20, 40, -0.3), (40, 101, -1.5)],
        "description": "Impact of bird interference"
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
        logo = Image.open("https://github.com/kushalgupta1203/SPICE.AI/blob/main/deployment/logo.png?raw=true")
        st.image(logo, use_container_width=True)  # Updated to use_container_width
    except FileNotFoundError:
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
    if 'model' not in st.session_state:
        try:
            model_path = r"D:\Projects\SPICE.AI\models\spice_ai_mobilenetv3_v1.0.pth"
            st.session_state.model = load_model(model_path, device)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.session_state.model = None
    
    with tabs[1]:
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader("Upload solar panel image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Preview uploaded image in compressed size
            image = open_image(uploaded_file)
            display_compressed_image(image)

            # Process and predict when an image is uploaded
            try:
                if st.session_state.model is not None:
                    tensor = preprocess_image(image)
                    # Store in session state
                    st.session_state.image = image
                    st.session_state.tensor = tensor
                    st.session_state.predictions = predict(tensor, st.session_state.model, device)
                    # Check Panel Detected Score
                    panel_detected_score = st.session_state.predictions.get("Panel Detected", 0)
                    if panel_detected_score < 10:
                        st.error("No panel detected! Please re-upload.")
                    else:
                        # Calculate inspection score and store in session state
                        final_score, weighted_scores = compute_inspection_score(st.session_state.predictions)
                        st.session_state.inspection_score = final_score

                        # Print label analysis to the terminal
                        print_label_analysis(st.session_state.predictions)

                        st.success("Inspection score generated! Check it out in the 'Total Score' tab.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    
    with tabs[2]:
        st.header("Total Score")
        
        if 'inspection_score' in st.session_state and 'predictions' in st.session_state:
            inspection_score = st.session_state.inspection_score
            
            st.markdown(f"## Total Inspection Score: {inspection_score:.1f}/100")
            
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
        else:
            st.write("Upload an image to get the total score.")
            
    with tabs[3]:
        st.header("Label Analysis")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions

            # Sort issues by confidence score in decreasing order
            issues = sorted(((label, score) for label, score in predictions.items() 
                             if label not in ['Clean Panel', 'Panel Detected'] and score > 30),
                            key=lambda x: x[1], reverse=True)

            if issues:
                st.write("Identified Issues:")
                for label, score in issues:
                    st.write(f"- {label}: {score:.1f}%")
            else:
                st.info("No significant issues detected based on the confidence scores.")
        else:
            st.write("Upload an image to see the label analysis.")
    
    with tabs[4]:
        st.header("Outcome")

        if 'predictions' in st.session_state and 'inspection_score' in st.session_state:
            predictions = st.session_state.predictions
            inspection_score = st.session_state.inspection_score

            # Display cleaning suggestions
            st.subheader("Cleaning Suggestions")
            suggestions = cleaning_suggestions(predictions)
            for suggestion in suggestions:
                st.write(f"- {suggestion}")

            # Calculate and display efficiency impact
            st.subheader("Efficiency Impact")
            efficiency_results = calculate_efficiency_impact(inspection_score, predictions)
            st.write(f"Total Efficiency Impact: {efficiency_results['total_impact']}%")
            st.write(f"Estimated Actual Efficiency: {efficiency_results['actual_efficiency']}%")
            st.write(f"Base Impact (from overall score): {efficiency_results['base_impact']}%")
            st.write(f"Additional Impact (from specific issues): {efficiency_results['additional_impact']}%")

            # Display efficiency impact breakdown
            st.write("Efficiency Impact Breakdown:")
            breakdown = efficiency_results['breakdown']
            for issue, impact in breakdown.items():
                st.write(f"- {issue.replace('_', ' ').title()}: {impact}%")
        else:
            st.write("Upload an image and run the analysis to see the outcome.")

# Run the main function
if __name__ == "__main__":
    main()
