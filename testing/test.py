import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model_path = os.path.join("model", "model.h5")
model = load_model(model_path)

# Load and preprocess test image
img_path = "C:/Users/kushal/Downloads/solar_panel_testing.jpg" 
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0] 
print(f"Predicted Class: {predicted_class}")
