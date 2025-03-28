from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model(r"D:\Projects\SPICE.AI\solar_panel_classifier.h5")

# Load and preprocess an image for testing
img_path = r"D:\Projects\SPICE.AI\test_image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict class
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

# Get class labels
class_labels = list(train_data.class_indices.keys())
print(f"Predicted Class: {class_labels[class_index]}")
