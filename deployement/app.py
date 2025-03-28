from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model("../model/model.h5")
class_labels = ['Clean', 'Dusty', 'Bird Drop', 'Snow Covered', 'Electrical Damage', 'Physical Damage']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return jsonify({"Predicted Class": class_labels[predicted_class]})

if __name__ == '__main__':
    app.run(debug=True)
