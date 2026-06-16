from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import gdown

app = Flask(__name__)

# Create models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "image_classifier_model.h5")

# Download model if not exists
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1Z2Kk7OaBYfrvwKtb6gEZbzT80w6rGQr1"
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        if not file:
            return jsonify({"error": "No file uploaded"})

        upload_folder = os.path.join(BASE_DIR, "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        result = class_labels[class_index]

        if result == 'notumor':
            label = "No Tumor"
        else:
            label = f"Tumor: {result}"

        return jsonify({
            "result": label,
            "confidence": f"{confidence*100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run()




