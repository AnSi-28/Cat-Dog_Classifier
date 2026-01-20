import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = 32
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
svm = joblib.load("models/svm_model.pkl")
rf = joblib.load("models/rf_model.pkl")
lr = joblib.load("models/lr_model.pkl")
kmeans = joblib.load("models/kmeans_model.pkl")

models = {
    "svm": svm,
    "rf": rf,
    "lr": lr,
    "kmeans": kmeans
}

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()
    img = img / 255.0
    return img.reshape(1, -1)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    model_name = request.form["model"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    model = models[model_name]

    if model_name == "kmeans":
        cluster = model.predict(img)[0]
        # Simple assumption: cluster 1 = Dog
        prediction = "Dog" if cluster == 1 else "Cat"
    else:
        pred = model.predict(img)[0]
        prediction = "Dog" if pred == 1 else "Cat"

    return render_template(
        "index.html",
        prediction=prediction,
        model_used=model_name.upper(),
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)
