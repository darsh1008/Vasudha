
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("crop_model.pkl")
encoders = joblib.load("encoders.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

@app.route("/")
def home():
    return "VASUDHA AI Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    sample = pd.DataFrame([data])

    sample["region"] = encoders["region"].transform(sample["region"])
    sample["month"] = encoders["month"].transform(sample["month"])
    sample["soil_type"] = encoders["soil_type"].transform(sample["soil_type"])
    sample["water_availability"] = encoders["water_availability"].transform(sample["water_availability"])

    prediction = model.predict(sample)

    crop = crop_encoder.inverse_transform(prediction)

    return jsonify({
        "crop": crop[0]
    })

if __name__ == "__main__":
    app.run()

