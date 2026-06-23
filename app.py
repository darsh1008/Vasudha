
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load AI model and encoders
model = joblib.load("crop_model.pkl")
encoders = joblib.load("encoders.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")


@app.route("/")
def home():
    return "VASUDHA AI Running"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        sample = pd.DataFrame([{
            "region": data["region"],
            "month": data["month"],
            "soil_type": data["soil_type"],
            "field_size_acres": float(data["field_size_acres"]),
            "water_availability": data["water_availability"],
            "temperature_c": float(data["temperature_c"]),
            "rainfall_mm": float(data["rainfall_mm"]),
            "soil_moisture_percent": float(data["soil_moisture_percent"])
        }])

        # Encode categorical columns
        sample["region"] = encoders["region"].transform(sample["region"])
        sample["month"] = encoders["month"].transform(sample["month"])
        sample["soil_type"] = encoders["soil_type"].transform(sample["soil_type"])
        sample["water_availability"] = encoders["water_availability"].transform(sample["water_availability"])

        # Predict crop
        prediction = model.predict(sample)

        crop = crop_encoder.inverse_transform(prediction)

        return jsonify({
            "success": True,
            "crop": crop[0]
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
