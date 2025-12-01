# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 17:34:26 2025

@author: HP
"""

# -*- coding: utf-8 -*-
"""
COMP309 - Bicycle Theft Prediction
Flask API + HTML Frontend (Production-Ready)
This API GUARANTEES predictions follow the requirements.
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# =======================
# LOAD MODEL + FEATURES
# =======================
BASE_DIR = r"C:\Users\HP\Downloads\COMP309_BicycleTheftProject"
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILE = os.path.join(MODEL_DIR, "bike_best_model.pkl")
FEATURES_FILE = os.path.join(MODEL_DIR, "bike_features.pkl")

model = pickle.load(open(MODEL_FILE, "rb"))
feature_names = pickle.load(open(FEATURES_FILE, "rb"))

print("Model loaded successfully.")
print("Feature count:", len(feature_names))

# =======================
# PREPROCESSOR FUNCTION
# =======================
def preprocess_input(data_dict):
    """
    Takes a dictionary of values from HTML form or JSON request.
    Converts it into:
      → pandas DataFrame
      → One-hot encoded
      → Reindexed to match training model columns
    """
    df_input = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_encoded

# =======================
# HOME PAGE (HTML FORM)
# =======================
@app.route("/")
def index():
    return render_template("index.html")

# =======================
# HTML FORM SUBMISSION
# =======================
@app.route("/predict-form", methods=["POST"])
def predict_form():

    # Get values from the form (all strings initially)
    form_data = {
        "OCC_YEAR": request.form.get("OCC_YEAR"),
        "OCC_MONTH": request.form.get("OCC_MONTH"),
        "OCC_DOW": request.form.get("OCC_DOW"),
        "OCC_HOUR": request.form.get("OCC_HOUR"),
        "DIVISION": request.form.get("DIVISION"),
        "BIKE_TYPE": request.form.get("BIKE_TYPE"),
        "BIKE_COLOUR": request.form.get("BIKE_COLOUR"),
        "BIKE_SPEED": request.form.get("BIKE_SPEED"),
        "LOCATION_TYPE": request.form.get("LOCATION_TYPE"),
        "BIKE_COST": request.form.get("BIKE_COST")
    }

    # Convert numeric fields
    cleaned = {}
    for key, value in form_data.items():
        if value == "" or value is None:
            continue
        if key in ["OCC_YEAR", "OCC_HOUR", "BIKE_SPEED"]:
            cleaned[key] = int(value)
        elif key in ["BIKE_COST"]:
            cleaned[key] = float(value)
        else:
            cleaned[key] = value

    if not cleaned:
        return render_template("index.html", error="Please enter at least one field.")

    # Preprocess the input to match model format
    X = preprocess_input(cleaned)

    # Predict
    prob = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]
    label = "Recovered" if pred == 1 else "Not Recovered"

    return render_template(
        "index.html",
        prediction=label,
        probability=round(prob, 4),
        form_values=form_data
    )

# =======================
# JSON API ENDPOINT
# =======================
@app.route("/predict-json", methods=["POST"])
def predict_json():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    try:
        # Preprocess JSON into encoded DataFrame
        X = preprocess_input(data)

        prob = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]
        label = "Recovered" if pred == 1 else "Not Recovered"

        return jsonify({
            "input": data,
            "prediction": int(pred),
            "prediction_label": label,
            "probability_recovered": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =======================
# RUN SERVER
# =======================
if __name__ == "__main__":
    app.run(debug=True)
