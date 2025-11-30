# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 15:02:45 2025

@author: HP
"""
# app.py  - FINAL VERSION FOR DECISION TREE MODEL

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# ============================
# Load Model + Feature List
# ============================

DATA_PATH = r"C:\Users\HP\Downloads"

MODEL_FILE = os.path.join(DATA_PATH, "bike_best_model.pkl")
FEATURES_FILE = os.path.join(DATA_PATH, "bike_features.pkl")

# Load trained Decision Tree model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Load feature names (from one-hot encoding)
with open(FEATURES_FILE, "rb") as f:
    feature_names = pickle.load(f)

# ============================
# Create Flask App
# ============================

app = Flask(__name__)

@app.route("/")
def home():
    return "🚲 Bike Theft Prediction API is running!"

# ============================
# Prediction Endpoint
# ============================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON format"}), 400

        # Create a DataFrame full of zeros for all feature columns
        x_input = pd.DataFrame(columns=feature_names)
        x_input.loc[0] = 0

        # Fill provided values
        for key, value in data.items():
            if key in x_input.columns:
                x_input.at[0, key] = value

        # Predict using the Decision Tree model
        prob = model.predict_proba(x_input)[0][1]
        pred = int(model.predict(x_input)[0])

        return jsonify({
            "prediction": pred,
            "probability_bike_recovered": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================
# Run API
# ============================

if __name__ == "__main__":
    app.run(debug=True)
