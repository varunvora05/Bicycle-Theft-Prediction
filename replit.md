# Bicycle Theft Prediction

## Overview
A Flask web application with a machine learning model for predicting whether a stolen bicycle in Toronto will be recovered. Uses a pre-trained Decision Tree classifier with 411 features.

## Project Structure
- `app.py` - Main Flask application with prediction API
- `bike_best_model.pkl` - Pre-trained ML model
- `bike_features.pkl` - Feature names for the model
- `templates/index.html` - Web form interface
- `GP.py`, `Groupproject_COMP309.py` - Supporting project files

## Running the Application
The Flask server runs on `0.0.0.0:5000`.

## API Endpoints
- `GET /` - Home page with prediction form
- `POST /predict-form` - HTML form submission
- `POST /predict-json` - JSON API endpoint

## Dependencies
- Flask
- pandas
- scikit-learn
