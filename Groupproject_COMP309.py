# -*- coding: utf-8 -*-
"""
COMP309 - Project #2
Toronto Bicycle Theft Prediction
FINAL STABLE VERSION (Decision Tree Only)
"""

#%% IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score
)
import pickle

plt.rcParams["figure.figsize"] = (8, 5)

#%% LOAD DATA
DATA_PATH = r"C:\Users\HP\Downloads"
FILENAME = "bicycle_thefts.csv.csv"  # <-- your file
fullpath = os.path.join(DATA_PATH, FILENAME)

df = pd.read_csv(fullpath)
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

#%% DROP HIGH-CARDINALITY COLUMNS
drop_cols = [
    "OBJECTID",
    "EVENT_UNIQUE_ID",
    "PRIMARY_OFFENCE",
    "OCC_DATE",
    "REPORT_DATE",
    "BIKE_MODEL",
    "BIKE_MAKE",
    "HOOD_140",
    "HOOD_158",
    "NEIGHBOURHOOD_140",
    "NEIGHBOURHOOD_158",
    "LONG_WGS84",
    "LAT_WGS84",
    "x",
    "y"
]

df_model = df.drop(columns=drop_cols, errors="ignore")
print("\nColumns kept:", df_model.columns.values)

#%% HANDLE MISSING VALUES
numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_model.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill numeric
df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

# Fill categorical
for col in cat_cols:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

print("\nMissing values after cleaning:")
print(df_model.isnull().sum())

#%% TARGET COLUMN
print("\nSTATUS value counts:")
print(df_model["STATUS"].value_counts())

def map_status(val):
    val = str(val).lower()
    if "recover" in val:
        return 1
    return 0

y = df_model["STATUS"].apply(map_status)

print("\nMapped targets (1=recovered, 0=not recovered):")
print(y.value_counts())

#%% FEATURES
X = df_model.drop(columns=["STATUS"])

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns.tolist()

print("\nShape after encoding:", X.shape)

#%% SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape, "Test:", X_test.shape)

#%% MODEL: DECISION TREE (NO SCALING NEEDED)
tree_clf = DecisionTreeClassifier(
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)
y_prob = tree_clf.predict_proba(X_test)[:, 1]

#%% RESULTS
print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label="Decision Tree")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#%% SAVE MODEL & FEATURES (NO SCALER REQUIRED)
MODEL_OUTFILE = os.path.join(DATA_PATH, "bike_best_model.pkl")
FEATURES_OUTFILE = os.path.join(DATA_PATH, "bike_features.pkl")

with open(MODEL_OUTFILE, "wb") as f:
    pickle.dump(tree_clf, f)

with open(FEATURES_OUTFILE, "wb") as f:
    pickle.dump(feature_names, f)

print("\nSaved model & feature list:")
print(MODEL_OUTFILE)
print(FEATURES_OUTFILE)

print("\n🎉 TRAINING COMPLETE — FINAL MODEL READY!")
