# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_FILE = "vetbot_model.pkl"        # or vetbot_model.pkl — use the correct filename
ENCODERS_FILE = "label_encoders.pkl"
FEATURES_FILE = "model_features.pkl"

# --- Helpers ---
def safe_float(val, default=0.0):
    try:
        if val is None or str(val).strip() == "":
            return default
        return float(val)
    except:
        return default

def safe_str(val):
    return "" if val is None else str(val).strip()

def transform_with_unknown(le, values):
    """
    Transform a list/Series of string values using a LabelEncoder `le`.
    If a value is unseen, replace it with "<unknown>" — adding "<unknown>" to
    the encoder classes if necessary.
    Returns numpy array of encoded ints.
    """
    # ensure numpy array input
    vals = np.array([safe_str(v) for v in values], dtype=object)

    # Standardize the missing marker that we used during training (if any)
    # We'll use "<unknown>" as the escape token for unseen values.
    unknown_token = "<unknown>"

    # If unknown_token is not already a class, add it so transform won't fail.
    if unknown_token not in le.classes_:
        le.classes_ = np.append(le.classes_, unknown_token)

    # Replace unseen values with unknown_token
    known = set(le.classes_)
    replaced = [v if v in known else unknown_token for v in vals]

    return le.transform(replaced)

# --- Load artifacts ---
if not os.path.exists(MODEL_FILE):
    raise SystemExit(f"Could not find model file: {MODEL_FILE}")
if not os.path.exists(ENCODERS_FILE):
    raise SystemExit(f"Could not find encoders file: {ENCODERS_FILE}")
if not os.path.exists(FEATURES_FILE):
    raise SystemExit(f"Could not find features file: {FEATURES_FILE}")

model = pickle.load(open(MODEL_FILE, "rb"))
encoders = pickle.load(open(ENCODERS_FILE, "rb"))
model_features = pickle.load(open(FEATURES_FILE, "rb"))

# Ensure we have a target encoder
if "Disease_Prediction" not in encoders:
    raise SystemExit("label_encoders.pkl must include a 'Disease_Prediction' encoder.")

target_le = encoders["Disease_Prediction"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Build input dict in memory — we will reorder to model.feature_names_in_
        input_dict = {
            # Use same feature names exact as model expects; we'll reorder below.
            "Animal_Type": safe_str(form.get("animal_type")).capitalize(),
            "Breed": safe_str(form.get("breed")).capitalize(),
            "Age": safe_float(form.get("age"), default=0.0),
            "Gender": safe_str(form.get("gender")).capitalize(),
            "Weight": safe_float(form.get("weight"), default=0.0),
            "Symptom_1": safe_str(form.get("symptom1")).title(),
            "Symptom_2": safe_str(form.get("symptom2")).title(),
            "Symptom_3": safe_str(form.get("symptom3")).title(),
            "Symptom_4": safe_str(form.get("symptom4")).title(),
            "Duration": safe_float(form.get("duration"), default=0.0),
            "Appetite_Loss": safe_str(form.get("appetite_loss", "No")).title(),
            "Vomiting": safe_str(form.get("vomiting", "No")).title(),
            "Diarrhea": safe_str(form.get("diarrhea", "No")).title(),
            "Coughing": safe_str(form.get("coughing", "No")).title(),
            "Labored_Breathing": safe_str(form.get("labored_breathing", "No")).title(),
            "Lameness": safe_str(form.get("lameness", "No")).title(),
            "Skin_Lesions": safe_str(form.get("skin_lesions", "No")).title(),
            "Nasal_Discharge": safe_str(form.get("nasal_discharge", "No")).title(),
            "Eye_Discharge": safe_str(form.get("eye_discharge", "No")).title(),
            "Body_Temperature": safe_float(form.get("body_temperature"), default=0.0),
            "Heart_Rate": safe_float(form.get("heart_rate"), default=0.0)
        }

        # Create DataFrame
        df = pd.DataFrame([input_dict])

        # Encode categorical columns using saved encoders.
        # encoders is a dict mapping column -> LabelEncoder
        for col, le in encoders.items():
            if col == "Disease_Prediction":
                continue  # skip target
            if col in df.columns:
                # transform as a vector and place encoded ints back into df
                encoded = transform_with_unknown(le, df[col].astype(str).values)
                df[col] = encoded

        # Reorder columns to exactly model.feature_names_in_
        if hasattr(model, "feature_names_in_"):
            expected_order = list(model.feature_names_in_)
        else:
            expected_order = list(model_features)

        # Add any missing columns with default 0 (numeric) or encoded "<unknown>" if categorical
        for c in expected_order:
            if c not in df.columns:
                # If we have an encoder for the column, make its unknown value encoded
                if c in encoders and c != "Disease_Prediction":
                    le = encoders[c]
                    # ensure unknown exists
                    if "<unknown>" not in le.classes_:
                        le.classes_ = np.append(le.classes_, "<unknown>")
                    unknown_code = le.transform([ "<unknown>" ])[0]
                    df[c] = unknown_code
                else:
                    df[c] = 0

        df = df[expected_order]  # reorder

        # Finally predict
        y_enc = model.predict(df)          # returns encoded label(s)
        # ensure y_enc is 1-d array
        y_enc = np.array(y_enc).ravel().astype(int)

        # Inverse-transform to disease name
        # Make sure target_le knows "<unknown>" if needed
        if "<unknown>" not in target_le.classes_:
            target_le.classes_ = np.append(target_le.classes_, "<unknown>")

        disease_names = target_le.inverse_transform(y_enc)
        predicted_name = disease_names[0] if len(disease_names) > 0 else "Unknown"

        return render_template("result.html", prediction=predicted_name, animal=input_dict["Animal_Type"], input_preview=input_dict)

    except Exception as e:
        # For debugging show the error message in the result page
        return render_template("result.html", prediction=f"Error: {e}", animal="Unknown", input_preview={})

if __name__ == "__main__":
    app.run(debug=True)
