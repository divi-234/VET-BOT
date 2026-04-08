import numpy as np
import pickle
import pandas as pd
from difflib import get_close_matches

# ============================
# LOAD MODEL & ENCODERS
# ============================
with open("dog_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("dog_label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("dog_target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

with open("dog_model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

TOP_K = 3
CONFIDENCE_THRESHOLD = 0.50

# ============================
# COLUMN TYPES (🔥 IMPORTANT)
# ============================
NUMERIC_COLS = ["Age", "Weight", "Body_Temperature", "Heart_Rate"]

CATEGORICAL_COLS = [
    col for col in model_features if col not in NUMERIC_COLS
]

# ============================
# SAFE ENCODING
# ============================
def safe_encode(value, encoder):
    if value is None:
        value = "None"

    value_str = str(value).strip()
    classes = list(encoder.classes_)
    classes_lower = [c.lower() for c in classes]

    # Exact match (case-insensitive)
    if value_str.lower() in classes_lower:
        original_class = classes[classes_lower.index(value_str.lower())]
        return encoder.transform([original_class])[0]

    # Fuzzy match
    match = get_close_matches(value_str.lower(), classes_lower, n=1, cutoff=0.7)
    if match:
        original_class = classes[classes_lower.index(match[0])]
        return encoder.transform([original_class])[0]

    # Fallback to "None" if it exists
    if "None" in classes:
        return encoder.transform(["None"])[0]

    # Absolute fallback
    return 0


# ============================
# NUMERIC SAFETY
# ============================
def to_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default

# ============================
# PREDICTION FUNCTION
# ============================
def predict_dog_disease_top3(input_data: dict):

    df = pd.DataFrame([input_data])

    # Ensure all features exist
    for col in model_features:
        if col not in df.columns:
            df[col] = "None"

    # 🔒 FORCE NUMERIC TYPES (NO ENCODING HERE)
    for col in NUMERIC_COLS:
        df[col] = df[col].apply(to_float)

    # 🔐 Encode ONLY categorical columns
    for col in CATEGORICAL_COLS:
        if col in label_encoders:
            encoder = label_encoders[col]
            df[col] = df[col].apply(lambda x: safe_encode(x, encoder))

    # Final order
    df = df[model_features]

    # 🚀 Predict
    probs = model.predict_proba(df)[0]
    top_idx = np.argsort(probs)[::-1][:TOP_K]

    results = []
    for idx in top_idx:
        disease = target_encoder.inverse_transform([idx])[0]
        confidence = round(probs[idx] * 100, 2)
        results.append((disease, confidence))

    best_conf = results[0][1] / 100

    if best_conf >= 0.70:
        confidence_level = "HIGH"
        final_prediction = results[0][0]
    elif best_conf >= CONFIDENCE_THRESHOLD:
        confidence_level = "MEDIUM"
        final_prediction = results[0][0]
    else:
        confidence_level = "LOW"
        final_prediction = "Uncertain — Consult Veterinarian"

    return {
        "final_prediction": final_prediction,
        "confidence_level": confidence_level,
        "top_confidence": results[0][1],
        "top_3_predictions": results
    }

# ============================
# TEST RUN
# ============================
if __name__ == "__main__":

    sample_input = {
        "Animal_Type": "Dog",
        "Breed": "Labrador",
        "Age": "4",
        "Gender": "Male",
        "Weight": "25",
        "Symptom_1": "Coughing",
        "Symptom_2": "Fever",
        "Symptom_3": "Nasal Discharge",
        "Symptom_4": "No",
        "Duration": "5",
        "Appetite_Loss": "Yes",
        "Vomiting": "No",
        "Diarrhea": "No",
        "Coughing": "Yes",
        "Labored_Breathing": "Yes",
        "Lameness": "No",
        "Skin_Lesions": "No",
        "Nasal_Discharge": "Yes",
        "Eye_Discharge": "No",
        "Body_Temperature": "39.5",
        "Heart_Rate": "110"
    }

    output = predict_dog_disease_top3(sample_input)

    print("\n🐶 FINAL PREDICTION:", output["final_prediction"])
    print("📈 CONFIDENCE LEVEL:", output["confidence_level"])
    print("\n📊 TOP 3 DISEASES:")
    for d, c in output["top_3_predictions"]:
        print(f"• {d} → {c}%")
