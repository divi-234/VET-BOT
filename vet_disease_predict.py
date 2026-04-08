import pickle
import pandas as pd
import numpy as np


# 🩺 Load model and encoders
with open("vet_disease_xgb_tuned.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

def predict_disease(input_data):
    """Predict disease based on input data dict"""
    df = pd.DataFrame([input_data])

    # ✅ Ensure all required features exist
    for col in model_features:
        if col not in df.columns:
            df[col] = None

    # ✅ Encode categorical features (using saved encoders)
    for col, le in label_encoders.items():
        if col in df.columns:
            if df[col].iloc[0] not in le.classes_:
                # Handle unseen category
                df[col] = "<unknown>"
                le.classes_ = np.append(le.classes_, "<unknown>")
            df[col] = le.transform(df[col])

    # ✅ Convert numeric columns to float
    numeric_cols = ["Age", "Weight", "Body_Temperature", "Heart_Rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[model_features]  # ensure correct order

    # ✅ Predict
    pred = model.predict(df)[0]

    # ✅ Decode prediction (if possible)
    disease_encoder = label_encoders.get("Disease_Prediction", None)
    disease_name = disease_encoder.inverse_transform([int(pred)])[0] if disease_encoder else pred

    print(f"🩺 Predicted Disease: {disease_name}")
    return disease_name


# 🧪 Example input
sample_input = {
    "Animal_Type": "Dog",
    "Breed": "Labrador Retriever",
    "Age": 5,
    "Gender": "Male",
    "Weight": 30.0,
    "Symptom_1": "Coughing",
    "Symptom_2": "Appetite Loss",
    "Symptom_3": "Lethargy",
    "Symptom_4": "Nasal Discharge",
    "Duration": "3 days",
    "Appetite_Loss": "Yes",
    "Vomiting": "No",
    "Diarrhea": "No",
    "Coughing": "Yes",
    "Labored_Breathing": "No",
    "Lameness": "No",
    "Skin_Lesions": "No",
    "Nasal_Discharge": "Yes",
    "Eye_Discharge": "No",
    "Body_Temperature": 39.5,
    "Heart_Rate": 110
}

if __name__ == "__main__":
    print("📊 Predicting disease for sample input...\n")
    predict_disease(sample_input)
