# chatbot.py (fixed)
import os
import joblib
import numpy as np
import pandas as pd
import traceback

# --- config: filenames to try ---
MODEL_FILES = [
    "vet_disease_xgb.pkl",
    "vet_disease_xgb_model.pkl",
    "vet_disease_xgb_tuned.pkl",
    "vet_disease_xgb_model.joblib",
    "vet_disease_xgb.joblib",
]
ENCODERS_FILE = "label_encoders.pkl"   # dict of LabelEncoders (including target)
MODEL_FEATURES_FILE = "model_features.pkl"
TARGET_ENCODER_FILE = "target_encoder.pkl"  # optional separate file

# yes/no columns used in your dataset
YESNO_COLS = {
    "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing", "Labored_Breathing",
    "Lameness", "Skin_Lesions", "Nasal_Discharge", "Eye_Discharge"
}

NUMERIC_COLS = {"Age", "Weight", "Duration", "Body_Temperature", "Heart_Rate"}

def find_model_path():
    for p in MODEL_FILES:
        if os.path.exists(p):
            return p
    return None

def safe_load_model_and_encoders():
    model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError("No model file found. Expected one of: " + ", ".join(MODEL_FILES))

    model = joblib.load(model_path)
    encoders = None
    if os.path.exists(ENCODERS_FILE):
        encoders = joblib.load(ENCODERS_FILE)
    else:
        encoders = {}

    model_features = None
    if os.path.exists(MODEL_FEATURES_FILE):
        model_features = joblib.load(MODEL_FEATURES_FILE)
    else:
        raise FileNotFoundError(f"{MODEL_FEATURES_FILE} not found")

    target_encoder = None
    if os.path.exists(TARGET_ENCODER_FILE):
        target_encoder = joblib.load(TARGET_ENCODER_FILE)
    else:
        # try to find target encoder inside encoders dict
        for cand in ("Disease_Prediction", "disease", "target", "label"):
            if cand in encoders:
                target_encoder = encoders[cand]
                break
    return model, encoders, target_encoder, model_features

def normalize_yesno(text):
    if text is None:
        return "<missing>"
    s = str(text).strip().lower()
    if s == "":
        return "<missing>"
    if s in ("yes", "y", "true", "1"):
        return "Yes"
    if s in ("no", "n", "false", "0"):
        return "No"
    # otherwise return capitalized
    return s.capitalize()

def normalize_categorical(col, val):
    if val is None or str(val).strip() == "":
        return "<missing>"
    s = str(val).strip()
    # for symptoms we used Title case in training
    if col.startswith("Symptom"):
        return s.title()
    # for animal type & gender
    if col in ("Animal_Type", "Gender"):
        return s.capitalize()
    # breed - title case
    if col == "Breed":
        return s.title()
    # default
    return s

def ensure_le_array(le):
    # convert classes_ to numpy array if it's a list (or not numpy)
    if not hasattr(le, "classes_"):
        return
    if not isinstance(le.classes_, np.ndarray):
        le.classes_ = np.array(list(le.classes_), dtype=object)

def encode_column_with_le(le, values):
    """
    values: list-like of string values
    returns: np.array of encoded ints
    Strategy:
      - ensure le.classes_ is np.ndarray
      - unseen -> map to "<missing>" if present in classes_, else -> "<unknown>" (and add it)
    """
    ensure_le_array(le)
    classes_set = set(le.classes_.tolist())
    use_missing = "<missing>" in classes_set

    mapped = []
    for v in values:
        if v in classes_set:
            mapped.append(v)
        else:
            if use_missing:
                mapped.append("<missing>")
            else:
                mapped.append("<unknown>")

    # if we used <unknown> but it's not present in classes_, append it to classes_
    if "<unknown>" in mapped and "<unknown>" not in classes_set:
        le.classes_ = np.append(le.classes_, "<unknown>")
    # transform
    return le.transform(mapped)

def main():
    print("🐾 Welcome to VetBot – your intelligent veterinary assistant!")
    try:
        model, encoders, target_encoder, model_features = safe_load_model_and_encoders()
    except Exception as e:
        print("❌ Found model file:", e)
        return

    print("✅ Found model and encoders!")
    print("✅ Model and encoders loaded successfully!\n")

    # Collect user input
    animal_type = input("👉 What kind of animal is it? (Dog/Cow): ").strip()
    breed = input("👉 What is the breed? (e.g., Labrador, Jersey): ").strip()
    gender = input("👉 What is the gender? (Male/Female): ").strip()
    try:
        age = float(input("👉 How old is your animal? (in years): ").strip())
    except:
        age = np.nan
    try:
        weight = float(input("👉 What is the weight? (in kg): ").strip())
    except:
        weight = np.nan
    try:
        duration = float(input("👉 How long has the animal shown symptoms? (in days): ").strip())
    except:
        duration = np.nan

    print("\n👉 Enter up to 4 symptoms (type 'none' or leave blank if not applicable):")
    s1 = input("Symptom 1: ").strip()
    s2 = input("Symptom 2: ").strip()
    s3 = input("Symptom 3: ").strip()
    s4 = input("Symptom 4: ").strip()

    print("\nAdditional questions (optional):")
    appetite_loss = normalize_yesno(input("Loss of appetite? (yes/no or press enter to skip): ").strip() if True else "")
    vomiting = normalize_yesno(input("Vomiting? (yes/no or press enter to skip): ").strip() if True else "")
    diarrhea = normalize_yesno(input("Diarrhea? (yes/no or press enter to skip): ").strip() if True else "")
    coughing = normalize_yesno(input("Coughing? (yes/no or press enter to skip): ").strip() if True else "")
    labored_breathing = normalize_yesno(input("Labored breathing? (yes/no or press enter to skip): ").strip() if True else "")
    lameness = normalize_yesno(input("Lameness or limping? (yes/no or press enter to skip): ").strip() if True else "")
    skin_lesions = normalize_yesno(input("Skin lesions or rashes? (yes/no or press enter to skip): ").strip() if True else "")
    nasal_discharge = normalize_yesno(input("Nasal discharge? (yes/no or press enter to skip): ").strip() if True else "")
    eye_discharge = normalize_yesno(input("Eye discharge? (yes/no or press enter to skip): ").strip() if True else "")

    # numeric vitals (optional)
    bt = input("Body temperature (in °C, press Enter if unknown): ").strip()
    hr = input("Heart rate (press Enter if unknown): ").strip()
    try:
        bt_val = float(bt) if bt != "" else np.nan
    except:
        bt_val = np.nan
    try:
        hr_val = float(hr) if hr != "" else np.nan
    except:
        hr_val = np.nan

    # Prepare row dict with default "<missing>" for categorical blanks (training used "<missing>")
    row = {}
    for col in model_features:
        # default fill later
        row[col] = np.nan

    # fill in known values (convert names to training-style)
    row["Animal_Type"] = normalize_categorical("Animal_Type", animal_type)
    row["Breed"] = normalize_categorical("Breed", breed)
    row["Gender"] = normalize_categorical("Gender", gender)
    row["Age"] = age
    row["Weight"] = weight
    # Duration expected by your model (some models required it)
    row["Duration"] = duration

    row["Symptom_1"] = normalize_categorical("Symptom_1", s1)
    row["Symptom_2"] = normalize_categorical("Symptom_2", s2)
    row["Symptom_3"] = normalize_categorical("Symptom_3", s3)
    row["Symptom_4"] = normalize_categorical("Symptom_4", s4)

    row["Appetite_Loss"] = appetite_loss
    row["Vomiting"] = vomiting
    row["Diarrhea"] = diarrhea
    row["Coughing"] = coughing
    row["Labored_Breathing"] = labored_breathing
    row["Lameness"] = lameness
    row["Skin_Lesions"] = skin_lesions
    row["Nasal_Discharge"] = nasal_discharge
    row["Eye_Discharge"] = eye_discharge

    row["Body_Temperature"] = bt_val
    row["Heart_Rate"] = hr_val

    # Create DataFrame (1 row)
    df = pd.DataFrame([row], columns=model_features)

    # For categorical columns, fill empty with "<missing>" (same token training used)
    for col in df.columns:
        if col in NUMERIC_COLS:
            # keep numeric; convert to float (NaN stays NaN)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # categorical: if empty/NaN -> "<missing>"
            df[col] = df[col].astype(str).fillna("<missing>")
            # treat user typed 'none' as missing
            df[col] = df[col].replace({"None": "<missing>", "none": "<missing>", "nan": "<missing>"}).astype(str)

    # Encode categorical fields using loaded label encoders
    try:
        for col in df.columns:
            if col in encoders:
                le = encoders[col]
                # ensure classes_ is numpy array
                if not hasattr(le, "classes_"):
                    # some broken encoder - skip encoding
                    continue
                if not isinstance(le.classes_, np.ndarray):
                    le.classes_ = np.array(list(le.classes_), dtype=object)

                # values to encode
                vals = df[col].astype(str).tolist()
                # map unseen values -> "<missing>" if present else "<unknown>"
                classes_set = set(le.classes_.tolist())
                mapped_vals = []
                for v in vals:
                    if v in classes_set:
                        mapped_vals.append(v)
                    else:
                        mapped_vals.append("<missing>" if "<missing>" in classes_set else "<unknown>")

                # if we used <unknown> but it's not in classes_, add it
                if "<unknown>" in mapped_vals and "<unknown>" not in classes_set:
                    le.classes_ = np.append(le.classes_, "<unknown>")

                # finally transform
                df[col] = le.transform(mapped_vals)
    except Exception as e:
        # Capture full traceback for debugging but present readable message
        print("\n❌ Error during encoding:")
        traceback.print_exc()
        print("\nThe encoders might have unexpected structure. Please let me know and I will help inspect them.")
        return

    # final check: ensure column order matches model_features
    df = df[model_features]

    # now predict
    print("\n🔍 Analyzing symptoms...\n")
    try:
        y_pred = model.predict(df)          # returns encoded labels (ints)
        # decode
        decoded = None
        if target_encoder is not None:
            # ensure it's a LabelEncoder-like object
            if hasattr(target_encoder, "inverse_transform"):
                decoded = target_encoder.inverse_transform(y_pred)
            else:
                # if stored as list/array of class names
                try:
                    decoded = [str(target_encoder[int(i)]) for i in y_pred]
                except:
                    decoded = [str(i) for i in y_pred]
        else:
            # try to find disease encoder in encoders dict
            for cand in ("Disease_Prediction", "disease", "target", "label"):
                if cand in encoders:
                    le = encoders[cand]
                    if hasattr(le, "inverse_transform"):
                        decoded = le.inverse_transform(y_pred)
                        break
            # fallback to model.classes_ (if sklearn wrapper set it)
            if decoded is None:
                try:
                    decoded = [model.classes_[int(i)] for i in y_pred]
                except:
                    decoded = [str(i) for i in y_pred]

        disease_name = decoded[0] if decoded is not None else str(y_pred[0])
        print(f"✅ Based on the symptoms, the predicted disease is likely: **{disease_name}** 🩺")
    except Exception as e:
        print("❌ Error during prediction:", e)
        traceback.print_exc()
        print("\n(If the error mentions `feature_names` or mismatched columns, make sure `model_features.pkl` matches the model.)")

    print("\n🐕 Thank you for using VetBot! Take your pet to a vet for professional confirmation.")

if __name__ == "__main__":
    main()
