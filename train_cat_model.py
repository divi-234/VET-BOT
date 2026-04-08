# ============================================================
# 🐱 Cat Disease Prediction Model Training
# Python 3.7 Compatible Version
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
import os

# ---------------------------
# 1. LOAD DATA
# ---------------------------
DATA_PATH = "cat_disease_data.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data from: {DATA_PATH}")
print(f"   Total rows: {len(df)}")

print(f"\n🐱 TOTAL CAT ROWS: {len(df)}")

if len(df) == 0:
    print("\n❌ No cat data found!")
    print("🎯 Solution: Run 'python generate_cat_data.py' first!")
    exit()

# ---------------------------
# 2. FEATURE DEFINITIONS
# ---------------------------
TARGET_COL = "Disease_Prediction"

CATEGORICAL_COLS = [
    "Breed", "Gender",
    "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"
]

NUMERICAL_COLS = [
    "Age", "Weight", "Duration",
    "Body_Temperature", "Heart_Rate"
]

BINARY_COLS = [
    "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing",
    "Labored_Breathing", "Lameness", "Skin_Lesions",
    "Nasal_Discharge", "Eye_Discharge"
]

# ---------------------------
# FIX: Convert Duration from "X days" → X
# ---------------------------
df["Duration"] = (
    df["Duration"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

# ---------------------------
# FIX: Convert Body_Temperature from "39.6°C" → 39.6
# ---------------------------
df["Body_Temperature"] = (
    df["Body_Temperature"]
    .astype(str)
    .str.replace("°C", "", regex=False)
    .str.strip()
)
df["Body_Temperature"] = pd.to_numeric(
    df["Body_Temperature"],
    errors="coerce"
)

# ---------------------------
# 3. DATA CLEANING
# ---------------------------
df = df.dropna(subset=[TARGET_COL])

# Convert binary columns from Yes/No to 1/0
for col in BINARY_COLS:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Fill missing values
df[BINARY_COLS] = df[BINARY_COLS].fillna(0).astype(int)
df[NUMERICAL_COLS] = df[NUMERICAL_COLS].fillna(df[NUMERICAL_COLS].median())
df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown")

print("\n📊 Disease counts:")
print(df[TARGET_COL].value_counts())

# ---------------------------
# 4. DISEASE NAME STANDARDIZATION
# ---------------------------
# Standardize cat disease names if needed
disease_mapping = {
    'Feline Panleukopenia': 'Panleukopenia',
    'Feline Calicivirus': 'Calicivirus',
    'Feline Herpesvirus': 'Herpesvirus',
    'FIV': 'Feline Immunodeficiency Virus',
    'FeLV': 'Feline Leukemia Virus'
}

df[TARGET_COL] = df[TARGET_COL].replace(disease_mapping)

print("\n📊 Disease counts AFTER standardization:")
print(df[TARGET_COL].value_counts())

# ---------------------------
# 5. RARE DISEASE FILTERING
# ---------------------------
MIN_CASES = 10  # Lower for cats since there might be less data
disease_counts = df[TARGET_COL].value_counts()
valid_diseases = disease_counts[disease_counts >= MIN_CASES].index
df = df[df[TARGET_COL].isin(valid_diseases)]

print(f"\n🧹 After filtering (min {MIN_CASES} cases):")
print(f"Remaining rows: {len(df)}")
print(f"Remaining disease classes: {df[TARGET_COL].nunique()}")

if len(df) < 100:
    print("\n⚠️  WARNING: Very limited cat data. Model accuracy may be lower.")

# ---------------------------
# 6. SAFE LABEL ENCODER
# ---------------------------
class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        return [
            self.classes_.tolist().index(x) if x in self.classes_ else -1
            for x in y
        ]

# Encode categorical features
encoders = {}
for col in CATEGORICAL_COLS:
    le = SafeLabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df[TARGET_COL] = target_encoder.fit_transform(df[TARGET_COL])

# ---------------------------
# 7. TRAIN / TEST SPLIT
# ---------------------------
X = df.drop([TARGET_COL, "Animal_Type"], axis=1)
y = df[TARGET_COL]

# Use smaller test size if data is limited
test_size = 0.2 if len(df) > 200 else 0.15

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    stratify=y,
    random_state=42
)

print("\n✅ Train-test split successful")

# ---------------------------
# 8. TRAIN MODEL
# ---------------------------
print("\n🤖 Training XGBoost model...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=y.nunique(),
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# 9. MODEL EVALUATION
# ---------------------------
y_pred = model.predict(X_test)

print("\n✅ MODEL EVALUATION")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=target_encoder.classes_,
    zero_division=0
))

# ---------------------------
# 10. SAVE MODEL (Python 3.7 Compatible - Protocol 4)
# ---------------------------
print("\n💾 Saving models...")

with open("cat_disease_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)

with open("cat_label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f, protocol=4)

with open("cat_target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f, protocol=4)

with open("cat_model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f, protocol=4)

with open("cat_numerical_medians.pkl", "wb") as f:
    pickle.dump(df[NUMERICAL_COLS].median().to_dict(), f, protocol=4)

print("✅ Cat model & encoders saved successfully with protocol=4 (Python 3.7 compatible)!")
print("\n" + "="*60)
print("🎉 Training Complete!")
print("="*60)
print(f"Model files saved in: D:\\project")
print("\nCat model files created:")
print("  • cat_disease_model.pkl")
print("  • cat_label_encoders.pkl")
print("  • cat_target_encoder.pkl")
print("  • cat_model_features.pkl")
print("  • cat_numerical_medians.pkl")
print("\nNow update api.py and index.html to load cat model!")
print("="*60)