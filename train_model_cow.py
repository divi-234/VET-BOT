# ============================================================
# 🐄 Cow Disease Prediction Model Training
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
DATA_PATH = "cow_dog_data_augmented_final.csv"
df = pd.read_csv(DATA_PATH)

# Keep only cows
df = df[df["Animal_Type"].str.lower() == "cow"].reset_index(drop=True)

print(f"\n🐄 TOTAL COW ROWS: {len(df)}")

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
# 4. RARE DISEASE FILTERING
# ---------------------------
MIN_CASES = 20
disease_counts = df[TARGET_COL].value_counts()
valid_diseases = disease_counts[disease_counts >= MIN_CASES].index
df = df[df[TARGET_COL].isin(valid_diseases)]

print(f"\n🧹 After filtering (min {MIN_CASES} cases):")
print(f"Remaining rows: {len(df)}")
print(f"Remaining disease classes: {df[TARGET_COL].nunique()}")

# ---------------------------
# 5. SAFE LABEL ENCODER
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
# 6. TRAIN / TEST SPLIT
# ---------------------------
X = df.drop([TARGET_COL, "Animal_Type"], axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n✅ Train-test split successful")

# ---------------------------
# 7. TRAIN MODEL
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
# 8. MODEL EVALUATION
# ---------------------------
y_pred = model.predict(X_test)

print("\n✅ MODEL EVALUATION")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=target_encoder.classes_
))

# ---------------------------
# 9. SAVE MODEL (Python 3.7 Compatible - Protocol 4)
# ---------------------------
print("\n💾 Saving models...")

with open("cow_disease_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)

with open("cow_label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f, protocol=4)

with open("cow_target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f, protocol=4)

with open("cow_model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f, protocol=4)

with open("cow_numerical_medians.pkl", "wb") as f:
    pickle.dump(df[NUMERICAL_COLS].median().to_dict(), f, protocol=4)

print("✅ Cow model & encoders saved successfully with protocol=4 (Python 3.7 compatible)!")
print("\n" + "="*60)
print("🎉 Training Complete!")
print("="*60)
print(f"Model files saved in: D:\\project")
print("\nCow model files created:")
print("  • cow_disease_model.pkl")
print("  • cow_label_encoders.pkl")
print("  • cow_target_encoder.pkl")
print("  • cow_model_features.pkl")
print("  • cow_numerical_medians.pkl")
print("\nNow update api.py to load these files!")
print("="*60)