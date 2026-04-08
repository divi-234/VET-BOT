# vet_disease_xgb_fixed.py
import os
import re
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier, callback

warnings.filterwarnings("ignore")
np.random.seed(42)

CSV_PATH = "cow_dog_data_augmented_v4.csv"
ENCODING = "latin1"

# ------------------------- CLEANING HELPERS -------------------------
def clean_temperature(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[^\d\.\-]", "", str(x))
    return float(s) if s else np.nan

def clean_heartrate(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[^\d\.\-]", "", str(x))
    return float(s) if s else np.nan

# ------------------------- LOAD DATA -------------------------
print("📂 Loading dataset...")
data = pd.read_csv(CSV_PATH, encoding=ENCODING)
print("✅ Loaded:", data.shape)

if "Body_Temperature" in data.columns:
    data["Body_Temperature"] = data["Body_Temperature"].apply(clean_temperature)
if "Heart_Rate" in data.columns:
    data["Heart_Rate"] = data["Heart_Rate"].apply(clean_heartrate)

data = data.dropna(subset=["Animal_Type", "Disease_Prediction"]).reset_index(drop=True)
data = data[data["Animal_Type"].str.lower().isin(["dog", "cow"])]

# Drop rare disease labels (<2 samples)
counts = data["Disease_Prediction"].value_counts()
valid_labels = counts[counts >= 2].index
data = data[data["Disease_Prediction"].isin(valid_labels)]

# ------------------------- FEATURES -------------------------
feature_cols = [
    "Animal_Type", "Breed", "Age", "Gender", "Weight",
    "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4",
    "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing",
    "Labored_Breathing", "Lameness", "Skin_Lesions",
    "Nasal_Discharge", "Eye_Discharge", "Body_Temperature", "Heart_Rate"
]
feature_cols = [c for c in feature_cols if c in data.columns]
X = data[feature_cols].copy()
y = data["Disease_Prediction"].copy()

# ------------------------- ENCODING -------------------------
print("🔠 Encoding categorical features...")
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna("Unknown"))
    label_encoders[col] = le

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)
label_encoders["Disease_Prediction"] = target_le

# ------------------------- SPLIT -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ------------------------- RANDOMIZED SEARCH -------------------------
print("🎯 Running hyperparameter tuning...")
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(target_le.classes_),
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False
)

param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_params = search.best_params_
print("✅ Best Params:", best_params)

# ------------------------- FINAL TRAINING -------------------------
print("🚀 Final training (manual early stopping)...")

from sklearn.metrics import log_loss

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

final_model = XGBClassifier(
    **best_params,
    objective="multi:softprob",
    num_class=len(target_le.classes_),
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False
)

best_iter = 0
best_loss = float("inf")
no_improve = 0
patience = 30

for i in range(1, best_params["n_estimators"] + 1):
    final_model.set_params(n_estimators=i)
    final_model.fit(X_train_final, y_train_final, verbose=False)
    y_val_pred = final_model.predict_proba(X_val)
    loss = log_loss(y_val, y_val_pred, labels=np.arange(len(target_le.classes_)))

    if loss < best_loss - 1e-5:
        best_loss = loss
        best_iter = i
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"⏹ Early stopping at {i} rounds (best={best_iter}, loss={best_loss:.4f})")
        break

print(f"✅ Training completed with {best_iter} estimators (best logloss={best_loss:.4f})")



# ------------------------- EVALUATE -------------------------
y_pred = final_model.predict(X_test)
print("\n📊 Results:")
print("Train Accuracy:", accuracy_score(y_train, final_model.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test, y_pred))
from sklearn.utils.multiclass import unique_labels

present_labels = unique_labels(y_test, y_pred)
present_names = target_le.inverse_transform(present_labels)
print("\n", classification_report(y_test, y_pred, labels=present_labels, target_names=present_names, zero_division=0))


# ------------------------- SAVE -------------------------
print("\n💾 Saving model and encoders...")
joblib.dump(final_model, "vet_disease_xgb.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(list(X.columns), "model_features.pkl")
print("✅ All artifacts saved successfully.")
