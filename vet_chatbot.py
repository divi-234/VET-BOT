# train_vet_xgb.py
import os
import re
import joblib
import warnings
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------- CONFIG ----------
# Primary filename you said you have
POTENTIAL_FILES = [
    "cow_dog_data_augmented_final.csv",
    "cow_dog_data_augmented_v4.csv",
    "cow_dog_disease_dataset_v5 (1).csv",
    "cow_dog_data_augmented_v3.csv"
]
CSV_PATH = next((f for f in POTENTIAL_FILES if os.path.exists(f)), None)
if CSV_PATH is None:
    raise SystemExit("No dataset found. Put your CSV in the working directory and re-run.")

ENCODING = "latin1"   # handles degree symbol artifacts
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ITER = 30           # RandomizedSearchCV iterations (reduce for speed)
CV_FOLDS = 3
FINAL_VAL_SIZE = 0.15

# Outputs
MODEL_SKLEARN_OUT = "vet_disease_xgb_model.pkl"
MODEL_BOOSTER_OUT = "vet_disease_xgb_booster.json"
ENCODERS_OUT = "label_encoders.pkl"
FEATURES_OUT = "model_features.pkl"

# ---------- UTILITIES ----------
def clean_numeric_from_string(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"[^\d\.\-]", "", s)   # keep digits, dot, minus
    try:
        return float(s) if s != "" else np.nan
    except:
        return np.nan

# ---------- LOAD ----------
print("📂 Loading dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH, encoding=ENCODING)
print("✅ Loaded rows:", df.shape)
print("-" * 60)

# ---------- BASIC CLEAN ----------
# Clean Body_Temperature and Heart_Rate
if "Body_Temperature" in df.columns:
    df["Body_Temperature"] = df["Body_Temperature"].apply(clean_numeric_from_string)
if "Heart_Rate" in df.columns:
    df["Heart_Rate"] = df["Heart_Rate"].apply(clean_numeric_from_string)

# Normalize symptom text (optional)
symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
for c in symptom_cols:
    df[c] = df[c].astype(str).str.strip().str.title().replace({"None": np.nan, "Nan": np.nan})

# Normalize yes/no columns if present
yesno_cols = ["Appetite_Loss", "Vomiting", "Diarrhea", "Coughing", "Labored_Breathing",
              "Lameness", "Skin_Lesions", "Nasal_Discharge", "Eye_Discharge"]
for c in yesno_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({
            "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
            "no": "No", "n": "No", "false": "No", "0": "No",
            "nan": np.nan, "none": np.nan
        })

# Drop rows missing target or Animal_Type
TARGET = "Disease_Prediction"
df = df.dropna(subset=["Animal_Type", TARGET]).reset_index(drop=True)

# Keep only Dogs and Cows
df = df[df["Animal_Type"].str.lower().isin(["dog", "cow"])].reset_index(drop=True)

# Remove extremely rare disease labels (<2 samples)
min_count = 2
counts = df[TARGET].value_counts()
valid = counts[counts >= min_count].index
removed = set(df[TARGET].unique()) - set(valid)
if removed:
    print(f"Removing {len(removed)} rare disease labels (<{min_count} samples).")
    df = df[df[TARGET].isin(valid)].reset_index(drop=True)

print("After cleaning - rows:", len(df), "unique diseases:", df[TARGET].nunique())
print("-" * 60)

# ---------- FEATURES ----------
# Choose features (based on earlier conversation)
feature_cols = [
    "Animal_Type", "Breed", "Age", "Gender", "Weight",
    "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4",
    "Duration", "Appetite_Loss", "Vomiting", "Diarrhea",
    "Coughing", "Labored_Breathing", "Lameness", "Skin_Lesions",
    "Nasal_Discharge", "Eye_Discharge", "Body_Temperature", "Heart_Rate"
]
# keep only existing columns
feature_cols = [c for c in feature_cols if c in df.columns]
print("Using features:", feature_cols)

X = df[feature_cols].copy()
y = df[TARGET].copy()

# ---------- ENCODING ----------
print("🔠 Encoding categorical columns (LabelEncoder)...")
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    # ensure reproducible mapping by filling missing consistently
    X[col] = X[col].fillna("<missing>").astype(str)
    le.fit(X[col])
    # ensure '<unknown>' is present for later inference
    if "<unknown>" not in le.classes_:
        le.classes_ = np.append(le.classes_, "<unknown>")
    X[col] = le.transform(X[col])
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)
label_encoders[TARGET] = target_le

# Save features order (needed by chatbot)
joblib.dump(list(X.columns), FEATURES_OUT)

print("Encoders built for:", list(label_encoders.keys()))
print("-" * 60)

# ---------- SPLIT ----------
print("🔀 Train/test split (stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("-" * 60)

# ---------- RANDOMIZED SEARCH ----------
print("🎯 Hyperparameter tuning (RandomizedSearchCV)...")
base = XGBClassifier(
    objective="multi:softprob",
    num_class=len(target_le.classes_),
    use_label_encoder=False,
    verbosity=0,
    random_state=RANDOM_STATE
)

param_dist = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "gamma": [0, 0.5, 1],
    "min_child_weight": [1, 3, 5]
}

rs = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

start = time()
rs.fit(X_train, y_train)
elapsed = time() - start
print(f"✅ RandomizedSearchCV done in {elapsed:.1f}s")
print("Best params:", rs.best_params_)
print("Best CV score:", rs.best_score_)
print("-" * 60)
best_params = rs.best_params_

# ---------- FINAL TRAINING with early stopping (robust) ----------
print("🚀 Final training — attempting early stopping (handles xgboost version differences)...")
# Create train/val split from X_train
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=FINAL_VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
)

# Try sklearn-wrapper early stopping first (most convenient)
final_model = XGBClassifier(
    **best_params,
    objective="multi:softprob",
    num_class=len(target_le.classes_),
    use_label_encoder=False,
    verbosity=0,
    random_state=RANDOM_STATE
)

trained_with_sklearn_es = False
try:
    final_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )
    trained_with_sklearn_es = True
    print("Trained using XGBClassifier.fit(..., early_stopping_rounds=...)")
except TypeError:
    # fallback: use xgboost.train with DMatrix & callbacks (works across versions)
    print("sklearn .fit(... early_stopping_rounds ...) not supported in this xgboost version — falling back to xgb.train()")
    params = final_model.get_xgb_params()
    dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "val")]
    num_boost_round = best_params.get("n_estimators", 400)
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False
    )
    # attach booster to sklearn wrapper so predict() works similarly
    final_model._Booster = bst
    final_model.__dict__['n_features_in_'] = X.shape[1]
    final_model.__dict__['feature_names_in_'] = np.array(X.columns)
    # set best_iteration if present
    if hasattr(bst, "best_iteration"):
        final_model.best_iteration = int(bst.best_iteration)
    print("Trained using xgboost.train() and attached Booster to sklearn wrapper.")

# ---------- EVALUATION ----------
print("📊 Evaluating on hold-out test set...")
# If sklearn fit used, final_model is ready; if fallback used we attached booster
y_pred = final_model.predict(X_test)
train_acc = accuracy_score(y_train, final_model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print("\nClassification report (partial if many classes):")
print(classification_report(y_test, y_pred, zero_division=0))
print("-" * 60)

# ---------- SAVE ARTIFACTS ----------
print("💾 Saving model and encoders...")
# Save sklearn wrapper (joblib)
joblib.dump(final_model, MODEL_SKLEARN_OUT)
# Also save raw Booster (for cross-version loading)
try:
    # If sklearn wrapper contains booster:
    booster = final_model.get_booster()
    booster.save_model(MODEL_BOOSTER_OUT)
except Exception:
    # if _Booster attached
    if hasattr(final_model, "_Booster") and final_model._Booster is not None:
        final_model._Booster.save_model(MODEL_BOOSTER_OUT)

joblib.dump(label_encoders, ENCODERS_OUT)
joblib.dump(list(X.columns), FEATURES_OUT)

print("Saved:")
print(" -", MODEL_SKLEARN_OUT)
print(" -", MODEL_BOOSTER_OUT)
print(" -", ENCODERS_OUT)
print(" -", FEATURES_OUT)
print("🎉 Training finished.")
