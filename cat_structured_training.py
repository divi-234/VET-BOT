# ============================================================
# 🐱 Cat Disease Prediction - STRUCTURED APPROACH
# Same as Dog/Cow Models - Tabular Data
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - SAME AS DOG/COW
# ============================================================
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

TARGET_COL = "Disease"

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "="*60)
print("🐱 CAT DISEASE MODEL TRAINING - STRUCTURED APPROACH")
print("="*60)

# You need to convert your text-based dataset to structured format
# Expected columns:
# Age, Weight, Breed, Gender, Symptom_1, Symptom_2, Symptom_3, Symptom_4,
# Duration, Appetite_Loss, Vomiting, Diarrhea, Coughing, Labored_Breathing,
# Lameness, Skin_Lesions, Nasal_Discharge, Eye_Discharge,
# Body_Temperature, Heart_Rate, Disease

# For now, I'll show you how to create a sample dataset from your text data
# You'll need to run this conversion script first

DATA_PATH = "cat_disease_dataset_structured.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print(f"\n✅ Loaded {len(df)} cat disease records")
except FileNotFoundError:
    print(f"\n❌ File not found: {DATA_PATH}")
    print("\n📋 REQUIRED FORMAT:")
    print("You need to convert your text-based dataset to structured format.")
    print("\nExpected columns:")
    for col in CATEGORICAL_COLS + NUMERICAL_COLS + BINARY_COLS + [TARGET_COL]:
        print(f"  • {col}")
    print("\nSee conversion script below ⬇️")
    exit(1)

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\n📊 Dataset Overview:")
print(f"Total records: {len(df)}")
print(f"Diseases: {df[TARGET_COL].nunique()}")
print(f"\nDisease distribution:")
print(df[TARGET_COL].value_counts())

# Handle missing values
for col in NUMERICAL_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Fill categorical missing values
for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].fillna('unknown')

# Ensure binary columns are 0/1
for col in BINARY_COLS:
    if col in df.columns:
        df[col] = df[col].map({
            'yes': 1, 'Yes': 1, 'YES': 1, 'y': 1, 'Y': 1, 1: 1,
            'no': 0, 'No': 0, 'NO': 0, 'n': 0, 'N': 0, 0: 0
        })
        df[col] = df[col].fillna(0).astype(int)

# ============================================================
# 3. LABEL ENCODING (Same as Dog/Cow)
# ============================================================
print("\n🔧 Encoding categorical features...")

label_encoders = {}

for col in CATEGORICAL_COLS:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Target encoding
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[TARGET_COL])

print(f"✅ Encoded {len(label_encoders)} categorical columns")
print(f"✅ Target classes: {len(target_encoder.classes_)}")

# ============================================================
# 4. FEATURE PREPARATION
# ============================================================
# Get all feature columns in correct order
feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS + BINARY_COLS

# Ensure all columns exist
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[feature_cols]

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train-test split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# ============================================================
# 6. MODEL TRAINING (RandomForest like Dog/Cow)
# ============================================================
print("\n🤖 Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# ============================================================
# 7. EVALUATION
# ============================================================
print("\n📊 Model Evaluation:")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.2%}")
print("\n📋 Classification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=target_encoder.classes_,
    zero_division=0
))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================
# 8. SAVE MODELS (Same format as Dog/Cow)
# ============================================================
print("\n💾 Saving model files...")

# Save numerical medians for preprocessing
numerical_medians = {}
for col in NUMERICAL_COLS:
    if col in df.columns:
        numerical_medians[col] = float(df[col].median())

with open("cat_disease_model_improved.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)

with open("cat_label_encoders_improved.pkl", "wb") as f:
    pickle.dump(label_encoders, f, protocol=4)

with open("cat_target_encoder_improved.pkl", "wb") as f:
    pickle.dump(target_encoder, f, protocol=4)

with open("cat_model_features_improved.pkl", "wb") as f:
    pickle.dump(feature_cols, f, protocol=4)

with open("cat_numerical_medians.pkl", "wb") as f:
    pickle.dump(numerical_medians, f, protocol=4)

print("\n✅ Model files saved:")
print("   • cat_disease_model_improved.pkl")
print("   • cat_label_encoders_improved.pkl")
print("   • cat_target_encoder_improved.pkl")
print("   • cat_model_features_improved.pkl")
print("   • cat_numerical_medians.pkl")

print("\n" + "="*60)
print("🎉 CAT MODEL TRAINING COMPLETE!")
print("="*60)
print("\n📋 Next steps:")
print("1. Update Flask API to use structured cat model")
print("2. Test with real cases")
print("3. Deploy!")
print("\n" + "="*60)