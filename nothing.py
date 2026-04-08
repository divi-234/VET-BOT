# ============================================================
# 🐄 Cow Disease Prediction Model + CowVetBot Chatbot
# Dataset: cow_dog_data_augmented_final.csv
# Algorithm: XGBoost (Multi-class)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# ---------------------------
# 1. LOAD DATA
# ---------------------------
DATA_PATH = "cow_dog_data_augmented_final.csv"
df = pd.read_csv(DATA_PATH)

# Keep only cows
df = df[df["Animal_Type"].str.lower() == "cow"].reset_index(drop=True)

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

# ---------------------------
# 4. RARE DISEASE FILTERING
# ---------------------------
MIN_CASES = 20
disease_counts = df[TARGET_COL].value_counts()
valid_diseases = disease_counts[disease_counts >= MIN_CASES].index
df = df[df[TARGET_COL].isin(valid_diseases)]

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

# ---------------------------
# 7. TRAIN MODEL
# ---------------------------
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
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=target_encoder.classes_
))

# ---------------------------
# 9. TOP-3 PREDICTION FUNCTION
# ---------------------------
def predict_top_3(input_df):
    probs = model.predict_proba(input_df)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    return [
        (target_encoder.inverse_transform([i])[0], probs[i])
        for i in top3_idx
    ]

# ---------------------------
# 10. COWVETBOT CHATBOT
# ---------------------------
def CowVetBot():
    print("\n🐄 CowVetBot – Cow Disease Risk Predictor\n")
    print("Please provide accurate information for the best prediction.")
    print("You can leave fields blank if unknown (will use median/default values).\n")

    user_input = {}

    for col in X.columns:
        user_input[col] = input(f"{col}: ")

    input_df = pd.DataFrame([user_input])

    # Convert numerics - handle empty strings
    for col in NUMERICAL_COLS:
        # Replace empty strings with NaN without triggering warning
        input_df[col] = input_df[col].apply(lambda x: np.nan if x == '' else x)
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Convert binary columns - handle Yes/No and empty strings
    for col in BINARY_COLS:
        input_df[col] = input_df[col].str.lower().replace('', '0')
        input_df[col] = input_df[col].map({'yes': 1, 'y': 1, 'no': 0, 'n': 0, '0': 0, '1': 1})
        input_df[col] = input_df[col].fillna(0).astype(float)
    
    # Fill missing numerical values with median from training data
    for col in NUMERICAL_COLS:
        if input_df[col].isna().any():
            input_df[col] = input_df[col].fillna(df[col].median())

    # Encode categoricals safely - handle empty strings
    for col in CATEGORICAL_COLS:
        input_df[col] = input_df[col].replace('', 'unknown')
        input_df[col] = encoders[col].transform(input_df[col])

    predictions = predict_top_3(input_df)

    print("\n" + "="*60)
    print("🔍 TOP 3 POSSIBLE DISEASES:")
    print("="*60)
    
    for i, (disease, confidence) in enumerate(predictions, 1):
        print(f"\n{i}. {disease.upper()} → {confidence*100:.2f}% confidence")
        print_disease_info(disease, confidence)
        
        # Check for symptom conflicts
        conflicts = check_symptom_conflicts(disease, user_input)
        if conflicts and i == 1:  # Only show for top prediction
            print(f"\n   ⚠️  SYMPTOM CONFLICT DETECTED:")
            for conflict in conflicts:
                print(f"       • {conflict}")
    
    print("\n" + "="*60)
    if predictions[0][1] < 0.50:
        print("⚠️  LOW CONFIDENCE WARNING")
        print("    The model is uncertain about this diagnosis.")
        print("    Consult a veterinarian immediately for proper examination.")
    elif predictions[0][1] < 0.70:
        print("⚠️  MODERATE CONFIDENCE")
        print("    Veterinary confirmation strongly recommended.")
        print("    Physical examination needed to confirm diagnosis.")
    else:
        print("✅ HIGH CONFIDENCE")
        print("    Strong indication of this disease.")
        print("    Still recommend veterinary confirmation for treatment plan.")
    print("="*60)

# ---------------------------
# 11. DISEASE INFORMATION DATABASE
# ---------------------------
def print_disease_info(disease, confidence):
    """Print detailed information about predicted disease"""
    
    disease_info = {
        "Foot and Mouth Disease": {
            "key_symptoms": ["Fever", "Blisters/lesions on mouth, tongue, hooves, udder", "Lameness", "Excessive salivation", "Appetite loss"],
            "severity": "HIGHLY CONTAGIOUS - Reportable disease",
            "action": "IMMEDIATE ISOLATION and veterinary notification required"
        },
        "Mastitis": {
            "key_symptoms": ["Swollen, hot, hard udder", "Abnormal milk (clots, blood, watery)", "Reduced milk production", "Fever", "Appetite loss"],
            "severity": "Common - Can be acute or chronic",
            "action": "Veterinary examination, milk culture, antibiotic treatment if bacterial"
        },
        "Bovine Viral Diarrhea": {
            "key_symptoms": ["Diarrhea (often severe, bloody)", "Fever", "Nasal discharge", "Eye discharge", "Appetite loss", "Dehydration"],
            "severity": "CONTAGIOUS - Can cause reproductive issues",
            "action": "Isolation, supportive care, vaccination prevention"
        },
        "Parasitic Infection": {
            "key_symptoms": ["Weight loss", "Diarrhea", "Poor coat condition", "Anemia", "Bottle jaw (edema)", "Reduced growth"],
            "severity": "Common - Varies by parasite type",
            "action": "Fecal examination, deworming, pasture management"
        },
        "Respiratory Disease": {
            "key_symptoms": ["Coughing", "Labored breathing", "Nasal discharge", "Fever", "Reduced appetite", "Lethargy"],
            "severity": "Common - Can be viral or bacterial",
            "action": "Veterinary examination, possible antibiotics, improved ventilation"
        },
        "Bovine Tuberculosis": {
            "key_symptoms": ["Chronic weight loss", "Weakness", "Coughing", "Enlarged lymph nodes", "Progressive debilitation"],
            "severity": "REPORTABLE DISEASE - Zoonotic risk",
            "action": "TB testing, quarantine, regulatory veterinary involvement"
        },
        "Bovine Respiratory Disease": {
            "key_symptoms": ["Fever", "Coughing", "Nasal discharge", "Labored breathing", "Eye discharge", "Depression"],
            "severity": "Common in feedlots - Multiple pathogens",
            "action": "Early antibiotic treatment, vaccination, stress reduction"
        },
        "Bovine Coccidiosis": {
            "key_symptoms": ["Severe diarrhea (may be bloody)", "Straining to defecate", "Dehydration", "Weight loss", "Weakness"],
            "severity": "Common in young cattle - Parasitic",
            "action": "Anticoccidial drugs, supportive care, sanitation improvement"
        },
        "Johnes Disease": {
            "key_symptoms": ["Chronic diarrhea", "Progressive weight loss", "Normal appetite initially", "Submandibular edema", "Decreased milk production"],
            "severity": "Chronic - No cure, fatal",
            "action": "Testing, culling positive animals, herd management"
        },
        "Salmonellosis": {
            "key_symptoms": ["Acute diarrhea (often bloody, foul-smelling)", "Fever", "Dehydration", "Abortion in pregnant cows", "Sudden death possible"],
            "severity": "SERIOUS - Zoonotic risk",
            "action": "Isolation, fluid therapy, antibiotics (with caution), biosecurity"
        },
        "Bovine Respiratory Syncytial Virus": {
            "key_symptoms": ["Rapid breathing", "Coughing", "Nasal discharge", "Fever", "Mouth breathing", "Reluctance to move"],
            "severity": "Viral - Can be severe in young cattle",
            "action": "Supportive care, anti-inflammatories, vaccination prevention"
        }
    }
    
    info = disease_info.get(disease, None)
    
    if info:
        print(f"   Key Symptoms: {', '.join(info['key_symptoms'][:3])}...")
        print(f"   Severity: {info['severity']}")
        print(f"   Action: {info['action']}")
    else:
        print("   (Detailed information not available)")

def check_symptom_conflicts(disease, user_input):
    """Check if user's symptoms conflict with predicted disease"""
    conflicts = []
    
    # Convert user input to lowercase for comparison
    symptoms_lower = {k: str(v).lower().strip() for k, v in user_input.items()}
    
    if disease == "Foot and Mouth Disease":
        if symptoms_lower.get('Skin_Lesions') == 'no':
            conflicts.append("FMD typically presents with visible blisters/lesions, but you reported NO skin lesions")
            conflicts.append("Consider: Early-stage FMD, Mastitis, or joint/hoof injury instead")
    
    elif disease == "Bovine Viral Diarrhea":
        if symptoms_lower.get('Diarrhea') == 'no':
            conflicts.append("BVD typically causes diarrhea, but you reported NO diarrhea")
    
    elif disease == "Respiratory Disease" or disease == "Bovine Respiratory Disease":
        if symptoms_lower.get('Coughing') == 'no' and symptoms_lower.get('Labored_Breathing') == 'no':
            conflicts.append("Respiratory disease typically shows coughing or labored breathing")
    
    elif disease == "Mastitis":
        # No direct way to check udder symptoms from current inputs, but can suggest
        if symptoms_lower.get('Lameness') == 'yes':
            conflicts.append("Check udder for swelling, heat, or hardness (common with mastitis)")
    
    elif disease == "Parasitic Infection":
        if symptoms_lower.get('Diarrhea') == 'no':
            conflicts.append("Parasitic infections often cause diarrhea (though not always)")
    
    return conflicts

# ---------------------------
# 12. RUN CHATBOT
# ---------------------------
CowVetBot()