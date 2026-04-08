# ============================================================
# 🐕 Dog Disease Prediction Model + DogVetBot Chatbot
# Dataset: cow_dog_data_augmented_final.csv
# Algorithm: XGBoost (Multi-class) - IMPROVED VERSION
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

# ---------------------------
# 1. LOAD DATA
# ---------------------------
DATA_PATH = "cow_dog_data_augmented_final.csv"
df = pd.read_csv(DATA_PATH)

# Keep only dogs
df = df[df["Animal_Type"].str.lower() == "dog"].reset_index(drop=True)

print(f"\n🐕 TOTAL DOG ROWS: {len(df)}")

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
# 4. DISEASE NAME STANDARDIZATION
# ---------------------------
# Merge similar disease names
disease_mapping = {
    'Canine Parvovirus': 'Parvovirus',
    'Canine Flu': 'Canine Influenza',
    'Distemper': 'Canine Distemper',
    'Leptospirosis': 'Canine Leptospirosis',
    'Canine Infectious Hepatitis': 'Canine Hepatitis',
    'Canine Heartworm Disease': 'Heartworm Disease',
    'Bordetella Infection': 'Kennel Cough',
    'Canine Cough': 'Kennel Cough'
}

df[TARGET_COL] = df[TARGET_COL].replace(disease_mapping)

print("\n📊 Disease counts AFTER standardization:")
print(df[TARGET_COL].value_counts())

# ---------------------------
# 5. RARE DISEASE FILTERING (More lenient for dogs)
# ---------------------------
MIN_CASES = 15  # Lower threshold than cows (was 20)
disease_counts = df[TARGET_COL].value_counts()
valid_diseases = disease_counts[disease_counts >= MIN_CASES].index
df = df[df[TARGET_COL].isin(valid_diseases)]

print(f"\n🧹 After filtering (min {MIN_CASES} cases):")
print(f"Remaining rows: {len(df)}")
print(f"Remaining disease classes: {df[TARGET_COL].nunique()}")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n✅ Train-test split successful")

# ---------------------------
# 8. TRAIN MODEL (Enhanced parameters)
# ---------------------------
model = XGBClassifier(
    objective="multi:softprob",
    num_class=y.nunique(),
    max_depth=7,  # Increased from 6
    learning_rate=0.05,
    n_estimators=400,  # Increased from 300
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,  # Added regularization
    min_child_weight=3,  # Added regularization
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
# 10. SAVE MODEL
# ---------------------------
with open("dog_disease_model_improved.pkl", "wb") as f:
    pickle.dump(model, f)

with open("dog_label_encoders_improved.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("dog_target_encoder_improved.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

with open("dog_model_features_improved.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# Store medians for chatbot
with open("dog_numerical_medians.pkl", "wb") as f:
    pickle.dump(df[NUMERICAL_COLS].median().to_dict(), f)

print("\n💾 Model & encoders saved successfully!")

# ---------------------------
# 11. TOP-3 PREDICTION FUNCTION
# ---------------------------
def predict_top_3(input_df):
    probs = model.predict_proba(input_df)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    return [
        (target_encoder.inverse_transform([i])[0], probs[i])
        for i in top3_idx
    ]

# ---------------------------
# 12. DISEASE INFORMATION DATABASE
# ---------------------------
def print_disease_info(disease, confidence):
    """Print detailed information about predicted disease"""
    
    disease_info = {
        "Kennel Cough": {
            "key_symptoms": ["Harsh, dry cough (honking sound)", "Nasal discharge", "Mild fever", "Normal appetite usually", "Highly contagious"],
            "severity": "Common - Usually mild, self-limiting",
            "action": "Rest, cough suppressants, antibiotics if bacterial. Isolate from other dogs"
        },
        "Parvovirus": {
            "key_symptoms": ["Severe bloody diarrhea", "Vomiting", "Lethargy", "High fever", "Dehydration", "Loss of appetite"],
            "severity": "LIFE-THREATENING - Highly contagious, especially in puppies",
            "action": "EMERGENCY - Immediate hospitalization, IV fluids, intensive care"
        },
        "Gastroenteritis": {
            "key_symptoms": ["Vomiting", "Diarrhea", "Abdominal pain", "Loss of appetite", "Dehydration"],
            "severity": "Common - Can range from mild to severe",
            "action": "Bland diet, hydration, veterinary care if severe or prolonged"
        },
        "Lyme Disease": {
            "key_symptoms": ["Lameness (shifting leg)", "Joint swelling", "Fever", "Loss of appetite", "Lethargy", "Tick exposure"],
            "severity": "Tick-borne - Can cause chronic arthritis",
            "action": "Antibiotics (doxycycline), tick prevention, vaccination"
        },
        "Canine Distemper": {
            "key_symptoms": ["Fever", "Nasal/eye discharge", "Coughing", "Vomiting", "Diarrhea", "Seizures (late stage)"],
            "severity": "SERIOUS - Often fatal, highly contagious",
            "action": "Supportive care, isolation. Prevention through vaccination is critical"
        },
        "Arthritis": {
            "key_symptoms": ["Lameness", "Stiffness (worse after rest)", "Difficulty rising", "Reduced activity", "Pain when touched"],
            "severity": "Chronic - Common in older/large breed dogs",
            "action": "Pain management, weight control, joint supplements, physical therapy"
        },
        "Canine Influenza": {
            "key_symptoms": ["Coughing", "Nasal discharge", "Fever", "Lethargy", "Loss of appetite", "Highly contagious"],
            "severity": "Contagious - Usually mild but can be severe",
            "action": "Supportive care, rest, isolation, antibiotics if secondary infection"
        },
        "Canine Leptospirosis": {
            "key_symptoms": ["Fever", "Vomiting", "Lethargy", "Jaundice", "Kidney/liver failure", "Blood in urine"],
            "severity": "SERIOUS - Zoonotic (can infect humans)",
            "action": "Antibiotics, supportive care, vaccination prevention"
        },
        "Canine Hepatitis": {
            "key_symptoms": ["Fever", "Lethargy", "Loss of appetite", "Vomiting", "Jaundice", "Abdominal pain"],
            "severity": "Serious - Can be acute or chronic",
            "action": "Supportive care, vaccination prevention (CAV-1/CAV-2)"
        },
        "Tick-Borne Disease": {
            "key_symptoms": ["Fever", "Lethargy", "Joint pain/swelling", "Loss of appetite", "Tick attachment"],
            "severity": "Variable - Depends on specific pathogen",
            "action": "Testing, antibiotics, tick prevention, regular screening"
        },
        "Salmonellosis": {
            "key_symptoms": ["Diarrhea (often bloody)", "Vomiting", "Fever", "Abdominal pain", "Dehydration"],
            "severity": "Serious - Zoonotic risk",
            "action": "Supportive care, hydration, antibiotics in severe cases, sanitation"
        },
        "Pancreatitis": {
            "key_symptoms": ["Vomiting", "Abdominal pain (hunched posture)", "Loss of appetite", "Diarrhea", "Fever"],
            "severity": "Serious - Can be life-threatening",
            "action": "NPO (nothing by mouth), IV fluids, pain management, low-fat diet"
        },
        "Allergic Rhinitis": {
            "key_symptoms": ["Sneezing", "Nasal discharge", "Itchy eyes/nose", "Reverse sneezing", "Seasonal pattern"],
            "severity": "Mild - Chronic condition",
            "action": "Antihistamines, allergen avoidance, air filtration"
        },
        "Chronic Bronchitis": {
            "key_symptoms": ["Persistent dry cough (>2 months)", "Exercise intolerance", "Normal appetite", "Cough worse with excitement"],
            "severity": "Chronic - Progressive condition",
            "action": "Cough suppressants, bronchodilators, weight management, avoid irritants"
        },
        "Heartworm Disease": {
            "key_symptoms": ["Coughing", "Exercise intolerance", "Weight loss", "Labored breathing", "Heart failure signs"],
            "severity": "SERIOUS - Life-threatening if untreated",
            "action": "Heartworm treatment protocol, strict rest, prevention is key"
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
    warnings = []
    
    symptoms_lower = {k: str(v).lower().strip() for k, v in user_input.items()}
    
    if disease == "Parvovirus":
        if symptoms_lower.get('Diarrhea') == 'no' and symptoms_lower.get('Vomiting') == 'no':
            conflicts.append("Parvovirus typically causes severe vomiting AND diarrhea")
        elif symptoms_lower.get('Diarrhea') == '':
            warnings.append("CHECK: Is there bloody diarrhea? This is critical for Parvo diagnosis")
        if symptoms_lower.get('Age', '0') != '' and float(symptoms_lower.get('Age', 6)) < 1:
            warnings.append("URGENT: Puppies under 1 year are at highest risk - immediate vet care needed")
    
    elif disease == "Kennel Cough":
        if symptoms_lower.get('Coughing') == 'no':
            conflicts.append("Kennel Cough's defining symptom is a harsh, dry cough")
        if symptoms_lower.get('Diarrhea') == 'yes' or symptoms_lower.get('Vomiting') == 'yes':
            warnings.append("GI symptoms uncommon with Kennel Cough - consider other diagnoses")
    
    elif disease == "Lyme Disease":
        if symptoms_lower.get('Lameness') == 'no':
            conflicts.append("Lyme Disease typically causes lameness/joint pain")
        if symptoms_lower.get('Lameness') == '':
            warnings.append("CHECK: Is there shifting leg lameness? This is classic for Lyme Disease")
    
    elif disease == "Canine Distemper":
        if symptoms_lower.get('Nasal_Discharge') == 'no' and symptoms_lower.get('Eye_Discharge') == 'no':
            conflicts.append("Distemper usually presents with nasal and eye discharge")
        warnings.append("CRITICAL: Distemper can be fatal - immediate veterinary care required")
    
    elif disease == "Arthritis":
        if symptoms_lower.get('Lameness') == 'no':
            conflicts.append("Arthritis typically causes lameness or stiffness")
        if symptoms_lower.get('Age', '0') != '' and float(symptoms_lower.get('Age', 1)) < 3:
            warnings.append("Arthritis uncommon in young dogs - consider injury or developmental issue")
    
    elif disease == "Gastroenteritis":
        if symptoms_lower.get('Diarrhea') == 'no' and symptoms_lower.get('Vomiting') == 'no':
            conflicts.append("Gastroenteritis typically causes vomiting and/or diarrhea")
    
    elif disease in ["Canine Influenza"]:
        if symptoms_lower.get('Coughing') == 'no':
            conflicts.append("Canine Influenza typically involves coughing")
    
    return conflicts, warnings

# ---------------------------
# 13. VACCINATION STATUS DATABASE
# ---------------------------
VACCINE_PREVENTABLE_DISEASES = {
    "Parvovirus": {
        "vaccine": "DHPP/DA2PP (Core Vaccine)",
        "schedule": "Puppies: 6-8, 10-12, 14-16 weeks. Booster at 1 year, then every 3 years",
        "critical": True,
        "efficacy": "95-99% effective when fully vaccinated"
    },
    "Canine Distemper": {
        "vaccine": "DHPP/DA2PP (Core Vaccine)",
        "schedule": "Puppies: 6-8, 10-12, 14-16 weeks. Booster at 1 year, then every 3 years",
        "critical": True,
        "efficacy": "95-99% effective"
    },
    "Canine Hepatitis": {
        "vaccine": "DHPP/DA2PP (Core Vaccine - Adenovirus-2)",
        "schedule": "Puppies: 6-8, 10-12, 14-16 weeks. Booster at 1 year, then every 3 years",
        "critical": True,
        "efficacy": "Very effective"
    },
    "Kennel Cough": {
        "vaccine": "Bordetella (Non-core) - Injectable or Intranasal",
        "schedule": "Every 6-12 months if high risk (boarding, daycare, dog parks)",
        "critical": False,
        "efficacy": "60-70% effective (multiple strains exist)"
    },
    "Canine Influenza": {
        "vaccine": "CIV H3N8/H3N2 (Non-core)",
        "schedule": "2 doses 2-4 weeks apart, then annually if at risk",
        "critical": False,
        "efficacy": "Reduces severity and transmission"
    },
    "Canine Leptospirosis": {
        "vaccine": "Leptospira (Non-core, often in DHLPP)",
        "schedule": "Puppies: 12 weeks and 16 weeks. Annual boosters required",
        "critical": False,
        "efficacy": "70-80% effective, annual boosters essential"
    },
    "Lyme Disease": {
        "vaccine": "Lyme (Non-core)",
        "schedule": "2 doses 2-4 weeks apart at 12+ weeks. Annual boosters",
        "critical": False,
        "efficacy": "Tick prevention is primary defense. Vaccine helps in endemic areas"
    }
}

def check_vaccination_status(disease, age):
    """Check if disease is vaccine-preventable and provide guidance"""
    
    if disease not in VACCINE_PREVENTABLE_DISEASES:
        return None
    
    vax_info = VACCINE_PREVENTABLE_DISEASES[disease]
    age_float = float(age) if age else 0
    
    result = {
        "is_preventable": True,
        "vaccine_name": vax_info["vaccine"],
        "schedule": vax_info["schedule"],
        "is_critical": vax_info["critical"],
        "efficacy": vax_info["efficacy"],
        "age_appropriate": True,
        "warnings": []
    }
    
    # Age-specific warnings
    if age_float > 0:
        if age_float < 0.5:  # Less than 6 months
            if disease in ["Parvovirus", "Canine Distemper", "Canine Hepatitis"]:
                result["warnings"].append(
                    f"Puppy under 6 months: May not be fully vaccinated yet (requires 3 doses by 16 weeks)"
                )
                result["age_appropriate"] = False
        
        elif 0.5 <= age_float < 1.5:  # 6-18 months
            result["warnings"].append(
                "Critical vaccination period: Ensure all puppy shots and 1-year booster are complete"
            )
        
        elif age_float > 7:  # Senior dogs
            result["warnings"].append(
                "Senior dog: Verify vaccination status is up-to-date (immunity can wane)"
            )
    
    return result

def print_vaccination_info(disease, age):
    """Print vaccination information for predicted disease"""
    
    vax_status = check_vaccination_status(disease, age)
    
    if not vax_status:
        return
    
    print(f"\n   💉 VACCINATION INFORMATION:")
    print(f"       ✓ This disease is VACCINE-PREVENTABLE")
    print(f"       • Vaccine: {vax_status['vaccine_name']}")
    print(f"       • Efficacy: {vax_status['efficacy']}")
    
    if vax_status['is_critical']:
        print(f"       • Status: ⚠️  CORE VACCINE (Required for all dogs)")
    else:
        print(f"       • Status: Non-core (Risk-based vaccination)")
    
    print(f"       • Schedule: {vax_status['schedule']}")
    
    if vax_status['warnings']:
        for warning in vax_status['warnings']:
            print(f"       ⚠️  {warning}")
    
    if not vax_status['age_appropriate'] or vax_status['warnings']:
        print(f"       📋 ACTION: Verify vaccination records with your veterinarian")
def DogVetBot():
    print("\n🐕 DogVetBot – Dog Disease Risk Predictor\n")
    print("Please provide accurate information for the best prediction.")
    print("You can leave fields blank if unknown (will use median/default values).\n")

    user_input = {}

    for col in X.columns:
        user_input[col] = input(f"{col}: ")

    input_df = pd.DataFrame([user_input])

    # Convert numerics - handle empty strings
    for col in NUMERICAL_COLS:
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
        
        # Add vaccination information
        if i <= 2:  # Show for top 2 predictions
            print_vaccination_info(disease, user_input.get('Age', '0'))
        
        # Check for symptom conflicts and warnings
        conflicts, warnings = check_symptom_conflicts(disease, user_input)
        if i == 1:  # Only show for top prediction
            if conflicts:
                print(f"\n   ⚠️  SYMPTOM CONFLICT DETECTED:")
                for conflict in conflicts:
                    print(f"       • {conflict}")
            if warnings:
                print(f"\n   💡 CLINICAL NOTES:")
                for warning in warnings:
                    print(f"       • {warning}")
    
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
    
    # Special alert for puppies with GI symptoms
    age_str = user_input.get('Age', '0')
    try:
        age = float(age_str) if age_str else 0
    except:
        age = 0
    
    has_gi_symptoms = (
        user_input.get('Vomiting', '').lower() in ['yes', 'y', '1'] or
        user_input.get('Diarrhea', '').lower() in ['yes', 'y', '1'] or
        user_input.get('Appetite_Loss', '').lower() in ['yes', 'y', '1']
    )
    
    diarrhea_unknown = user_input.get('Diarrhea', '').strip() == ''
    
    # Check if Parvovirus is in top 3
    parvo_in_top3 = any('Parvovirus' in disease for disease, _ in predictions)
    
    if age > 0 and age < 1.5 and has_gi_symptoms and parvo_in_top3:
        print("\n" + "="*60)
        print("🚨 URGENT PUPPY ALERT - PARVOVIRUS RISK")
        print("="*60)
        print("⚠️  Puppies under 18 months with GI symptoms are at HIGH RISK for Parvovirus")
        print("⚠️  Parvovirus is LIFE-THREATENING and requires immediate emergency care")
        if diarrhea_unknown:
            print("\n❗ CRITICAL: You did not specify diarrhea status")
            print("   • If BLOODY DIARRHEA is present → EMERGENCY VET NOW")
            print("   • Parvovirus causes severe, bloody, foul-smelling diarrhea")
        print("\n💉 Prevention: Ensure puppy is fully vaccinated (3 rounds by 16 weeks)")
        print("🏥 If ANY doubt: GO TO EMERGENCY VET - Do not wait")
        print("="*60)
    
    print("="*60)

# ---------------------------
# 15. RUN CHATBOT
# ---------------------------
DogVetBot()