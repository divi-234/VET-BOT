# ============================================================
# VetBot Flask API - Complete with Clinical Logic
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ============================================================
# SAFE LABEL ENCODER
# ============================================================
from sklearn.preprocessing import LabelEncoder

class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        return [
            self.classes_.tolist().index(x) if x in self.classes_ else -1
            for x in y
        ]

# ============================================================
# LOAD MODELS
# ============================================================

# ---------- DOG ----------
try:
    with open("dog_disease_model_improved.pkl", "rb") as f:
        dog_model = pickle.load(f)
    with open("dog_label_encoders_improved.pkl", "rb") as f:
        dog_encoders = pickle.load(f)
    with open("dog_target_encoder_improved.pkl", "rb") as f:
        dog_target_encoder = pickle.load(f)
    with open("dog_model_features_improved.pkl", "rb") as f:
        dog_features = pickle.load(f)
    with open("dog_numerical_medians.pkl", "rb") as f:
        dog_medians = pickle.load(f)
    print("✅ Dog model loaded successfully")
except Exception as e:
    print(f"❌ Error loading dog model: {e}")
    dog_model = None

# ---------- COW ----------
try:
    with open("cow_disease_model.pkl", "rb") as f:
        cow_model = pickle.load(f)
    with open("cow_label_encoders.pkl", "rb") as f:
        cow_encoders = pickle.load(f)
    with open("cow_target_encoder.pkl", "rb") as f:
        cow_target_encoder = pickle.load(f)
    with open("cow_model_features.pkl", "rb") as f:
        cow_features = pickle.load(f)
    with open("cow_numerical_medians.pkl", "rb") as f:
        cow_medians = pickle.load(f)
    print("✅ Cow model loaded successfully")
except Exception as e:
    print(f"❌ Error loading cow model: {e}")
    cow_model = None

# ---------- CAT — Structured model (primary, same pattern as dog/cow) ----------
cat_structured_model   = None
cat_structured_encoders = None
cat_structured_target  = None
cat_structured_features = None
cat_structured_medians  = None

try:
    with open("cat_disease_model_improved.pkl", "rb") as f:
        cat_structured_model = pickle.load(f)
    with open("cat_label_encoders_improved.pkl", "rb") as f:
        cat_structured_encoders = pickle.load(f)
    with open("cat_target_encoder_improved.pkl", "rb") as f:
        cat_structured_target = pickle.load(f)
    with open("cat_model_features_improved.pkl", "rb") as f:
        cat_structured_features = pickle.load(f)
    with open("cat_numerical_medians.pkl", "rb") as f:
        cat_structured_medians = pickle.load(f)
    print("✅ Cat structured model loaded successfully")
except Exception as e:
    print(f"⚠️  Cat structured model not found ({e}) — will fall back to text-based model")

# ---------- CAT — Text-based model (fallback) ----------
cat_model      = None
cat_tfidf      = None
cat_mlb        = None
cat_text_based = False

try:
    with open("xgb_cat_model.pkl", "rb") as f:
        cat_model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        cat_tfidf = pickle.load(f)
    with open("mlb.pkl", "rb") as f:
        cat_mlb = pickle.load(f)
    print("✅ Cat text-based model loaded successfully (fallback)")
    cat_text_based = True
except Exception as e:
    print(f"⚠️  Cat text-based model not found: {e}")

# ============================================================
# CONFIGURATION
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

# ============================================================
# BREED-SPECIFIC RISKS
# ============================================================
BREED_RISKS = {
    'labrador': ['Foreign body ingestion', 'Dietary indiscretion', 'Pancreatitis'],
    'golden retriever': ['Foreign body ingestion', 'Dietary indiscretion'],
    'german shepherd': ['Gastric dilation-volvulus', 'Exocrine pancreatic insufficiency'],
    'bulldog': ['Brachycephalic issues', 'Skin infections'],
    'beagle': ['Dietary indiscretion', 'Foreign body ingestion'],
    'boxer': ['Cardiomyopathy', 'Tumors'],
    'dachshund': ['Intervertebral disc disease', 'Back problems'],
    'poodle': ['Addison\'s disease', 'Gastric issues'],
    'yorkshire': ['Hypoglycemia', 'Portosystemic shunt'],
    'chihuahua': ['Hypoglycemia', 'Dental issues']
}

COW_BREED_RISKS = {
    'holstein': ['Mastitis', 'Milk fever (hypocalcemia)', 'Displaced abomasum', 'Ketosis'],
    'jersey': ['Milk fever', 'Mastitis', 'Dystocia (calving difficulty)'],
    'angus': ['Pinkeye', 'Foot rot', 'Respiratory disease'],
    'hereford': ['Pinkeye', 'Cancer eye', 'Foot problems'],
    'charolais': ['Dystocia', 'Musculoskeletal issues'],
    'simmental': ['Respiratory disease', 'Digestive issues'],
    'brown swiss': ['Mastitis', 'Foot problems'],
    'guernsey': ['Mastitis', 'Milk fever'],
    'limousin': ['Dystocia', 'Muscular issues'],
    'brahman': ['Heat tolerance', 'Tick resistance (lower disease risk)']
}

CAT_BREED_RISKS = {
    'persian': ['Polycystic kidney disease', 'Respiratory issues', 'Eye problems'],
    'siamese': ['Asthma', 'Dental issues', 'Crossed eyes'],
    'maine coon': ['Hypertrophic cardiomyopathy', 'Hip dysplasia', 'Spinal muscular atrophy'],
    'bengal': ['Hypertrophic cardiomyopathy', 'Progressive retinal atrophy'],
    'ragdoll': ['Hypertrophic cardiomyopathy', 'Polycystic kidney disease'],
    'british shorthair': ['Hypertrophic cardiomyopathy', 'Obesity', 'Polycystic kidney disease'],
    'sphynx': ['Hypertrophic cardiomyopathy', 'Skin conditions', 'Respiratory infections'],
    'abyssinian': ['Progressive retinal atrophy', 'Renal amyloidosis', 'Gingivitis'],
    'scottish fold': ['Osteochondrodysplasia', 'Arthritis', 'Ear infections'],
    'russian blue': ['Generally healthy', 'Obesity prone']
}

# ============================================================
# DISEASE INFORMATION
# ============================================================
DOG_DISEASE_INFO = {
    "Kennel Cough": {
        "severity": "Common - Usually mild",
        "key_symptoms": "Harsh DRY cough, nasal discharge",
        "primary_symptom": "Coughing",
        "action": "Rest, cough suppressants, isolate from other dogs",
        "vaccine": "Bordetella (Non-core, every 6-12 months if at risk)",
        "requires_cough": True
    },
    "Parvovirus": {
        "severity": "LIFE-THREATENING - Emergency",
        "key_symptoms": "Severe bloody diarrhea, vomiting, lethargy",
        "primary_symptom": "Diarrhea",
        "action": "IMMEDIATE hospitalization, IV fluids, intensive care",
        "vaccine": "DHPP Core Vaccine (puppies: 6-8, 10-12, 14-16 weeks, then 1 year, then every 3 years)",
        "age_risk": "high_if_puppy",
        "requires_diarrhea": True
    },
    "Gastroenteritis": {
        "severity": "Common - Varies",
        "key_symptoms": "Vomiting, diarrhea, abdominal discomfort",
        "primary_symptom": "Vomiting or Diarrhea",
        "action": "Withhold food 12-24 hours (if vet-approved), then bland diet, hydration, vet if severe",
        "vaccine": "None - Vaccines exist for some infectious causes (e.g., parvovirus)",
        "common_causes": "Dietary indiscretion, foreign body, infection"
    },
    "Lyme Disease": {
        "severity": "Tick-borne - Chronic",
        "key_symptoms": "Lameness, joint swelling, fever",
        "primary_symptom": "Lameness",
        "action": "Antibiotics, tick prevention",
        "vaccine": "Lyme Vaccine (Non-core, 2 doses 2-4 weeks apart at 12+ weeks, annual boosters)",
        "requires_lameness": True
    },
    "Canine Distemper": {
        "severity": "SERIOUS - Often fatal",
        "key_symptoms": "Fever, discharge, coughing, seizures",
        "primary_symptom": "Multiple respiratory symptoms",
        "action": "Supportive care, isolation, prevention via vaccination",
        "vaccine": "DHPP Core Vaccine (same schedule as Parvovirus)"
    },
    "Canine Influenza": {
        "severity": "Contagious - Usually mild but can be severe",
        "key_symptoms": "Coughing, sneezing, fever, nasal discharge",
        "primary_symptom": "Coughing",
        "action": "Supportive care, rest, isolation, antibiotics if secondary infection",
        "vaccine": "CIV H3N8/H3N2 (Non-core, 2 doses 2-4 weeks apart, then annually if at risk)",
        "requires_cough": True
    },
    "Canine Leptospirosis": {
        "severity": "SERIOUS - Zoonotic (can infect humans)",
        "key_symptoms": "Fever, vomiting, jaundice, kidney/liver failure",
        "primary_symptom": "Vomiting",
        "action": "Antibiotics, supportive care, isolation",
        "vaccine": "Leptospira Vaccine (Often in DHLPP, 2 doses at 12 & 16 weeks, annual boosters essential)"
    },
    "Canine Hepatitis": {
        "severity": "Serious - Can be acute or chronic",
        "key_symptoms": "Fever, lethargy, jaundice, abdominal pain",
        "primary_symptom": "Lethargy",
        "action": "Supportive care, hospitalization if severe",
        "vaccine": "DHPP Core Vaccine - Adenovirus-2 (same schedule as Parvovirus)"
    },
    "Arthritis": {
        "severity": "Chronic - Common in older/large breeds",
        "key_symptoms": "Lameness, stiffness, difficulty rising",
        "primary_symptom": "Lameness",
        "action": "Pain management, weight control, joint supplements, physical therapy",
        "vaccine": "None",
        "requires_lameness": True
    }
}

CAT_DISEASE_INFO = {
    "Feline Asthma": {
        "severity": "SERIOUS - Chronic condition, can be life-threatening",
        "key_symptoms": "Labored breathing, wheezing, coughing, lethargy",
        "primary_symptom": "Labored breathing",
        "action": "IMMEDIATE veterinary care. Oxygen therapy if severe. Long-term management with bronchodilators and corticosteroids.",
        "vaccine": "None"
    },
    "Lower Respiratory Infection": {
        "severity": "SERIOUS - Requires immediate treatment",
        "key_symptoms": "Labored breathing, fever, lethargy, nasal discharge, coughing",
        "primary_symptom": "Labored breathing with fever",
        "action": "Urgent veterinary care. Chest X-rays, antibiotics, oxygen therapy if needed. Monitor closely.",
        "vaccine": "None - FVRCP protects against some viral causes"
    },
    "Severe Allergic Reaction": {
        "severity": "EMERGENCY - Can cause anaphylaxis",
        "key_symptoms": "Labored breathing, skin lesions, swelling, lethargy",
        "primary_symptom": "Acute respiratory distress with skin signs",
        "action": "EMERGENCY - Immediate veterinary care. Antihistamines, corticosteroids, oxygen if needed.",
        "vaccine": "None"
    },
    "Respiratory Distress": {
        "severity": "LIFE-THREATENING EMERGENCY",
        "key_symptoms": "Labored breathing, open-mouth breathing, blue gums",
        "primary_symptom": "Difficulty breathing",
        "action": "EMERGENCY - Immediate transport to veterinary hospital. Oxygen therapy critical. Can be caused by asthma, infection, pleural effusion, heart failure, or trauma.",
        "vaccine": "None"
    },
    "Feline Panleukopenia": {
        "severity": "LIFE-THREATENING - Highly contagious",
        "key_symptoms": "Severe vomiting, diarrhea, lethargy, fever, dehydration",
        "primary_symptom": "Vomiting and diarrhea",
        "action": "IMMEDIATE emergency veterinary care. IV fluids, antibiotics, isolation. High mortality without treatment.",
        "vaccine": "FVRCP Core Vaccine (kittens: 6-8, 10-12, 14-16 weeks, then annual or triennial)"
    },
    "Upper Respiratory Infection": {
        "severity": "Common - Usually mild but can be serious in kittens",
        "key_symptoms": "Sneezing, nasal discharge, eye discharge, fever, loss of appetite",
        "primary_symptom": "Sneezing or nasal discharge",
        "action": "Veterinary exam. Supportive care, antibiotics if bacterial, isolation from other cats",
        "vaccine": "FVRCP Core Vaccine protects against viral causes (Herpesvirus, Calicivirus)",
        "calicivirus_note": "Calicivirus can cause limping/lameness ('limping kitten syndrome')"
    },
    "Viral-associated Lameness": {
        "severity": "Usually self-limiting",
        "key_symptoms": "Lameness with URI signs (eye/nasal discharge)",
        "primary_symptom": "Lameness with respiratory signs",
        "action": "Veterinary exam to rule out injury. Supportive care. Usually resolves with URI treatment.",
        "vaccine": "FVRCP Core Vaccine (Calicivirus prevention)"
    },
    "Feline Conjunctivitis": {
        "severity": "Common - Can be chronic",
        "key_symptoms": "Red/swollen eyes, eye discharge, squinting, rubbing eyes",
        "primary_symptom": "Eye discharge or redness",
        "action": "Veterinary exam for diagnosis. Topical antibiotics or antivirals, keep eyes clean",
        "vaccine": "FVRCP vaccine helps prevent viral causes"
    },
    "Urinary Tract Infection": {
        "severity": "Common - Can become serious if untreated",
        "key_symptoms": "Frequent urination, straining, blood in urine, urinating outside litter box",
        "primary_symptom": "Urination issues",
        "action": "Veterinary exam required. Urinalysis, antibiotics, increased water intake. Monitor for blockage in males.",
        "vaccine": "None"
    },
    "Gastritis": {
        "severity": "Common - Varies from mild to severe",
        "key_symptoms": "Vomiting, loss of appetite, lethargy, abdominal pain",
        "primary_symptom": "Vomiting",
        "action": "Withhold food 12-24 hours, small bland meals, veterinary exam if severe or persistent",
        "vaccine": "None"
    },
    "Worm Infestation": {
        "severity": "Common - Serious in kittens",
        "key_symptoms": "Weight loss, pot-bellied appearance, diarrhea, vomiting, visible worms",
        "primary_symptom": "Weight loss or visible worms",
        "action": "Fecal exam to identify parasite type. Veterinarian-prescribed dewormer. Treat all cats in household.",
        "vaccine": "None - Prevention through regular deworming"
    },
    "Skin Allergy": {
        "severity": "Common - Chronic condition",
        "key_symptoms": "Excessive scratching, hair loss, red skin, scabs, over-grooming",
        "primary_symptom": "Scratching or hair loss",
        "action": "Veterinary exam to identify allergen (food, flea, environmental). Antihistamines, diet change, flea control.",
        "vaccine": "None"
    },
    "Dehydration": {
        "severity": "SERIOUS - Can be life-threatening",
        "key_symptoms": "Sunken eyes, dry gums, skin tenting, lethargy, loss of appetite",
        "primary_symptom": "Lethargy and dry mouth",
        "action": "IMMEDIATE veterinary care. Subcutaneous or IV fluids. Identify underlying cause.",
        "vaccine": "None - Symptom of underlying condition"
    },
    "Anemia": {
        "severity": "SERIOUS - Requires investigation",
        "key_symptoms": "Pale gums, weakness, rapid breathing, loss of appetite, lethargy",
        "primary_symptom": "Pale gums or weakness",
        "action": "URGENT veterinary exam. Blood work to determine cause. May need transfusion if severe.",
        "vaccine": "None - Can be caused by parasites, FeLV, FIV, or other conditions"
    },
    "Chronic Kidney Disease": {
        "severity": "SERIOUS - Progressive condition common in older cats",
        "key_symptoms": "Increased thirst/urination, weight loss, poor appetite, vomiting, bad breath",
        "primary_symptom": "Increased thirst and urination",
        "action": "Veterinary exam. Blood work and urinalysis. Special renal diet, subcutaneous fluids, medications. Regular monitoring.",
        "vaccine": "None"
    }
}

COW_DISEASE_INFO = {
    "Foot and Mouth Disease": {
        "severity": "HIGHLY CONTAGIOUS - Reportable",
        "key_symptoms": "Fever, blisters on mouth/hooves, lameness, excessive drooling",
        "primary_symptom": "Skin lesions or Lameness",
        "action": "IMMEDIATE isolation and veterinary notification - THIS IS A REPORTABLE DISEASE",
        "vaccine": "Available in some regions",
        "requires_lesions_or_lameness": True,
        "reportable": True
    },
    "Mastitis": {
        "severity": "Common - Acute/chronic",
        "key_symptoms": "Swollen udder, abnormal milk, fever, reduced milk production",
        "primary_symptom": "Udder abnormalities",
        "action": "Veterinary exam required. Veterinarian-prescribed antibiotics if bacterial, milk out affected quarter, anti-inflammatory drugs",
        "vaccine": "None - Prevention through hygiene and milking practices",
        "common_in": "dairy_breeds"
    },
    "Bovine Viral Diarrhea": {
        "severity": "CONTAGIOUS - Can cause reproductive issues",
        "key_symptoms": "Diarrhea, fever, nasal discharge, respiratory signs",
        "primary_symptom": "Diarrhea or respiratory signs",
        "action": "Isolation, supportive care, veterinary consultation for herd vaccination plan, test and cull persistently infected animals",
        "vaccine": "Available - Essential for breeding herds"
    },
    "Parasitic Infection": {
        "severity": "Common - Severity varies by parasite load and animal age",
        "key_symptoms": "Weight loss, diarrhea, poor coat condition, anemia, 'bottle jaw'",
        "primary_symptom": "Weight loss or poor condition",
        "action": "Fecal exam to identify parasite, veterinarian-prescribed appropriate dewormer, pasture rotation, monitor fecal egg counts",
        "vaccine": "None - Prevention through strategic deworming"
    },
    "Respiratory Disease": {
        "severity": "Common - Ranges from mild to severe (serious in young calves)",
        "key_symptoms": "Coughing, labored breathing, nasal discharge, fever",
        "primary_symptom": "Coughing or labored breathing",
        "action": "Veterinary examination required. Veterinarian-prescribed antibiotics if bacterial, anti-inflammatory drugs, improved ventilation, isolate sick animals",
        "vaccine": "Various respiratory vaccines available (IBR, PI3, BRSV, Mannheimia)",
        "requires_respiratory_signs": True
    },
    "Milk Fever": {
        "severity": "EMERGENCY - Common in high-producing dairy cows",
        "key_symptoms": "Weakness, inability to stand, muscle tremors, cold extremities",
        "primary_symptom": "Weakness or inability to stand",
        "action": "IMMEDIATE IV calcium gluconate administration - can be fatal within hours",
        "vaccine": "None - Prevention through dietary calcium management",
        "age_risk": "recently_calved",
        "common_in": "dairy_breeds"
    },
    "Bloat": {
        "severity": "LIFE-THREATENING EMERGENCY - Can be fatal within hours",
        "key_symptoms": "Severe abdominal distension (left side), labored breathing, distress, kicking at belly",
        "primary_symptom": "Abdominal distension/bloating",
        "action": "EMERGENCY - Immediate veterinary intervention. Pass stomach tube or emergency trocar. Keep animal standing if possible. Bloat relief medication. DO NOT DELAY.",
        "vaccine": "None - Prevention through diet management (avoid sudden diet changes, legume bloat)"
    },
    "Displaced Abomasum": {
        "severity": "Common post-calving - Requires surgery",
        "key_symptoms": "Reduced appetite, decreased milk production, 'ping' sound on left side",
        "primary_symptom": "Loss of appetite post-calving",
        "action": "Veterinary exam, surgical correction usually required",
        "vaccine": "None - Prevention through proper transition cow management",
        "age_risk": "recently_calved",
        "common_in": "dairy_breeds"
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_data(data, animal_type):
    """Preprocess user input data for model prediction (dog / cow / cat-structured)"""
    df = pd.DataFrame([data])

    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)

    if 'Body_Temperature' in df.columns:
        df['Body_Temperature'] = (
            df['Body_Temperature']
            .astype(str)
            .str.replace('°C', '', regex=False)
            .str.strip()
        )
        df['Body_Temperature'] = pd.to_numeric(df['Body_Temperature'], errors='coerce')

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].map({
                'yes': 1, 'y': 1, '1': 1,
                'no': 0, 'n': 0, '0': 0, '': 0
            })
            df[col] = df[col].fillna(0).astype(int)

    # Pick the right medians / encoders based on animal_type
    if animal_type == 'dog':
        medians  = dog_medians
        encoders = dog_encoders
    elif animal_type == 'cow':
        medians  = cow_medians
        encoders = cow_encoders
    else:                          # cat (structured)
        medians  = cat_structured_medians
        encoders = cat_structured_encoders

    if medians:
        for col in NUMERICAL_COLS:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.nan if x == '' else x)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    df[col] = df[col].fillna(medians.get(col, 0))

    if encoders:
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].replace('', 'unknown')
                try:
                    df[col] = encoders[col].transform(df[col])
                except Exception:
                    df[col] = 0

    return df


def apply_clinical_logic(predictions, user_data, animal_type):
    """Apply clinical logic to adjust predictions based on symptoms"""

    age = float(user_data.get('Age', 0))
    has_coughing          = user_data.get('Coughing',          '').lower() in ['yes', 'y', '1']
    has_diarrhea          = user_data.get('Diarrhea',          '').lower() in ['yes', 'y', '1']
    has_vomiting          = user_data.get('Vomiting',          '').lower() in ['yes', 'y', '1']
    has_lameness          = user_data.get('Lameness',          '').lower() in ['yes', 'y', '1']
    has_labored_breathing = user_data.get('Labored_Breathing', '').lower() in ['yes', 'y', '1']
    has_skin_lesions      = user_data.get('Skin_Lesions',      '').lower() in ['yes', 'y', '1']
    has_appetite_loss     = user_data.get('Appetite_Loss',     '').lower() in ['yes', 'y', '1']
    has_eye_discharge     = user_data.get('Eye_Discharge',     '').lower() in ['yes', 'y', '1']
    has_nasal_discharge   = user_data.get('Nasal_Discharge',   '').lower() in ['yes', 'y', '1']

    all_symptoms = ' '.join([
        str(user_data.get('Symptom_1', '')),
        str(user_data.get('Symptom_2', '')),
        str(user_data.get('Symptom_3', '')),
        str(user_data.get('Symptom_4', ''))
    ]).lower()

    # ----------------------------------------------------------
    # DOG clinical adjustments
    # ----------------------------------------------------------
    if animal_type == 'dog':
        has_bloody_diarrhea = any(kw in all_symptoms for kw in ['bloody', 'blood', 'hemorrhagic'])
        adjusted = []

        for pred in predictions:
            name       = pred['name']
            confidence = pred['confidence']

            if name == "Parvovirus":
                if age <= 1.5 and has_diarrhea:
                    confidence *= 3.0 if has_bloody_diarrhea else 2.0
                elif age > 1.5 and not has_diarrhea:
                    confidence *= 0.05
                elif not has_diarrhea:
                    confidence *= 0.2

            if name == "Kennel Cough" and not has_coughing:
                confidence *= 0.01

            if name in ["Lyme Disease", "Arthritis"] and not has_lameness:
                confidence *= 0.1

            if name == "Canine Influenza" and not has_coughing:
                confidence *= 0.1

            adjusted.append({'name': name, 'confidence': confidence})

        predictions = adjusted

    # ----------------------------------------------------------
    # CAT clinical adjustments  (includes pre-model emergency overrides)
    # ----------------------------------------------------------
    elif animal_type == 'cat':
        breed = user_data.get('Breed', '').lower()
        is_asthma_prone = any(b in breed for b in ['siamese', 'bengal', 'oriental'])

        has_sneezing     = any(kw in all_symptoms for kw in ['sneez', 'sniffles'])
        has_scratching   = any(kw in all_symptoms for kw in ['scratch', 'itch', 'lick'])
        has_urinary_issues = any(kw in all_symptoms for kw in ['urinat', 'litter box', 'strain', 'blood in urine'])
        duration = float(user_data.get('Duration', 999))
        is_acute = duration <= 3

        # ==============================================================
        # EMERGENCY OVERRIDE #1 — LABORED BREATHING
        # ==============================================================
        if has_labored_breathing:
            emergency_predictions = []

            if is_asthma_prone:
                emergency_predictions.append({'name': 'Feline Asthma',             'confidence': 70.0, 'emergency': True})
                emergency_predictions.append({'name': 'Lower Respiratory Infection','confidence': 20.0, 'emergency': True})
                emergency_predictions.append({'name': 'Respiratory Distress',      'confidence': 10.0, 'emergency': True})
            else:
                emergency_predictions.append({'name': 'Respiratory Distress',      'confidence': 50.0, 'emergency': True})
                emergency_predictions.append({'name': 'Lower Respiratory Infection','confidence': 30.0, 'emergency': True})
                if has_skin_lesions:
                    emergency_predictions.append({'name': 'Severe Allergic Reaction','confidence': 20.0, 'emergency': True})
                else:
                    emergency_predictions.append({'name': 'Feline Asthma',          'confidence': 20.0, 'emergency': True})

            return emergency_predictions[:3]

        # ==============================================================
        # EMERGENCY OVERRIDE #2 — PANLEUKOPENIA (kitten + vomiting + diarrhea)
        # ==============================================================
        if has_vomiting and has_diarrhea and age < 1.5:
            panleuk_override = [
                {'name': 'Feline Panleukopenia', 'confidence': 75.0, 'emergency': True},
                {'name': 'Gastritis',            'confidence': 15.0},
                {'name': 'Worm Infestation',     'confidence': 10.0}
            ]
            return panleuk_override

        # ==============================================================
        # EMERGENCY OVERRIDE #3 — MALE CAT URINARY BLOCKAGE RISK
        # ==============================================================
        gender   = user_data.get('Gender', '').lower()
        is_male  = gender in ['male', 'm']

        if is_male and has_urinary_issues:
            urinary_override = [
                {'name': 'Urinary Tract Infection', 'confidence': 70.0, 'emergency': True},
                {'name': 'Gastritis',               'confidence': 20.0},
                {'name': 'Dehydration',             'confidence': 10.0}
            ]
            return urinary_override

        # ==============================================================
        # STANDARD cat adjustments (no emergency override triggered)
        # ==============================================================
        adjusted = []

        for pred in predictions:
            name       = pred['name']
            confidence = pred['confidence']

            if 'panleukopenia' in name.lower():
                if not has_diarrhea:
                    confidence *= 0.02
                elif has_diarrhea and not has_vomiting:
                    confidence *= 0.3
                elif has_diarrhea and has_vomiting:
                    confidence *= 2.0 if age < 1 else 0.5

            elif 'respiratory' in name.lower() or 'URI' in name:
                if has_sneezing or has_nasal_discharge or has_coughing:
                    confidence *= 2.0
                elif has_eye_discharge:
                    confidence *= 1.3
                else:
                    confidence *= 0.2

            elif 'gastritis' in name.lower():
                if (has_sneezing or has_nasal_discharge) and has_vomiting:
                    confidence *= 0.4
                elif has_vomiting and not (has_sneezing or has_nasal_discharge):
                    confidence *= 1.2
                elif not has_vomiting:
                    confidence *= 0.1

            elif 'conjunctivitis' in name.lower():
                confidence *= 1.8 if has_eye_discharge else 0.1

            elif 'urinary' in name.lower() or 'UTI' in name:
                confidence *= 2.0 if has_urinary_issues else 0.2

            elif 'skin' in name.lower() or 'allergy' in name.lower():
                if not has_scratching and not has_skin_lesions:
                    confidence *= 0.15
                else:
                    confidence *= 1.5

            elif 'worm' in name.lower() or 'parasite' in name.lower():
                if is_acute:
                    confidence *= 0.2

            elif 'kidney' in name.lower() and 'chronic' in name.lower():
                if age < 7:
                    confidence *= 0.3
                elif age >= 10:
                    confidence *= 1.8

            adjusted.append({'name': name, 'confidence': confidence})

        predictions = adjusted

    # ----------------------------------------------------------
    # COW clinical adjustments
    # ----------------------------------------------------------
    elif animal_type == 'cow':
        gender   = str(user_data.get('Gender', '')).lower()
        is_male  = gender in ['male', 'm', 'bull', 'steer']
        is_female = gender in ['female', 'f', 'cow', 'heifer']

        has_weakness = any(kw in all_symptoms for kw in [
            'weak', 'down', 'cant stand', 'unable to stand', 'unable', 'weakness', 'recumbent'
        ])
        has_bloat = any(kw in all_symptoms for kw in [
            'bloat', 'distended', 'swollen abdomen', 'bloating', 'gas', 'distension', 'tympany'
        ])
        has_udder_issues = any(kw in all_symptoms for kw in ['udder', 'mastitis', 'milk', 'swollen udder'])

        duration       = float(user_data.get('Duration', 999))
        is_post_calving = duration <= 7
        is_acute        = duration <= 3

        # ---------- Bloat emergency override ----------
        if has_bloat:
            bloat_override = [
                {'name': 'Bloat', 'confidence': 85.0, 'override': True}
            ]
            for pred in predictions:
                if pred['name'] != 'Bloat':
                    bloat_override.append({'name': pred['name'], 'confidence': pred['confidence'] * 0.15})
            total = sum(p['confidence'] for p in bloat_override)
            bloat_override = [{'name': p['name'], 'confidence': (p['confidence'] / total) * 100} for p in bloat_override]
            bloat_override.sort(key=lambda x: x['confidence'], reverse=True)
            return bloat_override[:3]

        # ---------- Milk Fever override (dairy female, post-calving, weak) ----------
        breed    = user_data.get('Breed', '').lower()
        is_dairy = any(b in breed for b in ['holstein', 'jersey', 'guernsey', 'brown swiss'])

        if is_female and is_dairy and is_post_calving and has_weakness:
            mf_override = [
                {'name': 'Milk Fever', 'confidence': 80.0, 'override': True}
            ]
            for pred in predictions:
                if pred['name'] != 'Milk Fever':
                    mf_override.append({'name': pred['name'], 'confidence': pred['confidence'] * 0.2})
            total = sum(p['confidence'] for p in mf_override)
            mf_override = [{'name': p['name'], 'confidence': (p['confidence'] / total) * 100} for p in mf_override]
            mf_override.sort(key=lambda x: x['confidence'], reverse=True)
            return mf_override[:3]

        # ---------- Standard cow adjustments ----------
        adjusted = []

        for pred in predictions:
            name       = pred['name']
            confidence = pred['confidence']

            if is_male and name in ['Mastitis', 'Milk Fever', 'Displaced Abomasum']:
                confidence *= 0.01

            if ('coccidiosis' in name.lower() or 'coccidia' in name.lower()) and age > 1:
                confidence *= 0.01

            if name == "Parasitic Infection":
                if is_acute:
                    confidence *= 0.05
                elif not has_diarrhea:
                    confidence *= 0.3

            if name == "Foot and Mouth Disease":
                if not has_skin_lesions and not has_lameness:
                    confidence *= 0.05
                else:
                    confidence *= 1.5

            if name == "Respiratory Disease" or 'respiratory' in name.lower():
                if not has_coughing and not has_labored_breathing:
                    confidence *= 0.01
                elif has_coughing or (has_labored_breathing and has_nasal_discharge):
                    confidence *= 1.0
                elif has_labored_breathing and not has_nasal_discharge:
                    confidence *= 0.2

            if name == "Milk Fever":
                if is_male:
                    confidence *= 0.01
                elif is_female:
                    if is_dairy and is_post_calving and has_weakness:
                        confidence *= 3.0
                    elif is_dairy and has_weakness:
                        confidence *= 1.5
                    elif not is_dairy:
                        confidence *= 0.3
                    elif not has_weakness:
                        confidence *= 0.2

            if name == "Displaced Abomasum":
                if is_male:
                    confidence *= 0.01
                elif is_female:
                    if is_dairy and is_post_calving and has_appetite_loss:
                        confidence *= 2.0
                    elif not is_dairy:
                        confidence *= 0.4

            if name == "Bloat":
                if has_bloat:
                    confidence *= 5.0
                elif has_labored_breathing:
                    confidence *= 1.2
                else:
                    confidence *= 0.1

            if name == "Mastitis":
                if is_male:
                    confidence *= 0.01
                elif is_female:
                    if is_dairy and has_udder_issues:
                        confidence *= 2.5
                    elif is_dairy:
                        confidence *= 1.3
                    elif not is_dairy:
                        confidence *= 0.5

            if name == "Bovine Viral Diarrhea" or 'BVD' in name:
                if not has_diarrhea and not has_coughing:
                    confidence *= 0.1

            adjusted.append({'name': name, 'confidence': confidence})

        predictions = adjusted

    # ----------------------------------------------------------
    # Normalise & sort
    # ----------------------------------------------------------
    total = sum(p['confidence'] for p in predictions)
    if total > 0:
        predictions = [{'name': p['name'], 'confidence': (p['confidence'] / total) * 100} for p in predictions]

    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return predictions


def determine_confidence_level(top_confidence, user_data):
    """Determine confidence level based on missing vitals"""
    missing_vitals = []

    temp = str(user_data.get('Body_Temperature', '')).strip()
    hr   = str(user_data.get('Heart_Rate',        '')).strip()

    if not temp or temp in ['.', '', 'none', 'None']:
        missing_vitals.append('Body Temperature')
    if not hr   or hr   in ['.', '', 'none', 'None']:
        missing_vitals.append('Heart Rate')

    if len(missing_vitals) >= 2:
        level = 'MODERATE' if top_confidence >= 70 else 'LOW'
    elif len(missing_vitals) == 1:
        if   top_confidence >= 85: level = 'HIGH'
        elif top_confidence >= 60: level = 'MODERATE'
        else:                      level = 'LOW'
    else:
        if   top_confidence >= 70: level = 'HIGH'
        elif top_confidence >= 50: level = 'MODERATE'
        else:                      level = 'LOW'

    return level, missing_vitals


def get_breed_specific_warnings(breed, symptoms, animal_type='dog'):
    """Get breed-specific differential diagnoses"""
    breed_lower = breed.lower()
    warnings    = []
    gender      = str(symptoms.get('Gender', '')).lower()

    if   animal_type == 'dog': breed_db = BREED_RISKS
    elif animal_type == 'cow': breed_db = COW_BREED_RISKS
    else:                      breed_db = CAT_BREED_RISKS

    for breed_key, risks in breed_db.items():
        if breed_key in breed_lower:
            # Filter out female-only conditions for male cows
            if animal_type == 'cow' and gender in ['male', 'm', 'bull', 'steer']:
                risks = [r for r in risks if r not in ['Mastitis', 'Milk fever', 'Dystocia', 'Displaced abomasum']]
                if not risks:
                    warnings.append({
                        'type': 'info',
                        'message': f'🐄 Breed Info: {breed.title()} bull/steer - lower disease risk than dairy females'
                    })
                    break

            emoji = '🐕' if animal_type == 'dog' else ('🐄' if animal_type == 'cow' else '🐱')
            warnings.append({
                'type': 'info',
                'message': f'{emoji} Breed Alert: {breed.title()} {"cattle are" if animal_type == "cow" else ("cats are" if animal_type == "cat" else "s are")} prone to: {", ".join(risks)}'
            })

            # Extra dog breed warnings
            if animal_type == 'dog':
                symp_str = str(symptoms).lower()
                if 'vomiting' in symp_str and 'Foreign body ingestion' in risks:
                    warnings.append({
                        'type': 'warning',
                        'message': f'⚠️ Important: {breed.title()}s commonly ingest foreign objects. Monitor for repeated vomiting or abdominal pain.'
                    })
                if 'vomiting' in symp_str and 'Pancreatitis' in risks:
                    warnings.append({
                        'type': 'warning',
                        'message': '⚠️ Consider: Pancreatitis is common in this breed. Watch for abdominal pain and lethargy.'
                    })
            break

    return warnings


def generate_warnings(predictions, user_data, animal_type):
    """Generate contextual warnings"""
    warnings = []

    try:
        age = float(user_data.get('Age', 0))
    except Exception:
        age = 0

    top_disease = predictions[0]['name'] if predictions else ''

    all_symptoms = ' '.join([
        str(user_data.get('Symptom_1', '')),
        str(user_data.get('Symptom_2', '')),
        str(user_data.get('Symptom_3', '')),
        str(user_data.get('Symptom_4', ''))
    ]).lower()

    has_diarrhea  = user_data.get('Diarrhea',  '').lower() in ['yes', 'y', '1']
    has_vomiting  = user_data.get('Vomiting',  '').lower() in ['yes', 'y', '1']
    has_coughing  = user_data.get('Coughing',  '').lower() in ['yes', 'y', '1']
    has_labored_breathing = user_data.get('Labored_Breathing', '').lower() in ['yes', 'y', '1']
    has_skin_lesions      = user_data.get('Skin_Lesions',      '').lower() in ['yes', 'y', '1']

    # ----------------------------------------------------------
    # DOG warnings
    # ----------------------------------------------------------
    if animal_type == 'dog':
        has_bloody_diarrhea = any(kw in all_symptoms for kw in ['bloody', 'blood', 'hemorrhagic'])

        if age <= 1.5 and has_bloody_diarrhea and has_diarrhea:
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 LIFE-THREATENING EMERGENCY: Bloody diarrhea in a puppy is a PARVOVIRUS EMERGENCY. GO TO EMERGENCY VET IMMEDIATELY. Parvo is fatal without treatment.'
            })
        elif age <= 1.5 and has_diarrhea:
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 URGENT: Young dogs with diarrhea are at HIGH RISK for Parvovirus - seek emergency care if bloody diarrhea or lethargy develops'
            })

        breed = user_data.get('Breed', '')
        warnings.extend(get_breed_specific_warnings(breed, user_data, 'dog'))

        parvo_in_results = any('Parvovirus' in p['name'] for p in predictions)
        if age > 1.5 and parvo_in_results and not has_diarrhea:
            warnings.append({
                'type': 'info',
                'message': '💉 Note: Parvovirus risk is LOW in adult dogs without diarrhea. Ensure vaccination is up-to-date.'
            })

        if top_disease in DOG_DISEASE_INFO:
            vaccine_info = DOG_DISEASE_INFO[top_disease].get('vaccine', 'None')
            if vaccine_info and 'None' not in vaccine_info:
                warnings.append({'type': 'info', 'message': f'💉 Vaccine Available: {vaccine_info}'})

    # ----------------------------------------------------------
    # CAT warnings
    # ----------------------------------------------------------
    elif animal_type == 'cat':
        breed = user_data.get('Breed', '')
        warnings.extend(get_breed_specific_warnings(breed, user_data, 'cat'))

        # Labored breathing emergency
        if has_labored_breathing:
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 BREATHING EMERGENCY: Labored breathing in a cat is a LIFE-THREATENING EMERGENCY. Transport to emergency vet IMMEDIATELY. Oxygen therapy may be critical.'
            })

        # Panleukopenia emergency
        if top_disease == "Feline Panleukopenia" or (age < 1.5 and has_vomiting and has_diarrhea):
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 LIFE-THREATENING EMERGENCY: Feline Panleukopenia (Feline Distemper) is HIGHLY FATAL in kittens. IMMEDIATE emergency care required. This is extremely contagious.'
            })

        if top_disease == "Dehydration":
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 URGENT: Dehydration can be LIFE-THREATENING in cats. Immediate veterinary care for fluid therapy required.'
            })

        if top_disease == "Anemia":
            warnings.insert(0, {
                'type': 'critical',
                'message': '⚠️ SERIOUS: Anemia requires immediate investigation. Can be caused by parasites, FeLV, FIV, or blood loss. Urgent vet exam needed.'
            })

        # Male urinary blockage
        gender  = user_data.get('Gender', '').lower()
        is_male = gender in ['male', 'm']
        has_urinary_issues = any(kw in all_symptoms for kw in ['urinat', 'strain', 'litter box'])

        if is_male and (has_urinary_issues or 'urinary' in top_disease.lower()):
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 MALE CAT ALERT: Male cats can develop LIFE-THREATENING urinary blockages. If unable to urinate or straining with no urine, this is an EMERGENCY.'
            })

        if top_disease in CAT_DISEASE_INFO:
            vaccine_info = CAT_DISEASE_INFO[top_disease].get('vaccine', 'None')
            if vaccine_info and 'None' not in vaccine_info:
                warnings.append({'type': 'info', 'message': f'💉 Vaccine Available: {vaccine_info}'})

    # ----------------------------------------------------------
    # COW warnings
    # ----------------------------------------------------------
    elif animal_type == 'cow':
        breed = user_data.get('Breed', '')
        has_bloat_keyword = any(kw in all_symptoms for kw in [
            'bloat', 'distended', 'swollen abdomen', 'bloating', 'gas', 'distension', 'tympany'
        ])

        if has_bloat_keyword or top_disease == "Bloat":
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 BLOAT EMERGENCY: Abdominal bloating in cattle is LIFE-THREATENING. Animal can die within HOURS from respiratory compromise or shock. Call veterinarian NOW - DO NOT WAIT.'
            })

        warnings.extend(get_breed_specific_warnings(breed, user_data, 'cow'))

        if top_disease == "Foot and Mouth Disease":
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 REPORTABLE DISEASE: Foot and Mouth Disease is HIGHLY CONTAGIOUS and must be reported to authorities immediately. ISOLATE animal now.'
            })

        if top_disease == "Milk Fever" and not has_bloat_keyword:
            warnings.insert(0, {
                'type': 'critical',
                'message': '🚨 EMERGENCY: Milk fever can be FATAL within hours. Immediate IV calcium administration required. Call vet NOW.'
            })

        if top_disease in COW_DISEASE_INFO:
            vaccine_info = COW_DISEASE_INFO[top_disease].get('vaccine', 'None')
            if vaccine_info and 'None' not in vaccine_info:
                warnings.append({'type': 'info', 'message': f'💉 Vaccine Available: {vaccine_info}'})

    return warnings


def get_recommendations(predictions, animal_type, user_data, missing_vitals):
    """Generate clinical recommendations"""
    top_disease = predictions[0]['name'] if predictions else ''

    if   animal_type == 'dog': disease_info = DOG_DISEASE_INFO
    elif animal_type == 'cow': disease_info = COW_DISEASE_INFO
    else:                      disease_info = CAT_DISEASE_INFO

    recommendations = []

    if top_disease in disease_info:
        info = disease_info[top_disease]
        recommendations.append(f"🎯 Recommended Action: {info['action']}")
        recommendations.append(f"📋 Severity: {info['severity']}")
        recommendations.append("")

    # ----------------------------------------------------------
    # CAT recommendations
    # ----------------------------------------------------------
    if animal_type == 'cat':
        has_vomiting        = user_data.get('Vomiting',        '').lower() in ['yes', 'y', '1']
        has_nasal_discharge = user_data.get('Nasal_Discharge', '').lower() in ['yes', 'y', '1']
        has_coughing        = user_data.get('Coughing',        '').lower() in ['yes', 'y', '1']

        all_symptoms = ' '.join([
            str(user_data.get('Symptom_1', '')),
            str(user_data.get('Symptom_2', '')),
            str(user_data.get('Symptom_3', '')),
            str(user_data.get('Symptom_4', ''))
        ]).lower()

        has_sneezing = 'sneez' in all_symptoms or has_coughing

        # --- Labored-breathing / respiratory emergency ---
        if top_disease in ["Respiratory Distress", "Feline Asthma", "Lower Respiratory Infection", "Severe Allergic Reaction"]:
            recommendations.append("🚨 RESPIRATORY EMERGENCY PROTOCOL:")
            recommendations.append("   • Transport to emergency veterinarian IMMEDIATELY")
            recommendations.append("   • Keep cat calm and in a well-ventilated area during transport")
            recommendations.append("   • Do NOT muzzle or restrict the cat's airway")
            recommendations.append("   • Oxygen therapy will likely be required on arrival")
            if top_disease == "Feline Asthma":
                recommendations.append("   • If cat has a known inhaler prescription, administer now")
                recommendations.append("   • Long-term: bronchodilators + corticosteroids managed by vet")
            if top_disease == "Severe Allergic Reaction":
                recommendations.append("   • Antihistamines / corticosteroids will be administered by vet")
            recommendations.append("")
            recommendations.append("⚠️ DO NOT DELAY — breathing emergencies are time-critical")

        elif 'respiratory' in top_disease.lower() or 'URI' in top_disease:
            recommendations.append("📋 Feline Upper Respiratory Complex (Viral URI):")
            recommendations.append("   • Likely caused by Feline Herpesvirus (FHV-1) or Calicivirus (FCV)")
            recommendations.append("   • Keep cat INDOORS and ISOLATED from other cats (highly contagious)")
            recommendations.append("   • Gently wipe nasal/eye discharge with warm damp cloth")
            recommendations.append("   • Use humidifier or steam (bathroom) to help breathing")
            recommendations.append("   • Encourage eating - warm wet food to stimulate appetite")
            recommendations.append("   • Ensure adequate hydration")
            recommendations.append("")
            recommendations.append("🚨 Seek IMMEDIATE Vet Care If:")
            recommendations.append("   • Cat stops eating for >24 hours")
            recommendations.append("   • Breathing becomes labored or open-mouth breathing")
            recommendations.append("   • Nasal discharge becomes thick, green, or bloody")
            recommendations.append("   • Eye discharge becomes thick or eyes appear painful")
            recommendations.append("   • Cat becomes severely lethargic or unresponsive")
            recommendations.append("   • High fever persists >2-3 days")

        elif 'panleukopenia' in top_disease.lower():
            recommendations.append("🚨 FELINE PANLEUKOPENIA (Feline Distemper):")
            recommendations.append("   • THIS IS A LIFE-THREATENING EMERGENCY")
            recommendations.append("   • Go to emergency veterinarian IMMEDIATELY")
            recommendations.append("   • HIGHLY contagious to other cats - STRICT ISOLATION")
            recommendations.append("   • Requires hospitalization with IV fluids and intensive care")
            recommendations.append("   • High mortality rate without aggressive treatment")
            recommendations.append("   • Bring vaccination records if available")
            recommendations.append("")
            recommendations.append("⚠️ DO NOT DELAY - This is a medical emergency")

        elif 'gastritis' in top_disease.lower():
            if has_sneezing or has_nasal_discharge:
                recommendations.append("📋 Gastritis (Possibly Secondary to URI):")
                recommendations.append("   • Vomiting may be caused by post-nasal drip from URI")
                recommendations.append("   • Focus on treating respiratory symptoms")
                recommendations.append("   • Small frequent meals of bland food")
                recommendations.append("   • Ensure adequate hydration")
            else:
                recommendations.append("📋 Gastritis Management:")
                recommendations.append("   • Withhold food for 12 hours (water only)")
                recommendations.append("   • Then small frequent meals of bland food (boiled chicken/rice)")
                recommendations.append("   • Gradually reintroduce regular diet over 3-4 days")
            recommendations.append("")
            recommendations.append("🚨 Seek Vet Care If:")
            recommendations.append("   • Vomiting persists >24 hours")
            recommendations.append("   • Blood in vomit")
            recommendations.append("   • Severe lethargy or weakness")
            recommendations.append("   • Unable to keep water down")
            recommendations.append("   • Abdominal pain (hunched posture, crying)")

        elif 'urinary' in top_disease.lower():
            gender  = user_data.get('Gender', '').lower()
            is_male = gender in ['male', 'm']

            if is_male:
                recommendations.append("🚨 MALE CAT URINARY ALERT:")
                recommendations.append("   • Male cats can develop LIFE-THREATENING urinary blockages")
                recommendations.append("   • If unable to urinate or only drops of urine, this is an EMERGENCY")
                recommendations.append("   • Complete blockage can cause death within 24-48 hours")
                recommendations.append("")

            recommendations.append("📋 Urinary Tract Infection Management:")
            recommendations.append("   • Veterinary examination and urinalysis required")
            recommendations.append("   • Antibiotics will be prescribed if bacterial")
            recommendations.append("   • Increase water intake (wet food, water fountains)")
            recommendations.append("   • Keep litter box very clean")
            recommendations.append("   • Monitor for signs of blockage")
            recommendations.append("")
            recommendations.append("🚨 EMERGENCY - Seek Immediate Care If:")
            recommendations.append("   • Straining with NO urine produced (especially males)")
            recommendations.append("   • Crying/vocalizing while trying to urinate")
            recommendations.append("   • Lethargy, vomiting, loss of appetite")
            recommendations.append("   • Swollen/painful abdomen")

        elif 'dehydration' in top_disease.lower():
            recommendations.append("🚨 DEHYDRATION - URGENT CARE NEEDED:")
            recommendations.append("   • Dehydration can be LIFE-THREATENING in cats")
            recommendations.append("   • Immediate veterinary care for fluid therapy required")
            recommendations.append("   • Subcutaneous or IV fluids may be necessary")
            recommendations.append("   • Identify and treat underlying cause")

        elif 'anemia' in top_disease.lower():
            recommendations.append("🚨 ANEMIA - SERIOUS CONDITION:")
            recommendations.append("   • Urgent veterinary examination required")
            recommendations.append("   • Blood work needed to determine cause")
            recommendations.append("   • Possible causes: Parasites, FeLV, FIV, blood loss, toxins")
            recommendations.append("   • May require blood transfusion if severe")

        elif 'skin' in top_disease.lower() or 'allergy' in top_disease.lower():
            recommendations.append("📋 Skin Allergy Management:")
            recommendations.append("   • Veterinary exam to identify allergen type")
            recommendations.append("   • Common causes: Fleas, food, environmental")
            recommendations.append("   • Strict flea control for all pets in household")
            recommendations.append("   • Consider hypoallergenic diet trial")
            recommendations.append("   • Antihistamines or corticosteroids as prescribed")

        if missing_vitals:
            recommendations.append("")
            recommendations.append(f"⚠️ Missing Information: {', '.join(missing_vitals)}")
            recommendations.append("   • Consider checking these for better diagnosis accuracy")
            recommendations.append("   • Normal cat temperature: 38-39.2°C (100.5-102.5°F)")
            recommendations.append("   • Normal cat heart rate: 140-220 bpm")

        if top_disease not in ["Feline Panleukopenia", "Dehydration", "Anemia",
                                "Respiratory Distress", "Feline Asthma",
                                "Lower Respiratory Infection", "Severe Allergic Reaction"]:
            recommendations.append("")
            recommendations.append("🏥 General Care:")
            recommendations.append("   • Monitor food and water intake closely")
            recommendations.append("   • Keep cat indoors during illness")
            recommendations.append("   • Isolate from other cats if contagious disease suspected")
            recommendations.append("   • Document all symptoms and changes")
            recommendations.append("   • Verify FVRCP vaccination status is current")

    # ----------------------------------------------------------
    # DOG recommendations
    # ----------------------------------------------------------
    elif animal_type == 'dog':
        if top_disease == "Parvovirus":
            recommendations.append("")
            recommendations.append("🚨 PARVOVIRUS IS A MEDICAL EMERGENCY:")
            recommendations.append("   • Do NOT wait - go to emergency vet NOW")
            recommendations.append("   • Bring vaccination records if available")
            recommendations.append("   • Keep puppy isolated from other dogs")
            recommendations.append("   • This is FATAL without treatment")
            recommendations.append("   • Treatment requires hospitalization with IV fluids")

            if missing_vitals:
                recommendations.append("")
                recommendations.append(f"⚠️ Note: Missing vitals ({', '.join(missing_vitals)}) but clinical presentation strongly suggests Parvo")

        elif top_disease == "Gastroenteritis":
            info = DOG_DISEASE_INFO.get(top_disease, {})
            recommendations.append("")
            recommendations.append("🔍 Common Causes:")
            common_causes = info.get('common_causes', '').split(', ')
            for cause in common_causes:
                if cause:
                    recommendations.append(f"   • {cause.title()}")

            breed = user_data.get('Breed', '').lower()
            if any(b in breed for b in ['labrador', 'golden', 'beagle']):
                recommendations.append("")
                recommendations.append("⚠️ High Priority for This Breed:")
                recommendations.append("   • Rule out foreign body ingestion")
                recommendations.append("   • Consider pancreatitis if symptoms persist")

            recommendations.append("")
            recommendations.append("🚨 Seek Immediate Vet Care If:")
            recommendations.append("   • Vomiting persists >24-48 hours")
            recommendations.append("   • Lethargy or weakness develops")
            recommendations.append("   • Abdominal pain (hunched posture, whining)")
            recommendations.append("   • Black or bloody stool")
            recommendations.append("   • Inability to keep water down")

        if top_disease != "Parvovirus":
            recommendations.append("")
            recommendations.append("🏥 Veterinary Consultation:")
            recommendations.append("   • Professional examination recommended")
            recommendations.append("   • Monitor symptoms and document changes")
            recommendations.append("   • Ensure access to fresh water")
            recommendations.append("   • Verify vaccination status is current")

    # ----------------------------------------------------------
    # COW recommendations
    # ----------------------------------------------------------
    elif animal_type == 'cow':
        if top_disease == "Milk Fever":
            recommendations.append("")
            recommendations.append("🚨 IMMEDIATE ACTIONS:")
            recommendations.append("   • Call veterinarian NOW - this is time-critical")
            recommendations.append("   • Do not attempt to force animal to stand")
            recommendations.append("   • Keep animal warm and comfortable")
            recommendations.append("   • IV calcium must be given by veterinarian")
            recommendations.append("   • Response to treatment is usually dramatic if caught early")

        elif top_disease == "Bloat":
            recommendations.append("")
            recommendations.append("🚨 BLOAT EMERGENCY ACTIONS:")
            recommendations.append("   • Call veterinarian IMMEDIATELY - this is TIME-CRITICAL")
            recommendations.append("   • Do NOT feed or give water")
            recommendations.append("   • Keep animal STANDING if possible (prevents rumen pressure on lungs)")
            recommendations.append("   • Walking may help release gas if mild")
            recommendations.append("   • Veterinarian may pass stomach tube to relieve gas")
            recommendations.append("   • Emergency trocarization if life-threatening")
            recommendations.append("   • ⚠️ Can die from respiratory compromise or shock within hours")
            recommendations.append("   • ⚠️ Prevention: Avoid sudden diet changes, limit legume pasture")

        elif top_disease == "Foot and Mouth Disease":
            recommendations.append("")
            recommendations.append("🚨 REGULATORY REQUIREMENTS:")
            recommendations.append("   • IMMEDIATELY notify state/federal veterinarian")
            recommendations.append("   • ISOLATE all cattle - ZERO movement on/off farm")
            recommendations.append("   • This is a REPORTABLE disease - legal requirement")
            recommendations.append("   • Follow all quarantine procedures")
            recommendations.append("   • Do not allow ANY visitors to farm")
            recommendations.append("   • ⚠️ FOOD SAFETY: No animals or products may leave farm")
            recommendations.append("   • Massive economic impact - authorities will provide guidance")

        if missing_vitals and top_disease in ["Milk Fever", "Bloat", "Foot and Mouth Disease"]:
            recommendations.append("")
            recommendations.append(f"⚠️ Note: Missing vitals ({', '.join(missing_vitals)}) but symptoms require immediate action")

        if top_disease not in ["Milk Fever", "Bloat", "Foot and Mouth Disease"]:
            recommendations.append("")
            recommendations.append("🏥 Veterinary Consultation:")
            recommendations.append("   • Professional examination recommended")
            recommendations.append("   • Monitor herd for similar signs")
            recommendations.append("   • Ensure access to fresh water and quality forage")
            recommendations.append("   • Review herd health and vaccination protocols")

    # Catch-all for missing vitals
    if missing_vitals and not any('Missing' in str(r) for r in recommendations):
        recommendations.append("")
        recommendations.append(f"⚠️ Missing Information: {', '.join(missing_vitals)}")
        recommendations.append("   Consider checking these for better diagnosis accuracy")

    return recommendations


def get_top_predictions(model, target_encoder, input_df, top_k=3):
    """Get top K disease predictions from a structured model"""
    probs         = model.predict_proba(input_df)[0]
    top_indices   = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        disease    = target_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx] * 100)
        results.append({'name': disease, 'confidence': round(confidence, 2)})

    return results


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    cat_status = 'structured' if cat_structured_model else ('text-based' if cat_text_based else 'not loaded')
    return jsonify({
        'status': 'online',
        'message': 'VetBot API with Clinical Logic',
        'dog_model': 'loaded' if dog_model else 'not loaded',
        'cat_model': cat_status,
        'cow_model': 'loaded' if cow_model else 'not loaded'
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data        = request.json
        animal_type = data.get('animal_type', '').lower()
        user_data   = data.get('data', {})

        if not animal_type or not user_data:
            return jsonify({'error': 'Missing animal_type or data'}), 400

        # ==============================================================
        # CAT PREDICTION
        # ==============================================================
        if animal_type == 'cat':

            # ---- PATH A: Structured model (preferred, same as dog/cow) ----
            if cat_structured_model and cat_structured_target and cat_structured_features:
                print("🐱 Using STRUCTURED cat model")

                input_df = preprocess_data(user_data, 'cat')

                for feature in cat_structured_features:
                    if feature not in input_df.columns:
                        input_df[feature] = 0

                input_df     = input_df[cat_structured_features]
                predictions  = get_top_predictions(cat_structured_model, cat_structured_target, input_df, top_k=5)

            # ---- PATH B: Text-based fallback ----
            elif cat_model and cat_tfidf and cat_mlb:
                print("🐱 Using TEXT-BASED cat model (fallback)")

                symptoms_list = []
                for i in range(1, 5):
                    symptom = user_data.get(f'Symptom_{i}', '').strip()
                    if symptom and symptom.lower() not in ['none', 'unknown', '']:
                        symptoms_list.append(symptom)

                binary_symptom_map = {
                    'Appetite_Loss':     'loss of appetite',
                    'Vomiting':          'vomiting',
                    'Diarrhea':          'diarrhea',
                    'Coughing':          'coughing',
                    'Labored_Breathing': 'labored breathing',
                    'Lameness':          'lameness',
                    'Skin_Lesions':      'skin lesions',
                    'Nasal_Discharge':   'nasal discharge',
                    'Eye_Discharge':     'eye discharge'
                }

                for key, symptom_text in binary_symptom_map.items():
                    if user_data.get(key, '').lower() in ['yes', 'y', '1']:
                        symptoms_list.append(symptom_text)

                symptom_string = ', '.join(symptoms_list)
                if not symptom_string:
                    return jsonify({'error': 'No symptoms provided'}), 400

                print(f"🐱 Cat symptoms (text): {symptom_string}")

                X_input        = cat_tfidf.transform([symptom_string])
                y_pred_proba   = cat_model.predict_proba(X_input)[0]
                top_indices    = np.argsort(y_pred_proba)[::-1][:5]

                predictions = []
                for idx in top_indices:
                    disease    = cat_mlb.classes_[idx]
                    confidence = float(y_pred_proba[idx] * 100)
                    if confidence > 1.0:
                        predictions.append({'name': disease, 'confidence': round(confidence, 2)})

            else:
                return jsonify({'error': 'No cat model loaded (neither structured nor text-based)'}), 500

            # ---- Shared cat post-processing ----
            predictions       = apply_clinical_logic(predictions, user_data, 'cat')
            predictions       = predictions[:3]

            top_confidence    = predictions[0]['confidence'] if predictions else 0
            confidence_level, missing_vitals = determine_confidence_level(top_confidence, user_data)

            warnings         = generate_warnings(predictions, user_data, 'cat')
            recommendations  = get_recommendations(predictions, 'cat', user_data, missing_vitals)

            return jsonify({
                'success': True,
                'animal_type': 'cat',
                'top3': predictions,
                'confidence_level': confidence_level,
                'warnings': warnings,
                'recommendations': recommendations
            })

        # ==============================================================
        # DOG PREDICTION
        # ==============================================================
        elif animal_type == 'dog':
            if not dog_model:
                return jsonify({'error': 'Dog model not loaded'}), 500
            model          = dog_model
            target_encoder = dog_target_encoder
            features       = dog_features

        # ==============================================================
        # COW PREDICTION
        # ==============================================================
        elif animal_type == 'cow':
            if not cow_model:
                return jsonify({'error': 'Cow model not loaded'}), 500
            model          = cow_model
            target_encoder = cow_target_encoder
            features       = cow_features

        else:
            return jsonify({'error': 'Invalid animal_type. Must be dog, cat, or cow'}), 400

        # ==============================================================
        # DOG / COW shared structured prediction path
        # ==============================================================
        input_df = preprocess_data(user_data, animal_type)

        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[features]

        predictions      = get_top_predictions(model, target_encoder, input_df, top_k=5)
        predictions      = apply_clinical_logic(predictions, user_data, animal_type)
        predictions      = predictions[:3]

        top_confidence   = predictions[0]['confidence']
        confidence_level, missing_vitals = determine_confidence_level(top_confidence, user_data)

        warnings         = generate_warnings(predictions, user_data, animal_type)
        recommendations  = get_recommendations(predictions, animal_type, user_data, missing_vitals)

        return jsonify({
            'success': True,
            'animal_type': animal_type,
            'top3': predictions,
            'confidence_level': confidence_level,
            'warnings': warnings,
            'recommendations': recommendations
        })

    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/diseases/<animal_type>', methods=['GET'])
def get_diseases(animal_type):
    if   animal_type == 'dog': db = DOG_DISEASE_INFO
    elif animal_type == 'cow': db = COW_DISEASE_INFO
    elif animal_type == 'cat': db = CAT_DISEASE_INFO
    else: return jsonify({'error': 'Invalid animal_type'}), 400

    return jsonify({'animal_type': animal_type, 'diseases': list(db.keys())})


@app.route('/disease-info/<animal_type>/<disease_name>', methods=['GET'])
def get_disease_info(animal_type, disease_name):
    if   animal_type == 'dog': db = DOG_DISEASE_INFO
    elif animal_type == 'cow': db = COW_DISEASE_INFO
    elif animal_type == 'cat': db = CAT_DISEASE_INFO
    else: return jsonify({'error': 'Invalid animal_type'}), 400

    if disease_name in db:
        return jsonify({'animal_type': animal_type, 'disease': disease_name, 'info': db[disease_name]})
    return jsonify({'error': 'Disease not found'}), 404


@app.route('/find-vets', methods=['POST'])
def find_vets():
    try:
        data       = request.json
        latitude   = data.get('latitude')
        longitude  = data.get('longitude')
        animal_type = data.get('animal_type', 'dog').lower()
        emergency  = data.get('emergency', False)

        if not latitude or not longitude:
            return jsonify({'error': 'Location not provided'}), 400

        if emergency:
            search_query = "emergency veterinary clinic" if animal_type in ['dog', 'cat'] else "emergency large animal veterinarian"
        else:
            search_query = "veterinary clinic" if animal_type in ['dog', 'cat'] else "large animal veterinarian"

        maps_url = f"https://www.google.com/maps/search/{search_query}/@{latitude},{longitude},13z"

        return jsonify({
            'success': True,
            'location': {'latitude': latitude, 'longitude': longitude},
            'search_query': search_query,
            'maps_url': maps_url,
            'emergency': emergency,
            'animal_type': animal_type,
            'message': f'Opening Google Maps to find {search_query} near you...'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    cat_status = '✅ Structured' if cat_structured_model else ('✅ Text-based (fallback)' if cat_text_based else '❌ Not Loaded')

    print("\n" + "=" * 60)
    print("🏥 VetBot API with Clinical Logic")
    print("=" * 60)
    print(f"Dog Model : {'✅ Loaded' if dog_model else '❌ Not Loaded'}")
    print(f"Cat Model : {cat_status}")
    print(f"Cow Model : {'✅ Loaded' if cow_model else '❌ Not Loaded'}")
    print("=" * 60)
    print("\n🌐 Server running on http://localhost:5000")
    print("\n✨ Features:")
    print("   • Clinical logic filtering for dogs, cats & cows")
    print("   • Breed-specific risk alerts (10 dog, 10 cat, 10 cow breeds)")
    print("   • Confidence adjusted for missing vitals")
    print("   • Symptom-disease matching & emergency overrides")
    print("   • Cat: 3 emergency overrides (breathing / panleuk / male urinary)")
    print("   • Emergency detection (Parvo, Panleuk, Bloat, Milk Fever)")
    print("   • Age-based & gender-specific risk assessment")
    print("   • Cat dual-path: structured model → text-based fallback")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)