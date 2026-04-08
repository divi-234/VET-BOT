# ============================================================
# Convert Text-based Cat Dataset to Structured Format
# ============================================================

import pandas as pd
import numpy as np
import re

# ============================================================
# 1. LOAD YOUR EXISTING TEXT-BASED DATASET
# ============================================================
print("\n" + "="*60)
print("🐱 CONVERTING CAT DATASET TO STRUCTURED FORMAT")
print("="*60)

df_text = pd.read_csv("augmented_cat_disease_dataset.csv")
print(f"\n✅ Loaded {len(df_text)} records")
print(f"Columns: {list(df_text.columns)}")

# ============================================================
# 2. EXTRACT STRUCTURED FEATURES FROM TEXT
# ============================================================

# Symptom keyword mapping
SYMPTOM_KEYWORDS = {
    'vomiting': ['vomit', 'vomiting', 'throwing up', 'regurgitat'],
    'diarrhea': ['diarrhea', 'diarrhoea', 'loose stool', 'watery stool'],
    'coughing': ['cough', 'coughing', 'sneez', 'sniffles'],
    'labored_breathing': ['labored breathing', 'difficulty breathing', 'dyspnea', 'respiratory distress', 'wheezing'],
    'lethargy': ['lethar', 'tired', 'weak', 'inactive', 'depressed'],
    'appetite_loss': ['loss of appetite', 'not eating', 'anorexia', 'inappetence'],
    'fever': ['fever', 'high temperature', 'pyrexia'],
    'nasal_discharge': ['nasal discharge', 'runny nose', 'nasal'],
    'eye_discharge': ['eye discharge', 'ocular discharge', 'eye'],
    'lameness': ['lame', 'lameness', 'limping', 'limp'],
    'skin_lesions': ['skin lesion', 'rash', 'scratch', 'itch', 'dermatitis', 'hair loss']
}

# Common cat breeds
CAT_BREEDS = ['persian', 'siamese', 'maine coon', 'ragdoll', 'bengal', 
              'british shorthair', 'abyssinian', 'sphynx', 'scottish fold', 
              'russian blue', 'domestic shorthair', 'domestic longhair']

def extract_symptom(text, keywords):
    """Check if any keyword appears in text"""
    text_lower = str(text).lower()
    for keyword in keywords:
        if keyword in text_lower:
            return 'yes'
    return 'no'

def extract_breed(text):
    """Extract breed from text"""
    text_lower = str(text).lower()
    for breed in CAT_BREEDS:
        if breed in text_lower:
            return breed.title()
    return 'Domestic Shorthair'  # Default

def extract_age(text):
    """Extract age from text"""
    text_lower = str(text).lower()
    # Look for patterns like "2 years", "1 year old", "6 months"
    match = re.search(r'(\d+)\s*(year|month|yr|mo)', text_lower)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if 'month' in unit or 'mo' in unit:
            return round(num / 12, 1)  # Convert months to years
        return num
    return 2  # Default adult age

def extract_weight(text, age):
    """Extract or estimate weight"""
    text_lower = str(text).lower()
    match = re.search(r'(\d+\.?\d*)\s*(kg|kilogram)', text_lower)
    if match:
        return float(match.group(1))
    # Estimate based on age
    if age < 0.5:
        return 1.5  # Kitten
    elif age < 1:
        return 2.5
    else:
        return 4.5  # Average adult cat

def extract_gender(text):
    """Extract gender from text"""
    text_lower = str(text).lower()
    if 'male' in text_lower and 'female' not in text_lower:
        return 'Male'
    elif 'female' in text_lower:
        return 'Female'
    return 'Unknown'

def extract_duration(text):
    """Extract symptom duration"""
    text_lower = str(text).lower()
    match = re.search(r'(\d+)\s*(day|week|month)', text_lower)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if 'week' in unit:
            return num * 7
        elif 'month' in unit:
            return num * 30
        return num
    return 3  # Default acute

def extract_temperature(text):
    """Extract body temperature"""
    text_lower = str(text).lower()
    match = re.search(r'(\d+\.?\d*)\s*(°c|celsius|degrees)', text_lower)
    if match:
        return float(match.group(1))
    # Check for fever keyword
    if 'fever' in text_lower:
        return 39.5  # Typical fever temp
    return 38.5  # Normal cat temp

def extract_heart_rate(text):
    """Extract or estimate heart rate"""
    text_lower = str(text).lower()
    match = re.search(r'(\d+)\s*(bpm|beats)', text_lower)
    if match:
        return int(match.group(1))
    return 180  # Normal adult cat

def split_symptoms(symptom_text):
    """Split symptom text into up to 4 symptoms"""
    symptoms = [s.strip() for s in str(symptom_text).split(',')]
    # Pad with 'none' if less than 4
    while len(symptoms) < 4:
        symptoms.append('none')
    return symptoms[:4]  # Take only first 4

# ============================================================
# 3. CREATE STRUCTURED DATAFRAME WITH REALISTIC VARIATION
# ============================================================
print("\n🔧 Extracting structured features...")

structured_data = []

# Add random variation for realistic data
np.random.seed(42)

for idx, row in df_text.iterrows():
    symptoms_text = str(row.get('symptoms', ''))
    
    # Check if original dataset has these columns
    if 'age' in df_text.columns and pd.notna(row.get('age')):
        age = float(row['age'])
    else:
        # Generate realistic age distribution
        age = np.random.choice([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15], 
                               p=[0.05, 0.1, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07, 0.06, 0.06, 0.05, 0.03])
    
    if 'gender' in df_text.columns and pd.notna(row.get('gender')) and str(row.get('gender')).lower() != 'unknown':
        gender = str(row['gender']).title()
    else:
        # Randomly assign gender
        gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
    
    if 'breed' in df_text.columns and pd.notna(row.get('breed')):
        breed = str(row['breed']).title()
    else:
        breed = extract_breed(symptoms_text)
    
    if 'weight' in df_text.columns and pd.notna(row.get('weight')):
        weight = float(row['weight'])
    else:
        # Realistic weight based on age
        if age < 0.5:
            weight = np.random.uniform(0.5, 1.5)
        elif age < 1:
            weight = np.random.uniform(1.5, 3.0)
        elif age < 3:
            weight = np.random.uniform(3.0, 5.5)
        else:
            weight = np.random.uniform(3.5, 7.0)
    
    # Split symptoms into 4 fields
    symptoms_split = split_symptoms(symptoms_text)
    
    # Extract duration with variation
    duration = extract_duration(symptoms_text)
    if duration == 3:  # If default, add variation
        duration = np.random.choice([1, 2, 3, 4, 5, 7, 10, 14], 
                                    p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])
    
    # Extract binary symptoms
    record = {
        'Age': round(age, 1),
        'Weight': round(weight, 1),
        'Breed': breed,
        'Gender': gender,
        'Symptom_1': symptoms_split[0],
        'Symptom_2': symptoms_split[1],
        'Symptom_3': symptoms_split[2],
        'Symptom_4': symptoms_split[3],
        'Duration': duration,
        'Appetite_Loss': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['appetite_loss']),
        'Vomiting': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['vomiting']),
        'Diarrhea': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['diarrhea']),
        'Coughing': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['coughing']),
        'Labored_Breathing': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['labored_breathing']),
        'Lameness': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['lameness']),
        'Skin_Lesions': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['skin_lesions']),
        'Nasal_Discharge': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['nasal_discharge']),
        'Eye_Discharge': extract_symptom(symptoms_text, SYMPTOM_KEYWORDS['eye_discharge'])
    }
    
    # Temperature with variation
    base_temp = extract_temperature(symptoms_text)
    if base_temp == 38.5:  # If normal/default
        # Add natural variation
        record['Body_Temperature'] = round(np.random.normal(38.5, 0.3), 1)
        # Increase temp if fever keyword present
        if 'fever' in symptoms_text.lower():
            record['Body_Temperature'] = round(np.random.uniform(39.2, 40.5), 1)
    else:
        record['Body_Temperature'] = base_temp
    
    # Heart rate with variation based on age
    if age < 1:
        record['Heart_Rate'] = int(np.random.uniform(200, 240))
    elif age < 3:
        record['Heart_Rate'] = int(np.random.uniform(180, 220))
    else:
        record['Heart_Rate'] = int(np.random.uniform(160, 200))
    
    # Increase HR if respiratory/cardiac issues
    if record['Labored_Breathing'] == 'yes' or 'fever' in symptoms_text.lower():
        record['Heart_Rate'] += int(np.random.uniform(20, 40))
    
    # Handle disease column
    diseases = str(row.get('diseases', row.get('Disease', 'Unknown')))
    disease = diseases.split(',')[0].strip() if ',' in diseases else diseases.strip()
    record['Disease'] = disease
    
    structured_data.append(record)

df_structured = pd.DataFrame(structured_data)

# ============================================================
# 4. DATA QUALITY CHECKS
# ============================================================
print(f"\n✅ Converted {len(df_structured)} records")
print(f"\n📊 Data Quality Report:")
print(f"   Unique diseases: {df_structured['Disease'].nunique()}")
print(f"   Age range: {df_structured['Age'].min():.1f} - {df_structured['Age'].max():.1f} years")
print(f"   Weight range: {df_structured['Weight'].min():.1f} - {df_structured['Weight'].max():.1f} kg")
print(f"\n   Binary symptom distribution:")

for col in ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 'Labored_Breathing', 
            'Lameness', 'Skin_Lesions', 'Nasal_Discharge', 'Eye_Discharge']:
    yes_count = (df_structured[col] == 'yes').sum()
    print(f"      {col}: {yes_count} ({yes_count/len(df_structured)*100:.1f}%)")

print(f"\n   Disease distribution:")
print(df_structured['Disease'].value_counts().head(10))

# ============================================================
# 5. SAVE STRUCTURED DATASET
# ============================================================
output_file = "cat_disease_dataset_structured.csv"
df_structured.to_csv(output_file, index=False)

print(f"\n✅ Saved structured dataset to: {output_file}")
print("\n" + "="*60)
print("🎉 CONVERSION COMPLETE!")
print("="*60)
print("\n📋 Next steps:")
print("1. Review the structured dataset")
print("2. Run the training script: cat_structured_training.py")
print("3. Update Flask API to use new cat model")
print("\n" + "="*60)

# ============================================================
# 6. PREVIEW
# ============================================================
print("\n📋 Sample records:")
print(df_structured.head(3).to_string())