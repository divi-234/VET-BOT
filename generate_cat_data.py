# ============================================================
# 🐱 Generate Synthetic Cat Disease Training Data V2
# Creates realistic cat disease data with 12 diseases
# ============================================================

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================
# CAT-SPECIFIC PARAMETERS
# ============================================================

CAT_BREEDS = [
    "Persian", "Siamese", "Maine Coon", "Ragdoll", "British Shorthair",
    "Abyssinian", "Sphynx", "Bengal", "Scottish Fold", "American Shorthair",
    "Russian Blue", "Birman", "Oriental", "Domestic Shorthair", "Mixed"
]

# ============================================================
# 12 CAT DISEASES WITH IMPROVED SYMPTOM CORRELATIONS
# ============================================================

CAT_DISEASES = {
    "Upper Respiratory Infection": {
        "weight": 18,
        "symptoms": {
            "primary": ["sneezing", "nasal discharge", "eye discharge", "fever"],
            "secondary": ["coughing", "lethargy", "appetite loss"],
            "common": {
                "Nasal_Discharge": 0.95, "Eye_Discharge": 0.85, "Appetite_Loss": 0.60,
                "Vomiting": 0.10, "Diarrhea": 0.05, "Coughing": 0.40,
                "Labored_Breathing": 0.25, "Lameness": 0.00, "Skin_Lesions": 0.05
            }
        },
        "age_range": (0.2, 15), "temp_range": (39.5, 40.5), "duration": (3, 14)
    },
    "Gastroenteritis": {
        "weight": 15,
        "symptoms": {
            "primary": ["vomiting", "diarrhea", "abdominal pain", "lethargy"],
            "secondary": ["dehydration", "appetite loss", "fever"],
            "common": {
                "Nasal_Discharge": 0.05, "Eye_Discharge": 0.05, "Appetite_Loss": 0.65,
                "Vomiting": 0.90, "Diarrhea": 0.85, "Coughing": 0.00,
                "Labored_Breathing": 0.05, "Lameness": 0.00, "Skin_Lesions": 0.00
            }
        },
        "age_range": (0.5, 12), "temp_range": (38.5, 39.8), "duration": (1, 7)
    },
    "Panleukopenia": {
        "weight": 7,
        "symptoms": {
            "primary": ["vomiting", "severe diarrhea", "lethargy", "high fever"],
            "secondary": ["dehydration", "weakness", "appetite loss"],
            "common": {
                "Nasal_Discharge": 0.10, "Eye_Discharge": 0.10, "Appetite_Loss": 0.95,
                "Vomiting": 0.95, "Diarrhea": 0.95, "Coughing": 0.05,
                "Labored_Breathing": 0.15, "Lameness": 0.05, "Skin_Lesions": 0.00
            }
        },
        "age_range": (0.1, 1.5), "temp_range": (40.0, 41.5), "duration": (2, 7)
    },
    "Chronic Kidney Disease": {
        "weight": 13,
        "symptoms": {
            "primary": ["increased thirst", "increased urination", "weight loss", "lethargy"],
            "secondary": ["vomiting", "poor appetite", "bad breath"],
            "common": {
                "Nasal_Discharge": 0.05, "Eye_Discharge": 0.05, "Appetite_Loss": 0.75,
                "Vomiting": 0.65, "Diarrhea": 0.20, "Coughing": 0.05,
                "Labored_Breathing": 0.10, "Lameness": 0.05, "Skin_Lesions": 0.00
            }
        },
        "age_range": (7, 20), "temp_range": (37.5, 38.8), "duration": (30, 180)
    },
    "Hyperthyroidism": {
        "weight": 11,
        "symptoms": {
            "primary": ["weight loss", "increased appetite", "hyperactivity", "vomiting"],
            "secondary": ["diarrhea", "poor coat", "increased thirst"],
            "common": {
                "Nasal_Discharge": 0.05, "Eye_Discharge": 0.05, "Appetite_Loss": 0.05,
                "Vomiting": 0.60, "Diarrhea": 0.45, "Coughing": 0.05,
                "Labored_Breathing": 0.20, "Lameness": 0.00, "Skin_Lesions": 0.10
            }
        },
        "age_range": (8, 18), "temp_range": (38.5, 39.5), "duration": (60, 365)
    },
    "Diabetes Mellitus": {
        "weight": 9,
        "symptoms": {
            "primary": ["increased thirst", "increased urination", "weight loss", "increased appetite"],
            "secondary": ["lethargy", "weakness", "poor coat"],
            "common": {
                "Nasal_Discharge": 0.05, "Eye_Discharge": 0.05, "Appetite_Loss": 0.10,
                "Vomiting": 0.20, "Diarrhea": 0.10, "Coughing": 0.05,
                "Labored_Breathing": 0.05, "Lameness": 0.10, "Skin_Lesions": 0.00
            }
        },
        "age_range": (6, 15), "temp_range": (37.8, 38.8), "duration": (30, 365)
    },
    "Feline Lower Urinary Tract Disease": {
        "weight": 12,
        "symptoms": {
            "primary": ["straining to urinate", "blood in urine", "frequent urination", "licking genitals"],
            "secondary": ["crying while urinating", "urinating outside box", "lethargy"],
            "common": {
                "Nasal_Discharge": 0.00, "Eye_Discharge": 0.00, "Appetite_Loss": 0.40,
                "Vomiting": 0.25, "Diarrhea": 0.05, "Coughing": 0.00,
                "Labored_Breathing": 0.05, "Lameness": 0.00, "Skin_Lesions": 0.00
            }
        },
        "age_range": (1, 10), "temp_range": (38.3, 39.2), "duration": (2, 14)
    },
    "Inflammatory Bowel Disease": {
        "weight": 9,
        "symptoms": {
            "primary": ["chronic vomiting", "diarrhea", "weight loss", "poor appetite"],
            "secondary": ["lethargy", "poor coat", "gas"],
            "common": {
                "Nasal_Discharge": 0.05, "Eye_Discharge": 0.05, "Appetite_Loss": 0.75,
                "Vomiting": 0.90, "Diarrhea": 0.85, "Coughing": 0.00,
                "Labored_Breathing": 0.00, "Lameness": 0.00, "Skin_Lesions": 0.00
            }
        },
        "age_range": (2, 12), "temp_range": (37.8, 38.8), "duration": (30, 180)
    },
    "Feline Leukemia Virus": {
        "weight": 5,
        "symptoms": {
            "primary": ["weight loss", "lethargy", "anemia", "infections"],
            "secondary": ["fever", "poor appetite", "enlarged lymph nodes"],
            "common": {
                "Nasal_Discharge": 0.30, "Eye_Discharge": 0.25, "Appetite_Loss": 0.80,
                "Vomiting": 0.35, "Diarrhea": 0.30, "Coughing": 0.20,
                "Labored_Breathing": 0.25, "Lameness": 0.05, "Skin_Lesions": 0.15
            }
        },
        "age_range": (1, 8), "temp_range": (38.5, 40.0), "duration": (60, 365)
    },
    "Dental Disease": {
        "weight": 11,
        "symptoms": {
            "primary": ["bad breath", "difficulty eating", "drooling", "pawing at mouth"],
            "secondary": ["weight loss", "appetite loss", "bleeding gums"],
            "common": {
                "Nasal_Discharge": 0.10, "Eye_Discharge": 0.05, "Appetite_Loss": 0.70,
                "Vomiting": 0.15, "Diarrhea": 0.05, "Coughing": 0.00,
                "Labored_Breathing": 0.00, "Lameness": 0.00, "Skin_Lesions": 0.00
            }
        },
        "age_range": (3, 15), "temp_range": (38.0, 38.8), "duration": (30, 365)
    },
    "Skin Allergies": {
        "weight": 10,
        "symptoms": {
            "primary": ["itching", "scratching", "hair loss", "skin lesions"],
            "secondary": ["excessive grooming", "red skin", "scabs"],
            "common": {
                "Nasal_Discharge": 0.05, "Eye_Discharge": 0.10, "Appetite_Loss": 0.10,
                "Vomiting": 0.05, "Diarrhea": 0.05, "Coughing": 0.00,
                "Labored_Breathing": 0.00, "Lameness": 0.00, "Skin_Lesions": 0.95
            }
        },
        "age_range": (0.5, 12), "temp_range": (38.0, 38.8), "duration": (14, 180)
    },
    "Asthma": {
        "weight": 7,
        "symptoms": {
            "primary": ["coughing", "wheezing", "labored breathing", "lethargy"],
            "secondary": ["rapid breathing", "open-mouth breathing", "exercise intolerance"],
            "common": {
                "Nasal_Discharge": 0.10, "Eye_Discharge": 0.05, "Appetite_Loss": 0.20,
                "Vomiting": 0.10, "Diarrhea": 0.05, "Coughing": 0.95,
                "Labored_Breathing": 0.95, "Lameness": 0.00, "Skin_Lesions": 0.00
            }
        },
        "age_range": (1, 15), "temp_range": (38.0, 38.8), "duration": (3, 365)
    }
}

# ============================================================
# GENERATE DATASET
# ============================================================

def generate_cat_data(num_samples=2000):
    """Generate synthetic cat disease data"""
    
    data = []
    
    # Calculate disease distribution
    total_weight = sum(d["weight"] for d in CAT_DISEASES.values())
    
    for _ in range(num_samples):
        # Select disease based on weights
        rand = random.random() * total_weight
        cumulative = 0
        selected_disease = None
        
        for disease, info in CAT_DISEASES.items():
            cumulative += info["weight"]
            if rand <= cumulative:
                selected_disease = disease
                disease_info = info
                break
        
        # Generate cat profile
        breed = random.choice(CAT_BREEDS)
        gender = random.choice(["Male", "Female"])
        age = round(random.uniform(*disease_info["age_range"]), 1)
        
        # Weight based on breed and age
        if breed in ["Maine Coon", "Ragdoll"]:
            base_weight = random.uniform(5, 9)
        elif breed in ["Siamese", "Abyssinian"]:
            base_weight = random.uniform(3, 5)
        else:
            base_weight = random.uniform(3.5, 6.5)
        
        # Adjust weight for disease (weight loss common)
        if selected_disease in ["Hyperthyroidism", "Chronic Kidney Disease", "Diabetes Mellitus"]:
            weight = round(base_weight * random.uniform(0.7, 0.9), 1)
        else:
            weight = round(base_weight * random.uniform(0.9, 1.1), 1)
        
        # Generate symptoms
        primary_symptom = random.choice(disease_info["symptoms"]["primary"])
        secondary_symptom = random.choice(disease_info["symptoms"]["secondary"]) if random.random() > 0.3 else ""
        
        # Optional additional symptoms
        all_symptoms = disease_info["symptoms"]["primary"] + disease_info["symptoms"]["secondary"]
        symptom_3 = random.choice(all_symptoms) if random.random() > 0.6 else ""
        symptom_4 = random.choice(all_symptoms) if random.random() > 0.8 else ""
        
        # Duration
        duration = random.randint(*disease_info["duration"])
        
        # Binary symptoms
        binary_symptoms = {}
        for symptom, prob in disease_info["symptoms"]["common"].items():
            binary_symptoms[symptom] = "Yes" if random.random() < prob else "No"
        
        # Body temperature
        temp = round(random.uniform(*disease_info["temp_range"]), 1)
        
        # Heart rate (cats: 120-220 bpm, higher when stressed/sick)
        if selected_disease in ["Hyperthyroidism", "Panleukopenia"]:
            heart_rate = random.randint(180, 240)
        else:
            heart_rate = random.randint(120, 200)
        
        # Create record
        record = {
            "Animal_Type": "Cat",
            "Breed": breed,
            "Age": age,
            "Gender": gender,
            "Weight": weight,
            "Symptom_1": primary_symptom,
            "Symptom_2": secondary_symptom,
            "Symptom_3": symptom_3,
            "Symptom_4": symptom_4,
            "Duration": f"{duration} days",
            **binary_symptoms,
            "Body_Temperature": f"{temp}°C",
            "Heart_Rate": heart_rate,
            "Disease_Prediction": selected_disease
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# ============================================================
# GENERATE AND SAVE
# ============================================================

print("🐱 Generating IMPROVED synthetic cat disease data with 12 diseases...")
print("=" * 60)

cat_df = generate_cat_data(2000)

print(f"\n✅ Generated {len(cat_df)} cat records")
print(f"\n📊 Disease distribution (12 diseases):")
print(cat_df["Disease_Prediction"].value_counts())

print(f"\n✔️  Verify Gastroenteritis is present: {'YES ✅' if 'Gastroenteritis' in cat_df['Disease_Prediction'].values else 'NO ❌'}")
print(f"✔️  Verify Asthma is present: {'YES ✅' if 'Asthma' in cat_df['Disease_Prediction'].values else 'NO ❌'}")

print(f"\n📊 Breed distribution:")
print(cat_df["Breed"].value_counts().head(5))

print(f"\n📊 Age statistics:")
print(f"Mean age: {cat_df['Age'].mean():.1f} years")
print(f"Age range: {cat_df['Age'].min():.1f} - {cat_df['Age'].max():.1f} years")

# Save to CSV
output_file = "cat_disease_data.csv"
cat_df.to_csv(output_file, index=False)

print(f"\n💾 Saved to: {output_file}")
print("=" * 60)
print("\n🎯 Next steps:")
print("1. Train cat model: python train_cat_model.py")
print("2. Restart API: python api.py")
print("3. Test with GI symptoms - should predict Gastroenteritis/Panleukopenia!")
print("=" * 60)