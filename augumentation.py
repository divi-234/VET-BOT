# augment_dataset.py
import pandas as pd
import numpy as np
import random
import os

INPUT = "cow_dog_data_augmented_v4.csv"
OUTPUT = "cow_dog_data_augmented_final.csv"
TARGET_SIZE = 3000  # total desired rows
random.seed(42)
np.random.seed(42)

if not os.path.exists(INPUT):
    raise SystemExit(f"❌ File not found: {INPUT}")

df = pd.read_csv(INPUT, encoding="latin1")
print(f"✅ Loaded {len(df)} rows from {INPUT}")

# --- basic cleanup
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# fill missing
for c in ["Breed", "Gender", "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]:
    if c in df.columns:
        df[c] = df[c].fillna("<missing>").astype(str).str.strip().str.title()

# normalize some text columns
df["Animal_Type"] = df["Animal_Type"].str.title().replace({"Canine": "Dog", "Bovine": "Cow"})

# --- separate cow & dog datasets for logical augmentation
df_cow = df[df["Animal_Type"] == "Cow"].copy()
df_dog = df[df["Animal_Type"] == "Dog"].copy()

# --- helper functions
def vary_numeric(val, pct=0.1):
    """Add ±10% random noise."""
    try:
        v = float(val)
        return round(v * np.random.uniform(1 - pct, 1 + pct), 2)
    except:
        return val

def random_swap(symptoms):
    """Swap or slightly mutate symptom list."""
    s = [s for s in symptoms if s.lower() != "none" and s != "<missing>"]
    if not s: return ["<missing>", "<missing>", "<missing>", "<missing>"]
    n = len(s)
    # random dropout / reorder
    random.shuffle(s)
    if np.random.rand() < 0.3 and n > 1:
        s[random.randint(0, n - 1)] = "<missing>"
    while len(s) < 4:
        s.append("<missing>")
    return s[:4]

def synthesize(df_part, num_new):
    """Generate new samples from given subset."""
    diseases = df_part["Disease_Prediction"].unique().tolist()
    breeds = df_part["Breed"].unique().tolist()
    gender_opts = df_part["Gender"].unique().tolist()
    new_rows = []
    for _ in range(num_new):
        row = df_part.sample(1).iloc[0].copy()
        row["Breed"] = random.choice(breeds)
        row["Gender"] = random.choice(gender_opts)
        # vary numerics
        for ncol in ["Age", "Weight", "Duration", "Body_Temperature", "Heart_Rate"]:
            if ncol in df_part.columns:
                row[ncol] = vary_numeric(row[ncol])
        # vary symptoms
        syms = [row.get(f"Symptom_{i}", "<missing>") for i in range(1, 5)]
        s_new = random_swap(syms)
        for i, s in enumerate(s_new, start=1):
            row[f"Symptom_{i}"] = s
        new_rows.append(row)
    return pd.DataFrame(new_rows)

# --- balance dataset
current_size = len(df)
if current_size >= TARGET_SIZE:
    print(f"⚠️ Already has {current_size} rows — skipping augmentation.")
    df_final = df
else:
    per_animal = TARGET_SIZE // 2
    cow_needed = max(0, per_animal - len(df_cow))
    dog_needed = max(0, per_animal - len(df_dog))

    print(f"Generating {cow_needed} new cow rows and {dog_needed} new dog rows...")

    cows_aug = synthesize(df_cow, cow_needed)
    dogs_aug = synthesize(df_dog, dog_needed)
    df_final = pd.concat([df, cows_aug, dogs_aug], ignore_index=True)

# --- shuffle and reset
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✅ Final dataset: {len(df_final)} rows")

df_final.to_csv(OUTPUT, index=False)
print(f"💾 Saved augmented dataset to: {OUTPUT}")
