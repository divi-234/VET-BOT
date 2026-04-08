import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 🩺 Load your dataset
df = pd.read_csv("cow_dog_data_augmented_final.csv")
print("✅ Dataset loaded successfully!")
print("Columns:", df.columns.tolist())

# 🧹 Drop missing target rows
df = df.dropna(subset=["Disease_Prediction"])

# 🎯 Define features and target
X = df.drop(columns=["Disease_Prediction"])
y = df["Disease_Prediction"]

# 🔠 Label encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode the target variable too
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# 🧩 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ Train the XGBoost model
print("🚀 Training model...")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# 💾 Save model and encoders
with open("vet_disease_xgb.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

# Also save the feature order
with open("model_features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\n🎉 All files saved successfully!")
print("Files generated:")
print("- vet_disease_xgb.pkl")
print("- label_encoders.pkl")
print("- target_encoder.pkl")
print("- model_features.pkl")
