import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the cleaned dataset
data = pd.read_csv("data/processed/heart_cleaned.csv")
print(f"Dataset Shape: {data.shape}")

# Categorical columns to encode
categorical_columns = ['heartdisease', 'smoking', 'alcoholdrinking', 'stroke', 
                       'diffwalking', 'sex', 'agecategory', 'race', 'diabetic']

# Encode categorical variables
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features (X) and target (y)
X = data.drop(columns=["heartdisease"])  # Drop the target variable from features
y = data["heartdisease"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_smote)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model and preprocessing artifacts
joblib.dump(model, "models/heart_model.pkl")
joblib.dump(scaler, "models/heart_scaler.pkl")
joblib.dump(label_encoders, "models/heart_label_encoders.pkl")
print("Model and preprocessing artifacts saved successfully.")
