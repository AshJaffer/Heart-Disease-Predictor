import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load cleaned dataset
data = pd.read_csv("data/processed/heart_cleaned_v2.csv")

# Standardize column names
data.columns = data.columns.str.lower()

# Identify categorical columns
categorical_columns = ['heartdisease', 'smoking', 'alcoholdrinking', 'stroke', 'diffwalking', 'sex',
                       'agecategory', 'race', 'diabetic', 'asthma']

# Encode categorical variables
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Convert 'physicalactivity' to numeric if not already
if data["physicalactivity"].dtype == "object":
    data["physicalactivity"] = data["physicalactivity"].map({"Yes": 1, "No": 0})

# Split data into features (X) and target (y)
X = data.drop(columns=["heartdisease"])
y = data["heartdisease"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
model.fit(X_train_smote, y_train_smote)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model, scalers, and label encoders
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, os.path.join(output_dir, "heart_model.pkl"))
joblib.dump(label_encoders, os.path.join(output_dir, "heart_label_encoders.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(output_dir, "heart_feature_names.pkl"))
print("Model, label encoders, and feature names saved successfully!")
