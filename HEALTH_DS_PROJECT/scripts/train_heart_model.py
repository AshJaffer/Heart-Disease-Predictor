import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load dataset
data = pd.read_csv("data/processed/heart_for_ml.csv")
print("Dataset Shape:", data.shape)

# Identify categorical columns
categorical_columns = ['heartdisease', 'smoking', 'alcoholdrinking', 'stroke', 
                       'diffwalking', 'sex', 'agecategory', 'race', 
                       'diabetic', 'physicalactivity', 'genhealth', 
                       'asthma', 'kidneydisease', 'skincancer', 'health_status']
print("Categorical Columns:", categorical_columns)

# Encode categorical variables
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Save feature names for dashboard compatibility
feature_names = list(data.columns.drop("heartdisease"))

# Define features and target
X = data.drop(columns=["heartdisease"])
y = data["heartdisease"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape} Test shape: {X_test.shape}")

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights_dict)

# Train Random Forest model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
model.fit(X_train_scaled, y_train_smote)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model, preprocessor, and encoders
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

print("Saving the model and preprocessor...")
joblib.dump(model, os.path.join(output_dir, "heart_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "heart_scaler.pkl"))
joblib.dump(label_encoders, os.path.join(output_dir, "heart_label_encoders.pkl"))
joblib.dump(feature_names, os.path.join(output_dir, "heart_feature_names.pkl"))
print("Model, preprocessor, and label encoders saved successfully!")
