import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("data/processed/merged_data_v2.csv")
print(f"Dataset Shape: {data.shape}")

# Drop rows with NaN values
data = data.dropna()
print(f"Dataset Shape after dropping NaN rows: {data.shape}")

# Drop unnecessary columns
columns_to_drop = ["gender", "occupation"]
data = data.drop(columns=columns_to_drop, errors="ignore")

# Encode categorical columns
categorical_columns = [
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex",
    "AgeCategory", "Race", "Diabetic", "Asthma", "PhysicalActivity",
    "bmi_category", "sleep_disorder"
]

# Initialize encoders
encoders = {}
for col in categorical_columns:
    try:
        # Load existing encoders
        encoders[col] = joblib.load(f"models/{col}_encoder.pkl")
        data[col] = encoders[col].transform(data[col])
        print(f"Encoded column: {col}")
    except FileNotFoundError:
        # Create and save new encoder if not found
        print(f"Encoder for {col} not found. Creating a new one.")
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder
        joblib.dump(encoder, f"models/{col}_encoder.pkl")

# Encode target column
target_column = "HeartDisease"
if target_column in data.columns:
    try:
        target_encoder = joblib.load(f"models/{target_column}_encoder.pkl")
        data[target_column] = target_encoder.transform(data[target_column])
    except FileNotFoundError:
        print(f"Encoder for {target_column} not found. Creating a new one.")
        target_encoder = LabelEncoder()
        data[target_column] = target_encoder.fit_transform(data[target_column])
        joblib.dump(target_encoder, f"models/{target_column}_encoder.pkl")
else:
    raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

# Define features and target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape} Test shape: {X_test.shape}")

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE applied. Train shape: {X_train_smote.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train_smote)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model, scaler, and encoders
joblib.dump(model, "models/heart_model_v5.pkl")
joblib.dump(scaler, "models/scaler_v5.pkl")
joblib.dump(encoders, "models/label_encoders_v5.pkl")
joblib.dump(list(X.columns), "models/heart_feature_names.pkl")

print("Model, scaler, and encoders saved successfully!")
