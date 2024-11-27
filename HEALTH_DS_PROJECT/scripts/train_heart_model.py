import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("data/processed/merged_data_v2.csv", low_memory=False)
print(f"Dataset Shape: {data.shape}")

# Drop rows with NaN values
data = data.dropna()
print(f"Dataset Shape after dropping NaN rows: {data.shape}")

# Drop unnecessary columns - note both variations of BMI category
columns_to_drop = ["gender", "occupation", "bmi_category", "BMI Category", "blood_pressure"]
data = data.drop(columns=columns_to_drop, errors="ignore")
print(f"Dataset Shape after dropping unnecessary columns: {data.shape}")

# Map `sleep_disorder` to numeric values if not already encoded
sleep_disorder_map = {
    "No sleep disorder": 0,
    "Insomnia": 1,
    "Sleep Apnea": 2
}
if data["sleep_disorder"].dtype == "object":
    print("Encoding 'sleep_disorder' values...")
    data["sleep_disorder"] = data["sleep_disorder"].map(sleep_disorder_map)

# Ensure all numeric columns are float
numeric_columns = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime", 
                  "person_id", "age", "sleep_duration", "quality_of_sleep",
                  "physical_activity_level", "stress_level", "heart_rate",
                  "daily_steps", "blood_pressure_upper", "blood_pressure_lower"]
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Encode categorical columns
categorical_columns = [
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex",
    "AgeCategory", "Race", "Diabetic", "Asthma", "PhysicalActivity"
]

# Initialize encoders
encoders = {}
for col in categorical_columns:
    try:
        encoders[col] = joblib.load(f"models/{col}_encoder.pkl")
        data[col] = encoders[col].transform(data[col])
        print(f"Encoded column: {col}")
    except FileNotFoundError:
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

# Validate feature names
print(f"Feature columns before training: {X.columns.tolist()}")

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
joblib.dump(model, "models/heart_model_v6.pkl")
joblib.dump(scaler, "models/scaler_v6.pkl")
joblib.dump(encoders, "models/label_encoders_v6.pkl")
joblib.dump(list(X.columns), "models/heart_feature_names.pkl")

print("Model, scaler, and encoders saved successfully!")