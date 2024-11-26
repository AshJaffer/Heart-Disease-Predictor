import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load merged dataset
health_data = pd.read_csv("data/processed/merged_data.csv")
print(f"Initial health dataset shape: {health_data.shape}")

# Drop irrelevant or low-impact columns
columns_to_drop = [
    "Person ID", "Gender", "Age", "Occupation",
    "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "BMI Category", "Blood Pressure", "Heart Rate",
    "Daily Steps", "Sleep Disorder", "GenHealth", "KidneyDisease", "SkinCancer"
]
health_data = health_data.drop(columns=columns_to_drop, errors="ignore")
print(f"Health dataset shape after dropping low-impact columns: {health_data.shape}")

# Handle missing values in critical columns for health data
key_columns_health = ["BMI", "PhysicalActivity", "SleepTime"]
print(f"Missing values in key columns (health data) before drop:\n{health_data[key_columns_health].isna().sum()}")

health_data = health_data.dropna(subset=key_columns_health)
print(f"Health dataset shape after dropping rows with missing values: {health_data.shape}")

# Load and preprocess sleep dataset
print("Loading and preprocessing the sleep dataset...")
sleep_data = pd.read_csv("data/raw/sleep_health_and_lifestyle_dataset.csv")
sleep_data = sleep_data.rename(columns=lambda x: x.strip().replace(" ", "_").lower())

# Preprocess categorical variables in sleep dataset
label_enc = LabelEncoder()
for col in ["gender", "occupation", "bmi_category", "sleep_disorder"]:
    sleep_data[col] = label_enc.fit_transform(sleep_data[col])

# Split 'blood_pressure' into systolic and diastolic values
bp_split = sleep_data["blood_pressure"].str.split("/", expand=True)
sleep_data["blood_pressure_upper"] = pd.to_numeric(bp_split[0], errors="coerce")
sleep_data["blood_pressure_lower"] = pd.to_numeric(bp_split[1], errors="coerce")
sleep_data = sleep_data.drop(columns=["blood_pressure"], errors="ignore")

# Drop or impute missing values in sleep dataset
sleep_data.fillna(sleep_data.median(), inplace=True)
print(f"Sleep dataset shape after preprocessing: {sleep_data.shape}")

# Save cleaned sleep data
sleep_cleaned_path = "data/processed/sleep_cleaned.csv"
sleep_data.to_csv(sleep_cleaned_path, index=False)
print(f"Cleaned sleep dataset saved to '{sleep_cleaned_path}'.")

# Optionally merge health and sleep datasets
merged_data = pd.concat([health_data, sleep_data], axis=1)
print(f"Merged dataset shape: {merged_data.shape}")

# Save merged dataset
merged_output_path = "data/processed/merged_data_v2.csv"
merged_data.to_csv(merged_output_path, index=False)
print(f"Merged dataset saved to '{merged_output_path}'.")
