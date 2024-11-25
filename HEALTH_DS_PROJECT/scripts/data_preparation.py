import pandas as pd

# Load merged dataset
data = pd.read_csv("data/processed/merged_data.csv")
print(f"Initial dataset shape: {data.shape}")

# Drop irrelevant or low-impact columns
columns_to_drop = [
    "Person ID", "Gender", "Age", "Occupation",
    "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "BMI Category", "Blood Pressure", "Heart Rate",
    "Daily Steps", "Sleep Disorder", "GenHealth", "KidneyDisease", "SkinCancer"
]
data = data.drop(columns=columns_to_drop, errors="ignore")
print(f"Dataset shape after dropping low-impact columns: {data.shape}")

# Handle missing values in critical columns
key_columns = ["BMI", "PhysicalActivity", "SleepTime"]
print(f"Missing values in key columns before drop:\n{data[key_columns].isna().sum()}")

# Drop rows with missing values in key columns
data = data.dropna(subset=key_columns)
print(f"Dataset shape after dropping rows with missing values: {data.shape}")

# Save cleaned dataset
output_path = "data/processed/heart_cleaned_v2.csv"
data.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to '{output_path}'")
