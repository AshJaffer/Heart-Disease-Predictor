import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("data/processed/merged_data.csv")
print(f"Initial dataset shape: {data.shape}")

# Drop unnecessary columns
columns_to_remove = ['genhealth', 'health_status', 'kidneydisease', 'skincancer']
data = data.drop(columns=columns_to_remove, errors='ignore')
print(f"Dataset shape after dropping low-impact columns: {data.shape}")

# Handle missing values
data = data.dropna()  # Drop rows with NaN values
print(f"Dataset shape after dropping rows with missing values: {data.shape}")

# Save the cleaned dataset
data.to_csv("data/processed/heart_cleaned.csv", index=False)
print("Data preparation complete. Cleaned dataset saved.")
