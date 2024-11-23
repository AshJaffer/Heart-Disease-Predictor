import pandas as pd
import numpy as np  # Explicitly import NumPy

# Load the processed dataset
data = pd.read_csv("data/processed/processed_for_ml.csv")

# Display all column names
print("Columns in the dataset:")
print(data.columns)

# Check the distribution of the 'health_status' column
print("\nHealth Status Column Unique Values:")
print(data['health_status'].value_counts(dropna=False))

# Inspect the 'physical activity level' and 'sleep duration' columns
print("\nPhysical Activity Level Distribution:")
print(data['physical activity level'].describe())

print("\nSleep Duration Distribution:")
print(data['sleep duration'].describe())

# Check for missing values in key columns
print("\nNull Values Check:")
print(data[['physical activity level', 'sleep duration']].isnull().sum())

# Fill missing values for debugging (temporary step for inspection)
data['physical activity level'] = data['physical activity level'].fillna(0)
data['sleep duration'] = data['sleep duration'].fillna(0)

# Re-evaluate 'health_status' column logic (for validation)
data['health_status'] = np.where(
    (data['physical activity level'] > 60) & (data['sleep duration'] >= 7),
    "healthy",
    "unhealthy"
)

# Check if 'health_status' is properly populated
print("\nRevised Health Status Column Unique Values:")
print(data['health_status'].value_counts(dropna=False))
