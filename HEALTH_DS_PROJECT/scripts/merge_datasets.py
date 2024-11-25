import pandas as pd

# Load raw datasets
heart_data = pd.read_csv("data/raw/heart_2020_cleaned.csv")
sleep_data = pd.read_csv("data/raw/sleep_health_and_lifestyle_dataset.csv")

# Merge datasets horizontally (assuming no common key)
# If a common key exists, use merge() instead of concat()
merged_data = pd.concat([heart_data, sleep_data], axis=1)

# Save the merged dataset
output_path = "data/processed/merged_data.csv"
merged_data.to_csv(output_path, index=False)
print(f"Merged dataset saved to {output_path}")

# Display basic info and missing values summary
print("Merged dataset preview:")
print(merged_data.head())
print("\nMissing values summary:")
print(merged_data.isna().sum())
