import pandas as pd

# Function to map exact age to AgeCategory
def map_age_to_category(age):
    if 18 <= age <= 24:
        return "18-24"
    elif 25 <= age <= 29:
        return "25-29"
    elif 30 <= age <= 34:
        return "30-34"
    elif 35 <= age <= 39:
        return "35-39"
    elif 40 <= age <= 44:
        return "40-44"
    elif 45 <= age <= 49:
        return "45-49"
    elif 50 <= age <= 54:
        return "50-54"
    elif 55 <= age <= 59:
        return "55-59"
    elif 60 <= age <= 64:
        return "60-64"
    elif 65 <= age <= 69:
        return "65-69"
    elif 70 <= age <= 74:
        return "70-74"
    elif 75 <= age <= 79:
        return "75-79"
    else:
        return "80 or older"

# Clean and save Heart dataset
def clean_heart_data(input_path, output_path):
    heart_data = pd.read_csv(input_path)
    heart_data.columns = heart_data.columns.str.lower().str.strip()
    heart_data = heart_data.rename(columns={"sex": "sex", "agecategory": "agecategory"})
    heart_data.to_csv(output_path, index=False)
    print(f"Cleaned Heart Data saved to {output_path}")

# Clean and save Sleep dataset
def clean_sleep_data(input_path, output_path):
    sleep_data = pd.read_csv(input_path)
    sleep_data.columns = sleep_data.columns.str.lower().str.strip()
    sleep_data = sleep_data.rename(columns={"gender": "sex"})
    sleep_data["sex"] = sleep_data["sex"].str.lower().str.strip()
    sleep_data["agecategory"] = sleep_data["age"].apply(map_age_to_category)
    sleep_data.to_csv(output_path, index=False)
    print(f"Cleaned Sleep Data saved to {output_path}")

# Merge datasets
def merge_datasets(heart_data_path, sleep_data_path, output_path):
    heart_data = pd.read_csv(heart_data_path)
    sleep_data = pd.read_csv(sleep_data_path)
    merged_data = pd.merge(heart_data, sleep_data, on=["sex", "agecategory"], how="inner")
    merged_data.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path}")
    print(f"Merged dataset shape: {merged_data.shape}")

# Main execution
if __name__ == "__main__":
    print("Cleaning Heart Data...")
    clean_heart_data("data/raw/heart_2020_cleaned.csv", "data/processed/heart_cleaned.csv")

    print("Cleaning Sleep Data...")
    clean_sleep_data("data/raw/Sleep_health_and_lifestyle_dataset.csv", "data/processed/sleep_cleaned.csv")

    print("Merging Datasets...")
    merge_datasets("data/processed/heart_cleaned.csv", "data/processed/sleep_cleaned.csv", "data/processed/merged_data.csv")

    print("Data Preparation Completed.")
