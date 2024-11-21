import pandas as pd
from scipy.stats import ttest_ind, pearsonr

# Load Heart Dataset
heart_data = pd.read_csv("data/processed/heart_cleaned.csv")

# T-test for Physical Activity and Mental Health
active = heart_data[heart_data["physical_activity_binary"] == 1]["MentalHealth"]
inactive = heart_data[heart_data["physical_activity_binary"] == 0]["MentalHealth"]

t_stat, p_val = ttest_ind(active, inactive)
print(f"T-test Results for Heart Data:")
print(f"T-statistic: {t_stat}, P-value: {p_val}")

# Load Sleep Dataset
sleep_data = pd.read_csv("data/processed/sleep_cleaned.csv")

# Correlation between Physical Activity and Sleep Quality
correlation, p_val = pearsonr(sleep_data["physical_activity_level"], sleep_data["quality_of_sleep"])
print(f"Correlation between Physical Activity and Sleep Quality:")
print(f"Correlation Coefficient: {correlation}, P-value: {p_val}")

# Correlation between Physical Activity and Stress Level
correlation, p_val = pearsonr(sleep_data["physical_activity_level"], sleep_data["stress_level"])
print(f"Correlation between Physical Activity and Stress Level:")
print(f"Correlation Coefficient: {correlation}, P-value: {p_val}")
