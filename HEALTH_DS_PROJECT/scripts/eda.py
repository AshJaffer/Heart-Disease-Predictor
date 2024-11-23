import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the plots directory exists
os.makedirs("plots", exist_ok=True)

# Helper function to standardize column names
def standardize_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# Load datasets and standardize column names
heart_data = pd.read_csv("data/processed/heart_cleaned.csv")
heart_data = standardize_column_names(heart_data)

sleep_data = pd.read_csv("data/processed/sleep_cleaned.csv")
sleep_data = standardize_column_names(sleep_data)

merged_data = pd.read_csv("data/processed/merged_data.csv")
merged_data = standardize_column_names(merged_data)

# EDA for Heart Data
print("Heart Data Preview:")
print(heart_data.head())
sns.histplot(heart_data['bmi'], kde=True, bins=30)
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.savefig("plots/bmi_distribution.png")
plt.close()

# EDA for Sleep Data
print("Sleep Data Preview:")
print(sleep_data.head())
sns.boxplot(x="bmi_category", y="heart_rate", data=sleep_data)
plt.title("Heart Rate by BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Heart Rate")
plt.savefig("plots/heart_rate_bmi.png")
plt.close()

# Correlation Heatmap for Merged Data
if not merged_data.empty:
    numerical_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(merged_data[numerical_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()
else:
    print("Merged dataset is empty. Skipping merged data analysis.")

print("EDA finished.")

# Retain the existing EDA code here (dataset preview, initial plots, etc.)

# Add the new code for feature distribution, correlation, and categorical analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Processed Dataset
data = pd.read_csv("data/processed/merged_data.csv")

# New code for distributions, class imbalance, correlations, etc.
numerical_features = ['bmi', 'physical activity level', 'sleeptime', 'quality of sleep']
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.savefig(f"plots/{feature}_distribution.png")
    plt.show()

# Class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='health_status', data=data)
plt.title("Distribution of Health Status")
plt.savefig("plots/health_status_distribution.png")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("plots/correlation_matrix.png")
plt.show()

# Categorical analysis
categorical_features = ['genhealth', 'smoking', 'alcoholdrinking']
for feature in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=feature, hue='health_status', data=data)
    plt.title(f"{feature} vs Health Status")
    plt.savefig(f"plots/{feature}_vs_health_status.png")
    plt.show()


