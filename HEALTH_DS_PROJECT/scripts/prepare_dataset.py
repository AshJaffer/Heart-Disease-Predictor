import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load datasets
heart_data = pd.read_csv("data/processed/heart_prepared.csv")
sleep_data = pd.read_csv("data/processed/sleep_prepared.csv")

# Prepare heart data for ML
label_encoder = LabelEncoder()
heart_data['health_status_encoded'] = label_encoder.fit_transform(heart_data['health_status'])

# Select features and target
heart_features = heart_data.drop(columns=['health_status', 'health_status_encoded'])
heart_target = heart_data['health_status_encoded']

# Save prepared heart dataset
heart_data.to_csv("data/processed/heart_for_ml.csv", index=False)

# Prepare sleep data for ML
sleep_data['health_status_encoded'] = label_encoder.fit_transform(sleep_data['health_status'])

# Select features and target
sleep_features = sleep_data.drop(columns=['health_status', 'health_status_encoded'])
sleep_target = sleep_data['health_status_encoded']

# Save prepared sleep dataset
sleep_data.to_csv("data/processed/sleep_for_ml.csv", index=False)

print("Both datasets have been prepared for machine learning.")
