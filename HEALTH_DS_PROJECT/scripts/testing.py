import joblib
import pandas as pd

# Load model data
model_data = joblib.load("models/heart_model_final.pkl")

# Get scaler's feature names
if hasattr(model_data['scaler'], 'feature_names_in_'):
    print("Scaler's feature names:")
    print(model_data['scaler'].feature_names_in_)

# Get model feature names
print("\nModel's feature_names from dictionary:")
print(model_data['feature_names'])

# Compare feature names from scaler vs dictionary
scaler_features = set(model_data['scaler'].feature_names_in_)
dict_features = set(model_data['feature_names'])

print("\nDifferences between scaler and dictionary features:")
print("In scaler but not in dictionary:", scaler_features - dict_features)
print("In dictionary but not in scaler:", dict_features - scaler_features)

# Create test DataFrame
features = {name: [0] for name in model_data['scaler'].feature_names_in_}
test_df = pd.DataFrame(features)

print("\nTest DataFrame columns:")
print(test_df.columns.tolist())

# Try scaling
try:
    scaled = model_data['scaler'].transform(test_df)
    print("\nScaling successful!")
except Exception as e:
    print("\nScaling failed:", str(e))