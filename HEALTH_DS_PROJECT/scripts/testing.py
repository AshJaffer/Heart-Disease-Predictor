import joblib

# Load the trained model
model = joblib.load("models/heart_model_final.pkl")  # Adjust the path if needed

# Print the feature names
print("Feature names used during model training:")
if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)  # Most models trained with Scikit-learn have this attribute
else:
    print("The model does not have 'feature_names_in_' attribute. Ensure feature names were saved during training.")
