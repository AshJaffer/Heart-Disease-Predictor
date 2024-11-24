import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, encoders, and feature names
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/heart_scaler.pkl")
encoders = joblib.load("models/heart_label_encoders.pkl")
feature_names = joblib.load("models/heart_feature_names.pkl")

# Title of the app
st.title("Heart Disease Prediction")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Collecting user inputs
input_data = {
    "bmi": st.sidebar.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1),
    "smoking": st.sidebar.selectbox("Smoking", options=["No", "Yes"]),
    "alcoholdrinking": st.sidebar.selectbox("Alcohol Drinking", options=["No", "Yes"]),
    "stroke": st.sidebar.selectbox("Stroke", options=["No", "Yes"]),
    "diffwalking": st.sidebar.selectbox("Difficulty Walking", options=["No", "Yes"]),
    "physicalactivity": st.sidebar.slider("Physical Activity Level", min_value=0.0, max_value=100.0, value=30.0, step=0.1),
    "genhealth": st.sidebar.selectbox("General Health", options=["Poor", "Fair", "Good", "Very good", "Excellent"]),
    "asthma": st.sidebar.selectbox("Asthma", options=["No", "Yes"]),
    "kidneydisease": st.sidebar.selectbox("Kidney Disease", options=["No", "Yes"]),
    "skincancer": st.sidebar.selectbox("Skin Cancer", options=["No", "Yes"]),
    "agecategory": st.sidebar.selectbox("Age Category", options=[
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]),
    "race": st.sidebar.selectbox("Race", options=[
        "White", "Black", "Asian", "American Indian/Alaskan Native",
        "Other", "Hispanic"]),
    "sex": st.sidebar.selectbox("Sex", options=["Female", "Male"]),
    "health_status": st.sidebar.selectbox("Health Status", options=["healthy", "moderate", "unhealthy"]),
    "diabetic": st.sidebar.selectbox("Diabetic", options=["No", "Yes"]),
    "mentalhealth": st.sidebar.slider("Mental Health (days)", min_value=0, max_value=30, value=0, step=1),
    "physicalhealth": st.sidebar.slider("Physical Health (days)", min_value=0, max_value=30, value=0, step=1),
    "sleeptime": st.sidebar.slider("Sleep Time (hours)", min_value=0, max_value=24, value=7, step=1)
}

# Convert user inputs to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical inputs
for feature, encoder in encoders.items():
    if feature in input_df.columns:
        try:
            input_df[feature] = encoder.transform(input_df[feature])
        except ValueError:
            st.error(f"Invalid input for {feature}. Resetting to default.")
            input_df[feature] = encoder.transform([encoder.classes_[0]])

# Ensure columns match the model's training data
for feature in feature_names:
    if feature not in input_df.columns:
        input_df[feature] = 0  # Add missing features with default value (e.g., 0)

# Remove extra columns not in feature_names
input_df = input_df[feature_names]

# Scale numerical features
scaled_input = scaler.transform(input_df)

# Make a prediction
st.header("Heart Disease Prediction")
try:
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    if prediction[0] == 1:
        st.error(f"Prediction: Heart Disease Likely (Confidence: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"Prediction: No Heart Disease Likely (Confidence: {prediction_proba[0][0]:.2f})")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
